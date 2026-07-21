from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable

import pytest
from pydantic import BaseModel

from agent import Agent, AgentEvent, EventType, PolicyAction, ToolPolicy
from agent.events import console_event_listener, summarize_for_event
from agent.tools import Tool, ToolPermission, ToolRegistry

from .fakes import FakeClient, response, text_block, tool_use_block


class ValueArgs(BaseModel):
    value: str


class CountArgs(BaseModel):
    count: int


def make_tool(
    name: str,
    func: Callable[[Any], str],
    *,
    args_model: type[BaseModel] = ValueArgs,
    permission: ToolPermission = ToolPermission.READ,
) -> Tool[Any]:
    return Tool(
        name=name,
        description=f"测试工具 {name}",
        args_model=args_model,
        func=func,
        permission=permission,
    )


def make_agent(
    responses: list[Any],
    *,
    registry: ToolRegistry | None = None,
    policy: ToolPolicy | None = None,
    listeners: list[Callable[[AgentEvent], None]] | None = None,
    max_steps: int = 8,
    run_id_factory: Callable[[], str] | None = None,
) -> tuple[Agent, FakeClient, list[AgentEvent]]:
    client = FakeClient(responses)
    events: list[AgentEvent] = []
    configured_listeners = [events.append]
    if listeners:
        configured_listeners = listeners + configured_listeners
    agent = Agent(
        model="test-model",
        max_steps=max_steps,
        verbose=False,
        registry=registry,
        client=client,  # type: ignore[arg-type]
        policy=policy,
        listeners=configured_listeners,
        run_id_factory=run_id_factory or (lambda: "run-fixed"),
        clock=lambda: datetime(2026, 7, 20, 8, 30, tzinfo=timezone.utc),
        timer=lambda: 10.0,
    )
    return agent, client, events


def event_types(events: list[AgentEvent]) -> list[EventType]:
    return [event.event_type for event in events]


def test_text_run_emits_complete_order_and_metadata() -> None:
    agent, _, events = make_agent([response("end_turn", text_block("直接回答"))])

    assert agent.run("你好") == "直接回答"
    assert event_types(events) == [
        EventType.RUN_STARTED,
        EventType.MODEL_REQUESTED,
        EventType.MODEL_RESPONSE_RECEIVED,
        EventType.MODEL_TEXT_RECEIVED,
        EventType.RUN_FINISHED,
    ]
    assert {event.run_id for event in events} == {"run-fixed"}
    assert all(event.timestamp.utcoffset() == timezone.utc.utcoffset(None) for event in events)
    assert events[1].step == 1
    assert events[2].duration_ms == 0.0
    assert events[-1].output_summary == "直接回答"


def test_each_run_gets_new_id_and_resets_step() -> None:
    run_ids = iter(["run-1", "run-2"])
    agent, _, events = make_agent(
        [
            response("end_turn", text_block("一")),
            response("end_turn", text_block("二")),
        ],
        run_id_factory=lambda: next(run_ids),
    )

    assert agent.run("第一轮") == "一"
    assert agent.run("第二轮") == "二"
    started = [event for event in events if event.event_type == EventType.RUN_STARTED]
    requested = [event for event in events if event.event_type == EventType.MODEL_REQUESTED]
    assert [event.run_id for event in started] == ["run-1", "run-2"]
    assert [(event.run_id, event.step) for event in requested] == [
        ("run-1", 1),
        ("run-2", 1),
    ]


def test_tool_events_include_call_metadata_and_duration() -> None:
    registry = ToolRegistry()
    registry.register(make_tool("echo", lambda args: f"echo:{args.value}"))
    agent, _, events = make_agent(
        [
            response("tool_use", tool_use_block("call-1", "echo", {"value": "hi"})),
            response("end_turn", text_block("完成")),
        ],
        registry=registry,
    )

    assert agent.run("调用工具") == "完成"
    tool_events = [event for event in events if event.tool_use_id == "call-1"]
    assert event_types(tool_events) == [
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_FINISHED,
    ]
    assert all(event.tool_name == "echo" and event.step == 1 for event in tool_events)
    assert "hi" in (tool_events[0].input_summary or "")
    assert tool_events[1].output_summary == "echo:hi"
    assert tool_events[1].duration_ms == 0.0


def test_multiple_tools_preserve_event_order() -> None:
    registry = ToolRegistry()
    registry.register(make_tool("first", lambda args: "one"))
    registry.register(make_tool("second", lambda args: "two"))
    agent, _, events = make_agent(
        [
            response(
                "tool_use",
                tool_use_block("call-1", "first", {"value": "a"}),
                tool_use_block("call-2", "second", {"value": "b"}),
            ),
            response("end_turn", text_block("完成")),
        ],
        registry=registry,
    )

    assert agent.run("批量") == "完成"
    tool_events = [event for event in events if event.tool_use_id is not None]
    assert [(event.event_type, event.tool_use_id) for event in tool_events] == [
        (EventType.TOOL_CALL_STARTED, "call-1"),
        (EventType.TOOL_CALL_FINISHED, "call-1"),
        (EventType.TOOL_CALL_STARTED, "call-2"),
        (EventType.TOOL_CALL_FINISHED, "call-2"),
    ]


def test_approved_tool_emits_approval_before_execution() -> None:
    registry = ToolRegistry()
    registry.register(
        make_tool("write", lambda args: "written", permission=ToolPermission.WRITE)
    )
    agent, _, events = make_agent(
        [
            response("tool_use", tool_use_block("call-write", "write", {"value": "x"})),
            response("end_turn", text_block("完成")),
        ],
        registry=registry,
        policy=ToolPolicy(approver=lambda request: True),
    )

    assert agent.run("写入") == "完成"
    call_events = [event for event in events if event.tool_use_id == "call-write"]
    assert event_types(call_events) == [
        EventType.APPROVAL_REQUESTED,
        EventType.APPROVAL_RESOLVED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_FINISHED,
    ]
    assert "已批准" in (call_events[1].output_summary or "")


@pytest.mark.parametrize(
    ("approver", "reason"),
    [
        (lambda request: False, "用户拒绝"),
        (lambda request: (_ for _ in ()).throw(RuntimeError("offline")), "审批失败"),
    ],
)
def test_denied_or_failed_approval_emits_failure_without_start(
    approver: Callable[[Any], bool],
    reason: str,
) -> None:
    executed: list[str] = []
    registry = ToolRegistry()
    registry.register(
        make_tool(
            "delete",
            lambda args: executed.append(args.value) or "deleted",
            permission=ToolPermission.DELETE,
        )
    )
    agent, client, events = make_agent(
        [
            response("tool_use", tool_use_block("call-delete", "delete", {"value": "x"})),
            response("end_turn", text_block("停止")),
        ],
        registry=registry,
        policy=ToolPolicy(approver=approver),
    )

    assert agent.run("删除") == "停止"
    assert executed == []
    call_events = [event for event in events if event.tool_use_id == "call-delete"]
    assert event_types(call_events) == [
        EventType.APPROVAL_REQUESTED,
        EventType.APPROVAL_RESOLVED,
        EventType.TOOL_CALL_FAILED,
    ]
    assert reason in (call_events[-1].error or "")
    result = client.messages.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True


@pytest.mark.parametrize(
    ("tool", "raw_input", "error_part"),
    [
        (
            make_tool(
                "count",
                lambda args: str(args.count),
                args_model=CountArgs,
            ),
            {"count": "not-an-int"},
            "参数校验失败",
        ),
        (
            make_tool("broken", lambda args: (_ for _ in ()).throw(ValueError("boom"))),
            {"value": "x"},
            "工具执行出错",
        ),
    ],
)
def test_tool_validation_and_execution_errors_emit_failed(
    tool: Tool[Any],
    raw_input: dict[str, Any],
    error_part: str,
) -> None:
    registry = ToolRegistry()
    registry.register(tool)
    agent, _, events = make_agent(
        [
            response("tool_use", tool_use_block("call-bad", tool.name, raw_input)),
            response("end_turn", text_block("已处理")),
        ],
        registry=registry,
    )

    assert agent.run("执行") == "已处理"
    call_events = [event for event in events if event.tool_use_id == "call-bad"]
    assert event_types(call_events) == [
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_FAILED,
    ]
    assert error_part in (call_events[-1].error or "")


def test_unknown_tool_emits_failure_without_approval_or_start() -> None:
    agent, _, events = make_agent(
        [
            response("tool_use", tool_use_block("call-missing", "missing", {})),
            response("end_turn", text_block("完成")),
        ],
        registry=ToolRegistry(),
    )

    assert agent.run("未知") == "完成"
    call_events = [event for event in events if event.tool_use_id == "call-missing"]
    assert event_types(call_events) == [EventType.TOOL_CALL_FAILED]
    assert call_events[0].error == "未知工具：missing"


def test_malformed_tool_response_emits_invalid_and_recovers() -> None:
    agent, _, events = make_agent(
        [
            response("tool_use", text_block("我想调用工具")),
            response("end_turn", text_block("恢复完成")),
        ],
        registry=ToolRegistry(),
    )

    assert agent.run("继续") == "恢复完成"
    invalid = [event for event in events if event.event_type == EventType.MODEL_RESPONSE_INVALID]
    assert len(invalid) == 1
    assert invalid[0].step == 1
    assert [event.step for event in events if event.event_type == EventType.MODEL_REQUESTED] == [1, 2]


def test_compaction_event_is_between_text_and_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, _, events = make_agent([response("end_turn", text_block("完成"))])
    monkeypatch.setattr(agent.memory, "maybe_compact", lambda: True)

    assert agent.run("压缩") == "完成"
    types = event_types(events)
    assert types.index(EventType.MODEL_TEXT_RECEIVED) < types.index(EventType.CONTEXT_COMPACTED)
    assert types.index(EventType.CONTEXT_COMPACTED) < types.index(EventType.RUN_FINISHED)


def test_max_steps_emits_run_failed_and_keeps_return_value() -> None:
    agent, _, events = make_agent(
        [response("tool_use", tool_use_block("call-1", "missing", {}))],
        registry=ToolRegistry(),
        max_steps=1,
    )

    answer = agent.run("循环")

    assert answer == "（已达到最大步数上限，未能得出最终答案——可能陷入工具循环，建议检查工具描述或提高 max_steps）"
    assert events[-1].event_type == EventType.RUN_FAILED
    assert events[-1].error == "达到最大步数上限"
    assert EventType.RUN_FINISHED not in event_types(events)


def test_model_exception_emits_run_failed_and_reraises_same_object() -> None:
    failure = RuntimeError("model unavailable")
    agent, _, events = make_agent([failure])

    with pytest.raises(RuntimeError) as caught:
        agent.run("你好")

    assert caught.value is failure
    assert events[-1].event_type == EventType.RUN_FAILED
    assert events[-1].error == "RuntimeError: model unavailable"


def test_listener_failure_is_isolated_from_agent_and_later_listeners() -> None:
    received: list[AgentEvent] = []

    def broken_listener(event: AgentEvent) -> None:
        raise RuntimeError("listener failed")

    agent, _, events = make_agent(
        [response("end_turn", text_block("完成"))],
        listeners=[broken_listener, received.append],
    )

    assert agent.run("你好") == "完成"
    assert event_types(received) == event_types(events)
    assert len(received) == 5


def test_verbose_false_does_not_print_but_listener_receives_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent, _, events = make_agent([response("end_turn", text_block("静默"))])

    assert agent.run("你好") == "静默"
    assert capsys.readouterr().out == ""
    assert events


def test_console_listener_renders_structured_events(capsys: pytest.CaptureFixture[str]) -> None:
    timestamp = datetime(2026, 7, 20, tzinfo=timezone.utc)
    console_event_listener(
        AgentEvent(
            EventType.TOOL_CALL_STARTED,
            "run-1",
            timestamp,
            tool_name="echo",
            input_summary="{'value': 'x'}",
        )
    )
    console_event_listener(
        AgentEvent(
            EventType.TOOL_CALL_FINISHED,
            "run-1",
            timestamp,
            tool_name="echo",
            output_summary="ok",
        )
    )

    output = capsys.readouterr().out
    assert "🔧 调用 echo" in output
    assert "↳ ok" in output


def test_summary_redacts_nested_secrets_before_truncating_without_mutation() -> None:
    value = {
        "profile": {
            "password": "p@ssword",
            "headers": {"Authorization": "Bearer abcdefghijklmnop"},
        },
        "note": "token=hidden-value " + "x" * 300,
        "items": [{"api-key": "sk-super-secret-key"}],
    }
    original = deepcopy(value)

    summary = summarize_for_event(value, limit=120)

    assert value == original
    assert "p@ssword" not in summary
    assert "abcdefghijklmnop" not in summary
    assert "hidden-value" not in summary
    assert "sk-super-secret-key" not in summary
    assert "[REDACTED]" in summary
    assert summary.endswith("… [truncated]")


def test_sensitive_tool_hides_entire_input_and_output() -> None:
    registry = ToolRegistry()
    registry.register(
        make_tool(
            "vault",
            lambda args: "private-result",
            permission=ToolPermission.SENSITIVE,
        )
    )
    policy = ToolPolicy(rules={ToolPermission.SENSITIVE: PolicyAction.ALLOW})
    agent, _, events = make_agent(
        [
            response("tool_use", tool_use_block("call-vault", "vault", {"value": "secret"})),
            response("end_turn", text_block("完成")),
        ],
        registry=registry,
        policy=policy,
    )

    assert agent.run("读取敏感数据") == "完成"
    call_events = [event for event in events if event.tool_use_id == "call-vault"]
    assert call_events[0].input_summary == "[REDACTED]"
    assert call_events[1].output_summary == "[REDACTED]"
    assert all("secret" not in repr(event) for event in call_events)
    assert all("private-result" not in repr(event) for event in call_events)
