from __future__ import annotations

from collections.abc import Callable
from threading import Event
from typing import Any

import pytest
from pydantic import BaseModel

from agent import (
    Agent,
    CancellationToken,
    EventType,
    ReliabilityConfig,
    TerminationReason,
)
from agent.tools import (
    Tool,
    ToolExecutionResult,
    ToolFailureKind,
    ToolRegistry,
)

from .fakes import FakeClient, response, text_block, tool_use_block


class ValueArgs(BaseModel):
    value: str


class FakeTimer:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


ToolExecutor = Callable[
    [Tool[Any], dict[str, Any], float, CancellationToken],
    ToolExecutionResult,
]


def make_registry(*tools: Tool[Any]) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def make_tool(
    name: str,
    func: Callable[[ValueArgs], str] | None = None,
    *,
    idempotent: bool = False,
    retryable: bool = False,
) -> Tool[ValueArgs]:
    return Tool(
        name=name,
        description=f"测试工具 {name}",
        args_model=ValueArgs,
        func=func or (lambda args: args.value),
        idempotent=idempotent,
        retryable=retryable,
    )


def make_agent(
    responses: list[Any],
    *,
    reliability: ReliabilityConfig | None = None,
    registry: ToolRegistry | None = None,
    timer: FakeTimer | None = None,
    waiter: Callable[[float, CancellationToken], bool] | None = None,
    tool_executor: ToolExecutor | None = None,
    max_steps: int = 8,
) -> tuple[Agent, FakeClient, list[Any]]:
    client = FakeClient(responses)
    events: list[Any] = []
    agent = Agent(
        model="test-model",
        max_steps=max_steps,
        verbose=False,
        client=client,  # type: ignore[arg-type]
        registry=registry,
        listeners=[events.append],
        run_id_factory=lambda: "run-reliability",
        timer=timer or FakeTimer(),
        reliability=reliability,
        waiter=waiter,
        tool_executor=tool_executor,
    )
    return agent, client, events


def test_reliability_config_validates_and_caps_backoff() -> None:
    config = ReliabilityConfig(
        retry_base_delay_s=0.5,
        retry_multiplier=3,
        retry_max_delay_s=2,
    )

    assert config.retry_delay(1) == 0.5
    assert config.retry_delay(2) == 1.5
    assert config.retry_delay(3) == 2
    with pytest.raises(ValueError, match="model_max_retries"):
        ReliabilityConfig(model_max_retries=-1)
    with pytest.raises(ValueError, match="幂等"):
        make_tool("unsafe", idempotent=False, retryable=True)


def test_model_transient_errors_retry_with_exponential_backoff() -> None:
    timer = FakeTimer()
    waits: list[float] = []

    def wait(delay: float, cancellation: CancellationToken) -> bool:
        waits.append(delay)
        timer.advance(delay)
        return cancellation.is_cancelled

    config = ReliabilityConfig(
        model_max_retries=2,
        retry_base_delay_s=0.5,
        retry_multiplier=2,
    )
    agent, client, events = make_agent(
        [TimeoutError("one"), ConnectionError("two"), response("end_turn", text_block("ok"))],
        reliability=config,
        timer=timer,
        waiter=wait,
    )

    result = agent.run_with_result("重试")

    assert result.ok
    assert result.answer == "ok"
    assert result.model_attempts == 3
    assert len(client.messages.calls) == 3
    assert waits == [0.5, 1.0]
    retries = [event for event in events if event.event_type == EventType.MODEL_RETRY_SCHEDULED]
    assert [event.attempt for event in retries] == [2, 3]
    assert all(call["timeout"] <= config.model_timeout_s for call in client.messages.calls)


def test_permanent_model_error_is_not_retried_and_legacy_run_reraises() -> None:
    error = RuntimeError("bad request")
    agent, client, _ = make_agent([error])

    with pytest.raises(RuntimeError, match="bad request"):
        agent.run("失败")

    assert len(client.messages.calls) == 1
    assert agent.last_result is not None
    assert agent.last_result.termination_reason == TerminationReason.SYSTEM_ERROR


def test_cancellation_before_model_call_returns_structured_result() -> None:
    token = CancellationToken()
    token.cancel()
    agent, client, events = make_agent([])

    result = agent.run_with_result("取消", cancellation=token)

    assert result.termination_reason == TerminationReason.CANCELLED
    assert client.messages.calls == []
    assert events[-1].event_type == EventType.RUN_CANCELLED


def test_cancellation_interrupts_model_retry_wait() -> None:
    token = CancellationToken()

    def cancel_during_wait(delay: float, cancellation: CancellationToken) -> bool:
        cancellation.cancel()
        return True

    agent, client, _ = make_agent(
        [TimeoutError("offline")],
        waiter=cancel_during_wait,
    )

    result = agent.run_with_result("取消重试", cancellation=token)

    assert result.termination_reason == TerminationReason.CANCELLED
    assert len(client.messages.calls) == 1


def test_retry_is_stopped_when_backoff_would_cross_run_deadline() -> None:
    config = ReliabilityConfig(
        run_timeout_s=0.1,
        retry_base_delay_s=0.2,
        model_max_retries=2,
    )
    agent, client, _ = make_agent([TimeoutError("offline")], reliability=config)

    result = agent.run_with_result("超时")

    assert result.termination_reason == TerminationReason.TIMEOUT
    assert len(client.messages.calls) == 1


def test_token_budget_is_reported_separately() -> None:
    config = ReliabilityConfig(max_total_tokens=10)
    agent, _, _ = make_agent(
        [
            response(
                "end_turn",
                text_block("答案"),
                input_tokens=7,
                output_tokens=2,
                cache_read_input_tokens=2,
            )
        ],
        reliability=config,
    )

    result = agent.run_with_result("预算")

    assert result.termination_reason == TerminationReason.BUDGET_EXHAUSTED
    assert result.total_tokens == 11
    assert "预算" in result.answer


def test_model_output_is_truncated_and_reports_output_limit() -> None:
    config = ReliabilityConfig(max_model_output_chars=5)
    agent, _, _ = make_agent(
        [response("end_turn", text_block("123456789"))],
        reliability=config,
    )

    result = agent.run_with_result("长输出")

    assert result.termination_reason == TerminationReason.OUTPUT_LIMIT
    assert result.answer.startswith("12345")
    assert "已截断" in result.answer


def test_tool_call_limit_completes_all_tool_result_blocks() -> None:
    first = make_tool("first")
    second = make_tool("second")
    config = ReliabilityConfig(max_tool_calls=1)
    agent, _, _ = make_agent(
        [
            response(
                "tool_use",
                tool_use_block("call-1", "first", {"value": "a"}),
                tool_use_block("call-2", "second", {"value": "b"}),
            )
        ],
        reliability=config,
        registry=make_registry(first, second),
    )

    result = agent.run_with_result("两个工具")

    assert result.termination_reason == TerminationReason.TOOL_CALL_LIMIT
    assert result.tool_calls == 1
    tool_results = agent.memory.messages[-1]["content"]
    assert [item["tool_use_id"] for item in tool_results] == ["call-1", "call-2"]
    assert tool_results[1]["is_error"] is True


def test_tool_timeout_terminates_without_retrying() -> None:
    calls: list[str] = []

    def timeout_executor(
        tool: Tool[Any],
        raw_input: dict[str, Any],
        timeout_s: float,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        calls.append(tool.name)
        return ToolExecutionResult(False, "状态未知", ToolFailureKind.TIMEOUT)

    tool = make_tool("slow", idempotent=True, retryable=True)
    agent, _, events = make_agent(
        [response("tool_use", tool_use_block("call-1", "slow", {"value": "x"}))],
        registry=make_registry(tool),
        tool_executor=timeout_executor,
    )

    result = agent.run_with_result("慢工具")

    assert result.termination_reason == TerminationReason.TIMEOUT
    assert calls == ["slow"]
    assert EventType.TOOL_RETRY_SCHEDULED not in [event.event_type for event in events]
    assert EventType.TOOL_CALL_TIMED_OUT in [event.event_type for event in events]


def test_default_tool_executor_stops_waiting_after_timeout() -> None:
    release = Event()

    def slow_tool(args: ValueArgs) -> str:
        release.wait(timeout=1)
        return args.value

    tool = make_tool("slow_default", func=slow_tool, idempotent=True)
    config = ReliabilityConfig(tool_timeout_s=0.01)
    agent, _, _ = make_agent(
        [
            response(
                "tool_use",
                tool_use_block("call-1", "slow_default", {"value": "x"}),
            )
        ],
        reliability=config,
        registry=make_registry(tool),
    )

    result = agent.run_with_result("真实超时")
    release.set()

    assert result.termination_reason == TerminationReason.TIMEOUT
    assert result.tool_calls == 1
    assert "状态未知" in (result.error or "")


def test_retryable_idempotent_tool_retries_execution_error() -> None:
    outcomes = iter(
        [
            ToolExecutionResult(False, "临时失败", ToolFailureKind.EXECUTION),
            ToolExecutionResult(True, "完成"),
        ]
    )
    calls: list[str] = []
    waits: list[float] = []

    def executor(
        tool: Tool[Any],
        raw_input: dict[str, Any],
        timeout_s: float,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        calls.append(tool.name)
        return next(outcomes)

    def wait(delay: float, cancellation: CancellationToken) -> bool:
        waits.append(delay)
        return False

    tool = make_tool("lookup", idempotent=True, retryable=True)
    agent, client, events = make_agent(
        [
            response("tool_use", tool_use_block("call-1", "lookup", {"value": "x"})),
            response("end_turn", text_block("最终答案")),
        ],
        registry=make_registry(tool),
        tool_executor=executor,
        waiter=wait,
    )

    result = agent.run_with_result("查询")

    assert result.ok
    assert result.tool_calls == 2
    assert calls == ["lookup", "lookup"]
    assert waits == [0.25]
    assert client.messages.calls[1]["messages"][-1]["content"][0]["content"] == "完成"
    assert EventType.TOOL_RETRY_SCHEDULED in [event.event_type for event in events]


def test_non_idempotent_tool_failure_is_not_retried() -> None:
    calls: list[str] = []

    def executor(
        tool: Tool[Any],
        raw_input: dict[str, Any],
        timeout_s: float,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        calls.append(tool.name)
        return ToolExecutionResult(False, "写入失败", ToolFailureKind.EXECUTION)

    tool = make_tool("write_once")
    agent, client, _ = make_agent(
        [
            response("tool_use", tool_use_block("call-1", "write_once", {"value": "x"})),
            response("end_turn", text_block("已说明失败")),
        ],
        registry=make_registry(tool),
        tool_executor=executor,
    )

    result = agent.run_with_result("写入")

    assert result.ok
    assert calls == ["write_once"]
    tool_result = client.messages.calls[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True


def test_cancellation_mid_tool_batch_keeps_protocol_complete() -> None:
    token = CancellationToken()

    def cancelling_executor(
        tool: Tool[Any],
        raw_input: dict[str, Any],
        timeout_s: float,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        cancellation.cancel()
        return ToolExecutionResult(False, "已取消", ToolFailureKind.CANCELLED)

    first = make_tool("first")
    second = make_tool("second")
    agent, _, _ = make_agent(
        [
            response(
                "tool_use",
                tool_use_block("call-1", "first", {"value": "a"}),
                tool_use_block("call-2", "second", {"value": "b"}),
            )
        ],
        registry=make_registry(first, second),
        tool_executor=cancelling_executor,
    )

    result = agent.run_with_result("取消批次", cancellation=token)

    assert result.termination_reason == TerminationReason.CANCELLED
    tool_results = agent.memory.messages[-1]["content"]
    assert [item["tool_use_id"] for item in tool_results] == ["call-1", "call-2"]
    assert all(item["is_error"] for item in tool_results)


def test_tool_result_is_truncated_before_next_model_request() -> None:
    tool = make_tool("large", func=lambda args: "x" * 20)
    config = ReliabilityConfig(max_tool_result_chars=5)
    agent, client, _ = make_agent(
        [
            response("tool_use", tool_use_block("call-1", "large", {"value": "x"})),
            response("end_turn", text_block("完成")),
        ],
        reliability=config,
        registry=make_registry(tool),
    )

    result = agent.run_with_result("大结果")

    assert result.ok
    content = client.messages.calls[1]["messages"][-1]["content"][0]["content"]
    assert content.startswith("xxxxx")
    assert "已截断" in content


def test_context_compaction_uses_same_model_retry_policy() -> None:
    waits: list[float] = []

    def wait(delay: float, cancellation: CancellationToken) -> bool:
        waits.append(delay)
        return False

    agent, client, events = make_agent(
        [
            response("end_turn", text_block("最终答案"), input_tokens=2, output_tokens=1),
            TimeoutError("summary timeout"),
            response("end_turn", text_block("历史摘要"), input_tokens=4, output_tokens=2),
        ],
        waiter=wait,
    )
    for index in range(21):
        role = "user" if index % 2 == 0 else "assistant"
        agent.memory.add(role, f"历史消息 {index}")

    result = agent.run_with_result("继续")

    assert result.ok
    assert result.model_attempts == 3
    assert result.total_tokens == 9
    assert len(client.messages.calls) == 3
    assert waits == [0.25]
    event_types = [event.event_type for event in events]
    assert EventType.MODEL_RETRY_SCHEDULED in event_types
    assert EventType.CONTEXT_COMPACTED in event_types


def test_max_steps_has_structured_termination_reason() -> None:
    agent, _, _ = make_agent(
        [response("tool_use", tool_use_block("call-1", "missing", {}))],
        registry=ToolRegistry(),
        max_steps=1,
    )

    result = agent.run_with_result("循环")

    assert result.termination_reason == TerminationReason.MAX_STEPS
    assert result.steps == 1
