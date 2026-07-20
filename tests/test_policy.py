from __future__ import annotations

from typing import Any, Callable

import pytest
from pydantic import BaseModel

from agent import ApprovalRequest, PolicyAction, ToolPolicy
from agent.loop import Agent
from agent.tools import Tool, ToolPermission, ToolRegistry, build_default_registry
from repl import console_approve

from .fakes import FakeClient, response, text_block, tool_use_block


class ActionArgs(BaseModel):
    value: str


def make_tool(
    name: str,
    func: Callable[[ActionArgs], str],
    permission: ToolPermission = ToolPermission.READ,
) -> Tool[ActionArgs]:
    return Tool(
        name=name,
        description=f"测试工具 {name}",
        args_model=ActionArgs,
        func=func,
        permission=permission,
    )


def make_agent(
    responses: list[Any],
    registry: ToolRegistry,
    policy: ToolPolicy | None = None,
) -> tuple[Agent, FakeClient]:
    client = FakeClient(responses)
    agent = Agent(
        model="test-model",
        verbose=False,
        registry=registry,
        client=client,  # type: ignore[arg-type]
        policy=policy,
    )
    return agent, client


def test_default_registry_marks_protected_tools() -> None:
    registry = build_default_registry()

    assert registry.get("calculator").permission == ToolPermission.READ
    assert registry.get("read_file").permission == ToolPermission.READ
    assert registry.get("write_file").permission == ToolPermission.WRITE
    assert registry.get("delete_file").permission == ToolPermission.DELETE
    assert registry.get("web_search").permission == ToolPermission.NETWORK


def test_read_tool_bypasses_approval() -> None:
    executed: list[str] = []

    def unexpected_approval(request: ApprovalRequest) -> bool:
        raise AssertionError(f"只读工具不应审批：{request.tool_name}")

    registry = ToolRegistry()
    registry.register(make_tool("read", lambda args: executed.append(args.value) or "ok"))
    agent, _ = make_agent(
        [
            response("tool_use", tool_use_block("call-read", "read", {"value": "x"})),
            response("end_turn", text_block("完成")),
        ],
        registry,
        ToolPolicy(approver=unexpected_approval),
    )

    assert agent.run("读取") == "完成"
    assert executed == ["x"]


def test_approved_protected_tool_executes() -> None:
    requests: list[ApprovalRequest] = []
    executed: list[str] = []

    def approve(request: ApprovalRequest) -> bool:
        requests.append(request)
        return True

    registry = ToolRegistry()
    registry.register(
        make_tool(
            "write",
            lambda args: executed.append(args.value) or "written",
            ToolPermission.WRITE,
        )
    )
    agent, client = make_agent(
        [
            response("tool_use", tool_use_block("call-write", "write", {"value": "data"})),
            response("end_turn", text_block("写入完成")),
        ],
        registry,
        ToolPolicy(approver=approve),
    )

    assert agent.run("写入") == "写入完成"
    assert executed == ["data"]
    assert requests == [
        ApprovalRequest(
            tool_use_id="call-write",
            tool_name="write",
            permission=ToolPermission.WRITE,
            arguments={"value": "data"},
        )
    ]
    result = client.messages.calls[1]["messages"][-1]["content"][0]
    assert result == {
        "type": "tool_result",
        "tool_use_id": "call-write",
        "content": "written",
    }


def test_denied_tool_is_not_executed_and_returns_error_result() -> None:
    executed: list[str] = []
    registry = ToolRegistry()
    registry.register(
        make_tool(
            "delete",
            lambda args: executed.append(args.value) or "deleted",
            ToolPermission.DELETE,
        )
    )
    agent, client = make_agent(
        [
            response("tool_use", tool_use_block("call-delete", "delete", {"value": "a.txt"})),
            response("end_turn", text_block("已停止")),
        ],
        registry,
        ToolPolicy(approver=lambda request: False),
    )

    assert agent.run("删除") == "已停止"
    assert executed == []
    result = client.messages.calls[1]["messages"][-1]["content"][0]
    assert result["tool_use_id"] == "call-delete"
    assert result["is_error"] is True
    assert "用户拒绝执行工具 delete" in result["content"]


def test_protected_tool_without_approver_fails_closed() -> None:
    executed: list[str] = []
    registry = ToolRegistry()
    registry.register(
        make_tool(
            "write",
            lambda args: executed.append(args.value) or "written",
            ToolPermission.WRITE,
        )
    )
    agent, client = make_agent(
        [
            response("tool_use", tool_use_block("call-write", "write", {"value": "x"})),
            response("end_turn", text_block("未执行")),
        ],
        registry,
    )

    assert agent.run("写入") == "未执行"
    assert executed == []
    result = client.messages.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert "未配置审批器" in result["content"]


def test_approval_exception_fails_closed() -> None:
    executed: list[str] = []

    def broken_approver(request: ApprovalRequest) -> bool:
        raise RuntimeError("approval unavailable")

    registry = ToolRegistry()
    registry.register(
        make_tool(
            "write",
            lambda args: executed.append(args.value) or "written",
            ToolPermission.WRITE,
        )
    )
    agent, client = make_agent(
        [
            response("tool_use", tool_use_block("call-write", "write", {"value": "x"})),
            response("end_turn", text_block("未执行")),
        ],
        registry,
        ToolPolicy(approver=broken_approver),
    )

    assert agent.run("写入") == "未执行"
    assert executed == []
    result = client.messages.calls[1]["messages"][-1]["content"][0]
    assert "工具审批失败：RuntimeError: approval unavailable" in result["content"]


def test_approver_cannot_mutate_executed_arguments() -> None:
    executed: list[str] = []

    def mutate_copy(request: ApprovalRequest) -> bool:
        request.arguments["value"] = "changed"
        return True

    registry = ToolRegistry()
    registry.register(
        make_tool(
            "write",
            lambda args: executed.append(args.value) or "written",
            ToolPermission.WRITE,
        )
    )
    agent, _ = make_agent(
        [
            response(
                "tool_use",
                tool_use_block("call-write", "write", {"value": "original"}),
            ),
            response("end_turn", text_block("完成")),
        ],
        registry,
        ToolPolicy(approver=mutate_copy),
    )

    assert agent.run("写入") == "完成"
    assert executed == ["original"]


def test_mixed_batch_preserves_result_order() -> None:
    executed: list[str] = []

    def approve_selected(request: ApprovalRequest) -> bool:
        return request.tool_name == "write"

    registry = ToolRegistry()
    registry.register(make_tool("read", lambda args: executed.append("read") or "read-ok"))
    registry.register(
        make_tool(
            "write",
            lambda args: executed.append("write") or "write-ok",
            ToolPermission.WRITE,
        )
    )
    registry.register(
        make_tool(
            "delete",
            lambda args: executed.append("delete") or "delete-ok",
            ToolPermission.DELETE,
        )
    )
    agent, client = make_agent(
        [
            response(
                "tool_use",
                tool_use_block("call-read", "read", {"value": "r"}),
                tool_use_block("call-write", "write", {"value": "w"}),
                tool_use_block("call-delete", "delete", {"value": "d"}),
            ),
            response("end_turn", text_block("批量完成")),
        ],
        registry,
        ToolPolicy(approver=approve_selected),
    )

    assert agent.run("批量执行") == "批量完成"
    assert executed == ["read", "write"]
    results = client.messages.calls[1]["messages"][-1]["content"]
    assert [item["tool_use_id"] for item in results] == [
        "call-read",
        "call-write",
        "call-delete",
    ]
    assert "is_error" not in results[0]
    assert "is_error" not in results[1]
    assert results[2]["is_error"] is True


def test_non_boolean_approval_does_not_authorize() -> None:
    executed: list[str] = []
    registry = ToolRegistry()
    registry.register(
        make_tool(
            "write",
            lambda args: executed.append(args.value) or "written",
            ToolPermission.WRITE,
        )
    )
    policy = ToolPolicy(approver=lambda request: "yes")  # type: ignore[arg-type,return-value]
    agent, client = make_agent(
        [
            response("tool_use", tool_use_block("call-write", "write", {"value": "x"})),
            response("end_turn", text_block("未执行")),
        ],
        registry,
        policy,
    )

    assert agent.run("写入") == "未执行"
    assert executed == []
    result = client.messages.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True


def test_explicit_deny_does_not_call_approver() -> None:
    executed: list[str] = []

    def unexpected_approval(request: ApprovalRequest) -> bool:
        raise AssertionError("deny 规则不应询问审批器")

    policy = ToolPolicy(
        rules={ToolPermission.NETWORK: PolicyAction.DENY},
        approver=unexpected_approval,
    )
    registry = ToolRegistry()
    registry.register(
        make_tool(
            "network",
            lambda args: executed.append(args.value) or "network-ok",
            ToolPermission.NETWORK,
        )
    )
    agent, client = make_agent(
        [
            response("tool_use", tool_use_block("call-net", "network", {"value": "q"})),
            response("end_turn", text_block("已拒绝")),
        ],
        registry,
        policy,
    )

    assert agent.run("联网") == "已拒绝"
    assert executed == []
    result = client.messages.calls[1]["messages"][-1]["content"][0]
    assert "策略拒绝 network 权限" in result["content"]


def test_unknown_tool_does_not_request_approval() -> None:
    approvals: list[ApprovalRequest] = []
    agent, _ = make_agent(
        [
            response("tool_use", tool_use_block("call-missing", "missing", {})),
            response("end_turn", text_block("完成")),
        ],
        ToolRegistry(),
        ToolPolicy(approver=lambda request: approvals.append(request) or True),
    )

    assert agent.run("未知工具") == "完成"
    assert approvals == []


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("y", True), ("YES", True), ("n", False), ("", False), ("later", False)],
)
def test_console_approve_only_accepts_explicit_yes(
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
    expected: bool,
) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    request = ApprovalRequest(
        tool_use_id="call-1",
        tool_name="write_file",
        permission=ToolPermission.WRITE,
        arguments={"path": "a.txt"},
    )

    assert console_approve(request) is expected


@pytest.mark.parametrize("error", [EOFError(), KeyboardInterrupt()])
def test_console_approve_input_interrupt_is_denied(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    def interrupt(prompt: str) -> str:
        raise error

    monkeypatch.setattr("builtins.input", interrupt)
    request = ApprovalRequest(
        tool_use_id="call-1",
        tool_name="delete_file",
        permission=ToolPermission.DELETE,
        arguments={"path": "a.txt"},
    )

    assert console_approve(request) is False
