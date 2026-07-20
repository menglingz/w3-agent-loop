from __future__ import annotations

from typing import Any, Callable

import pytest
from pydantic import BaseModel

from agent.loop import Agent
from agent.tools import Tool, ToolRegistry

from .fakes import FakeClient, response, text_block, tool_use_block


class EchoArgs(BaseModel):
    text: str


class IntegerArgs(BaseModel):
    value: int


def registry_with(
    name: str,
    args_model: type[BaseModel],
    func: Callable[[BaseModel], str],
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool(name, f"测试工具 {name}", args_model, func))
    return registry


def make_agent(
    responses: list[Any],
    registry: ToolRegistry | None = None,
    max_steps: int = 8,
) -> tuple[Agent, FakeClient]:
    client = FakeClient(responses)
    agent = Agent(
        model="test-model",
        max_steps=max_steps,
        verbose=False,
        registry=registry,
        client=client,  # type: ignore[arg-type]
    )
    return agent, client


def test_returns_final_text_without_tools() -> None:
    agent, client = make_agent([response("end_turn", text_block("直接回答"))])

    assert agent.run("你好") == "直接回答"
    assert len(client.messages.calls) == 1
    assert client.messages.calls[0]["messages"] == [
        {"role": "user", "content": "你好"}
    ]


def test_executes_tool_and_replies_with_matching_tool_result() -> None:
    received: list[str] = []

    def echo(args: EchoArgs) -> str:
        received.append(args.text)
        return f"echo:{args.text}"

    registry = registry_with("echo", EchoArgs, echo)
    agent, client = make_agent(
        [
            response(
                "tool_use",
                tool_use_block("call-1", "echo", {"text": "hello"}),
            ),
            response("end_turn", text_block("工具完成")),
        ],
        registry,
    )

    assert agent.run("请调用 echo") == "工具完成"
    assert received == ["hello"]
    tool_message = client.messages.calls[1]["messages"][-1]
    assert tool_message == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call-1",
                "content": "echo:hello",
            }
        ],
    }


def test_executes_multiple_tools_and_batches_results() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool("first", "第一个测试工具", EchoArgs, lambda args: f"first:{args.text}")
    )
    registry.register(
        Tool("second", "第二个测试工具", EchoArgs, lambda args: f"second:{args.text}")
    )
    agent, client = make_agent(
        [
            response(
                "tool_use",
                tool_use_block("call-1", "first", {"text": "a"}),
                tool_use_block("call-2", "second", {"text": "b"}),
            ),
            response("end_turn", text_block("两个工具完成")),
        ],
        registry,
    )

    assert agent.run("请调用两个工具") == "两个工具完成"
    tool_message = client.messages.calls[1]["messages"][-1]
    assert tool_message["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "call-1",
            "content": "first:a",
        },
        {
            "type": "tool_result",
            "tool_use_id": "call-2",
            "content": "second:b",
        },
    ]


def test_unknown_tool_is_returned_to_model_without_default_registry() -> None:
    empty_registry = ToolRegistry()
    agent, client = make_agent(
        [
            response(
                "tool_use",
                tool_use_block("call-unknown", "missing", {}),
            ),
            response("end_turn", text_block("已处理未知工具")),
        ],
        empty_registry,
    )

    assert agent.registry is empty_registry
    assert agent.run("请调用不存在的工具") == "已处理未知工具"
    assert client.messages.calls[0]["tools"] == []
    assert client.messages.calls[1]["messages"][-1]["content"][0]["content"] == (
        "未知工具：missing"
    )


def test_validation_error_is_returned_to_model() -> None:
    registry = registry_with("integer", IntegerArgs, lambda args: str(args.value))
    agent, client = make_agent(
        [
            response(
                "tool_use",
                tool_use_block("call-invalid", "integer", {"value": "not-an-int"}),
            ),
            response("end_turn", text_block("参数已修正")),
        ],
        registry,
    )

    assert agent.run("调用整数工具") == "参数已修正"
    result = client.messages.calls[1]["messages"][-1]["content"][0]["content"]
    assert "参数校验失败" in result


def test_tool_exception_is_returned_to_model() -> None:
    def broken(args: EchoArgs) -> str:
        raise RuntimeError("boom")

    registry = registry_with("broken", EchoArgs, broken)
    agent, client = make_agent(
        [
            response(
                "tool_use",
                tool_use_block("call-broken", "broken", {"text": "x"}),
            ),
            response("end_turn", text_block("工具失败已处理")),
        ],
        registry,
    )

    assert agent.run("调用会失败的工具") == "工具失败已处理"
    result = client.messages.calls[1]["messages"][-1]["content"][0]["content"]
    assert "工具执行出错：RuntimeError: boom" in result


def test_malformed_tool_use_response_gets_correction_prompt() -> None:
    agent, client = make_agent(
        [
            response("tool_use", text_block("我应该调用工具")),
            response("end_turn", text_block("重新回答")),
        ],
    )

    assert agent.run("继续") == "重新回答"
    assert "stop_reason=tool_use" in client.messages.calls[1]["messages"][-1][
        "content"
    ]


def test_stops_after_max_steps() -> None:
    agent, client = make_agent(
        [
            response(
                "tool_use",
                tool_use_block("call-1", "missing", {}),
            ),
            response(
                "tool_use",
                tool_use_block("call-2", "missing", {}),
            ),
        ],
        registry=ToolRegistry(),
        max_steps=2,
    )

    result = agent.run("一直调用工具")

    assert "已达到最大步数上限" in result
    assert len(client.messages.calls) == 2


def test_model_client_exception_is_propagated() -> None:
    error = RuntimeError("model unavailable")
    agent, client = make_agent([error])

    with pytest.raises(RuntimeError, match="model unavailable"):
        agent.run("请求模型")

    assert len(client.messages.calls) == 1
