"""Anthropic SDK 形状的最小 Fake Client，用于测试 Agent Loop。"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from types import SimpleNamespace
from typing import Any


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use_block(
    block_id: str,
    name: str,
    input: dict[str, Any],
) -> SimpleNamespace:
    return SimpleNamespace(
        type="tool_use",
        id=block_id,
        name=name,
        input=input,
    )


def response(
    stop_reason: str,
    *blocks: SimpleNamespace,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=list(blocks),
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        ),
    )


class FakeMessages:
    """按顺序返回预置响应，并记录 messages.create 的请求。"""

    def __init__(self, responses: Iterable[SimpleNamespace | Exception]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        # Agent 会在请求返回后继续修改 memory，必须保存调用当时的快照。
        self.calls.append(deepcopy(kwargs))
        next_response = next(self._responses)
        if isinstance(next_response, Exception):
            raise next_response
        return next_response


class FakeClient:
    def __init__(self, responses: Iterable[SimpleNamespace | Exception]) -> None:
        self.messages = FakeMessages(responses)
