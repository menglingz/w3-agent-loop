from __future__ import annotations

from copy import deepcopy

import pytest

from agent.memory import ConversationMemory

from .fakes import FakeClient, response, text_block, tool_use_block


def make_memory(
    responses: list[object],
    *,
    max_messages: int = 20,
    keep_recent: int = 8,
) -> tuple[ConversationMemory, FakeClient]:
    client = FakeClient(responses)
    memory = ConversationMemory(
        client,
        model="test-model",
        max_messages=max_messages,
        keep_recent=keep_recent,
    )
    return memory, client


def add_plain_messages(memory: ConversationMemory, count: int) -> None:
    for index in range(count):
        memory.add("user" if index % 2 == 0 else "assistant", f"message-{index}")


def test_does_not_compact_at_threshold() -> None:
    memory, client = make_memory([], max_messages=4, keep_recent=2)
    add_plain_messages(memory, 4)
    before = deepcopy(memory.messages)

    assert memory.maybe_compact() is False
    assert memory.messages == before
    assert client.messages.calls == []


def test_compacts_plain_messages_and_writes_summary() -> None:
    memory, client = make_memory(
        [response("end_turn", text_block("保留用户正在学习 Agent Loop"))],
        max_messages=4,
        keep_recent=2,
    )
    add_plain_messages(memory, 5)

    assert memory.maybe_compact() is True
    assert len(memory.messages) == 3
    assert memory.messages[0] == {
        "role": "user",
        "content": "【以下是早前对话的摘要，供你保持上下文】\n保留用户正在学习 Agent Loop",
    }
    assert memory.messages[-2:] == [
        {"role": "assistant", "content": "message-3"},
        {"role": "user", "content": "message-4"},
    ]
    assert client.messages.calls[0]["model"] == "test-model"
    assert client.messages.calls[0]["max_tokens"] == 512
    assert len(client.messages.calls[0]["messages"]) == 1
    assert "message-0" in client.messages.calls[0]["messages"][0]["content"]
    assert "message-4" not in client.messages.calls[0]["messages"][0]["content"]


def test_keeps_tool_use_with_matching_tool_result() -> None:
    memory, _ = make_memory(
        [response("end_turn", text_block("工具交互摘要"))],
        max_messages=3,
        keep_recent=2,
    )
    memory.add("user", "读取文件")
    memory.add("assistant", [tool_use_block("call-1", "read_file", {"path": "a.txt"})])
    memory.add(
        "user",
        [{"type": "tool_result", "tool_use_id": "call-1", "content": "内容"}],
    )
    memory.add("assistant", [text_block("读取完成")])

    assert memory.maybe_compact() is True
    recent = memory.messages[1:]
    assert recent[0]["role"] == "assistant"
    assert recent[0]["content"][0].id == "call-1"
    assert recent[1] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "call-1", "content": "内容"}
        ],
    }


def test_keeps_all_tools_and_results_from_one_batch() -> None:
    memory, _ = make_memory(
        [response("end_turn", text_block("批量工具摘要"))],
        max_messages=3,
        keep_recent=2,
    )
    memory.add("user", "执行两个工具")
    memory.add(
        "assistant",
        [
            tool_use_block("call-1", "first", {"text": "a"}),
            tool_use_block("call-2", "second", {"text": "b"}),
        ],
    )
    memory.add(
        "user",
        [
            {"type": "tool_result", "tool_use_id": "call-1", "content": "A"},
            {"type": "tool_result", "tool_use_id": "call-2", "content": "B"},
        ],
    )
    memory.add("assistant", [text_block("完成")])

    assert memory.maybe_compact() is True
    tool_use_message = memory.messages[1]
    tool_result_message = memory.messages[2]
    assert [block.id for block in tool_use_message["content"]] == ["call-1", "call-2"]
    assert [block["tool_use_id"] for block in tool_result_message["content"]] == [
        "call-1",
        "call-2",
    ]


def test_does_not_leave_second_tool_result_without_its_call() -> None:
    memory, _ = make_memory(
        [response("end_turn", text_block("连续工具摘要"))],
        max_messages=6,
        keep_recent=2,
    )
    memory.add("user", "第一次")
    memory.add("assistant", [tool_use_block("call-1", "first", {})])
    memory.add("user", [{"type": "tool_result", "tool_use_id": "call-1", "content": "一"}])
    memory.add("assistant", [text_block("第一次完成")])
    memory.add("user", "第二次")
    memory.add("assistant", [tool_use_block("call-2", "second", {})])
    memory.add("user", [{"type": "tool_result", "tool_use_id": "call-2", "content": "二"}])
    memory.add("assistant", [text_block("第二次完成")])

    assert memory.maybe_compact() is True
    retained = memory.messages[1:]
    assert retained[0]["content"][0].id == "call-2"
    assert retained[1]["content"][0]["tool_use_id"] == "call-2"


def test_summary_failure_keeps_original_messages() -> None:
    memory, _ = make_memory(
        [RuntimeError("summary unavailable")],
        max_messages=3,
        keep_recent=2,
    )
    add_plain_messages(memory, 4)
    before = deepcopy(memory.messages)

    with pytest.raises(RuntimeError, match="summary unavailable"):
        memory.maybe_compact()

    assert memory.messages == before


def test_repeated_compaction_remains_bounded() -> None:
    memory, client = make_memory(
        [
            response("end_turn", text_block("第一次摘要")),
            response("end_turn", text_block("第二次摘要")),
        ],
        max_messages=4,
        keep_recent=2,
    )
    add_plain_messages(memory, 5)
    assert memory.maybe_compact() is True
    add_plain_messages(memory, 3)

    assert memory.maybe_compact() is True
    assert len(memory.messages) == 3
    assert len(client.messages.calls) == 2
    assert memory.messages[0]["content"].endswith("第二次摘要")


def test_keep_recent_must_be_positive() -> None:
    with pytest.raises(ValueError, match="keep_recent"):
        make_memory([], keep_recent=0)
