"""对话记忆与上下文压缩。

阶段二要点：上下文窗口有限，长对话必须管理。这里实现一个简单但真实的策略：
  - 永远保留最近 N 轮原始消息（保证近期细节不丢）。
  - 当消息条数超过阈值时，把更早的消息交给模型「摘要」成一段话，
    用一条系统性的 user 消息替代，从而压缩 token 占用。

这不是最优算法，但足以让你理解「记忆 = 可被压缩/检索的上下文」这一核心观念。
真实系统里会用更复杂的分层记忆 + 向量检索（阶段三 RAG 会接上）。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import anthropic
from anthropic.types import MessageParam


class ConversationMemory:
    """维护 messages 列表，并在过长时自动摘要压缩。

    Args:
        client: Anthropic 客户端，用于调用模型做摘要。
        model: 摘要使用的模型。
        max_messages: 触发压缩的消息条数阈值。
        keep_recent: 压缩时保留的最近消息条数。
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        max_messages: int = 20,
        keep_recent: int = 8,
    ) -> None:
        self.client = client
        self.model = model
        if keep_recent < 1:
            raise ValueError("keep_recent 必须大于等于 1")
        self.max_messages = max_messages
        self.keep_recent = keep_recent
        self.messages: list[MessageParam] = []

    def add(self, role: Literal["user", "assistant"], content: Any) -> None:
        """追加一条消息（content 可以是字符串或 content block 列表）。"""
        self.messages.append({"role": role, "content": content})

    def maybe_compact(
        self,
        summarizer: Callable[[str], str] | None = None,
    ) -> bool:
        """超过阈值时压缩较早的消息。

        Returns:
            是否真的执行了压缩（便于在 UI 提示用户）。
        """
        if len(self.messages) <= self.max_messages:
            return False

        # 先按消息数计算候选边界，再向前调整到完整工具交互的边界。
        boundary = _safe_compaction_boundary(
            self.messages, len(self.messages) - self.keep_recent
        )
        if boundary <= 0:
            return False
        to_summarize = self.messages[:boundary]
        recent = self.messages[boundary:]

        # 把旧消息拍平成纯文本喂给模型做摘要
        transcript = _flatten_messages(to_summarize)
        summarize = summarizer or self._summarize
        summary = summarize(transcript)

        # 用一条 user 消息承载摘要，替换掉一大段历史
        compacted: list[MessageParam] = [
            {
                "role": "user",
                "content": f"【以下是早前对话的摘要，供你保持上下文】\n{summary}",
            }
        ]
        # 摘要后第一条若仍是 user，会和上面那条相邻；Anthropic 允许连续同 role，无需特殊处理
        self.messages = compacted + recent
        return True

    def _summarize(self, transcript: str) -> str:
        """调用模型把一段对话转录压缩成要点摘要。"""
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system="你是对话摘要器。把给定对话浓缩成要点，保留关键事实、结论、用户偏好与未完成事项，去掉寒暄。",
            messages=[{"role": "user", "content": transcript}],
        )
        parts = [b.text for b in resp.content if b.type == "text"]
        return "\n".join(parts).strip() or "(无可摘要内容)"


def _block_value(block: Any, key: str) -> Any:
    """读取 SDK block 或 dict block 的统一字段。"""
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


def _blocks(content: Any) -> list[Any]:
    if isinstance(content, (list, tuple)):
        return list(content)
    return []


def _safe_compaction_boundary(
    messages: list[MessageParam],
    candidate: int,
) -> int:
    """将候选切点向前移动，避免拆开 tool_use/tool_result 配对。"""
    tool_use_positions: dict[str, int] = {}
    for index, message in enumerate(messages):
        for block in _blocks(message["content"]):
            if _block_value(block, "type") != "tool_use":
                continue
            tool_use_id = _block_value(block, "id")
            if tool_use_id:
                tool_use_positions[tool_use_id] = index

    boundary = candidate
    while boundary > 0:
        adjusted = False
        for result_index in range(boundary, len(messages)):
            for block in _blocks(messages[result_index]["content"]):
                if _block_value(block, "type") != "tool_result":
                    continue
                tool_use_id = _block_value(block, "tool_use_id")
                use_index = tool_use_positions.get(tool_use_id)
                if use_index is not None and use_index < boundary:
                    boundary = use_index
                    adjusted = True
        if not adjusted:
            break
    return boundary


def _flatten_messages(messages: list[MessageParam]) -> str:
    """把 messages（含 tool_use / tool_result block）拍平成可读文本，供摘要使用。"""
    lines: list[str] = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if isinstance(content, str):
            lines.append(f"{role}: {content}")
            continue
        # content 是 block 列表：分别处理文本 / 工具调用 / 工具结果
        for block in content:
            if isinstance(block, dict):
                if block["type"] == "text":
                    lines.append(f"{role}: {block['text']}")
                elif block["type"] == "tool_use":
                    lines.append(f"{role}: [调用工具 {block['name']}]")
                elif block["type"] == "tool_result":
                    lines.append(f"{role}: [工具返回结果]")
                continue

            if block.type == "text":
                lines.append(f"{role}: {block.text}")
            elif block.type == "tool_use":
                lines.append(f"{role}: [调用工具 {block.name}]")
            elif block.type == "tool_result":
                lines.append(f"{role}: [工具返回结果]")
    return "\n".join(lines)
