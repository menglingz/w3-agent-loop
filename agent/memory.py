"""对话记忆与上下文压缩。

阶段二要点：上下文窗口有限，长对话必须管理。这里实现一个简单但真实的策略：
  - 永远保留最近 N 轮原始消息（保证近期细节不丢）。
  - 当消息条数超过阈值时，把更早的消息交给模型「摘要」成一段话，
    用一条系统性的 user 消息替代，从而压缩 token 占用。

这不是最优算法，但足以让你理解「记忆 = 可被压缩/检索的上下文」这一核心观念。
真实系统里会用更复杂的分层记忆 + 向量检索（阶段三 RAG 会接上）。
"""
from __future__ import annotations

from typing import Any

import anthropic


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
        self.max_messages = max_messages
        self.keep_recent = keep_recent
        self.messages: list[dict[str, Any]] = []

    def add(self, role: str, content: Any) -> None:
        """追加一条消息（content 可以是字符串或 content block 列表）。"""
        self.messages.append({"role": role, "content": content})

    def maybe_compact(self) -> bool:
        """超过阈值时压缩较早的消息。

        Returns:
            是否真的执行了压缩（便于在 UI 提示用户）。
        """
        if len(self.messages) <= self.max_messages:
            return False

        # 切分：要被摘要的旧消息 vs 原样保留的近期消息
        to_summarize = self.messages[: -self.keep_recent]
        recent = self.messages[-self.keep_recent :]

        # 把旧消息拍平成纯文本喂给模型做摘要
        transcript = _flatten_messages(to_summarize)
        summary = self._summarize(transcript)

        # 用一条 user 消息承载摘要，替换掉一大段历史
        compacted: list[dict[str, Any]] = [
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


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
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
            btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            if btype == "text":
                text = block["text"] if isinstance(block, dict) else block.text
                lines.append(f"{role}: {text}")
            elif btype == "tool_use":
                name = block["name"] if isinstance(block, dict) else block.name
                lines.append(f"{role}: [调用工具 {name}]")
            elif btype == "tool_result":
                lines.append(f"{role}: [工具返回结果]")
    return "\n".join(lines)
