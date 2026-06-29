"""Agent 核心：手写的 Agent Loop（不依赖任何 Agent 框架）。

★ 这是阶段二最该吃透的文件。一句话概括 Agent Loop：

    while 模型还在请求工具:
        把对话发给模型
        模型回复里若有 tool_use → 本地执行对应工具 → 把结果作为 tool_result 回填
    模型不再请求工具 → 输出最终文本答案

围绕这个循环，工程上还要处理：
  - 多工具：一轮可能请求多个工具，要逐个执行后一起回填。
  - 健壮性：未知工具、参数非法、工具异常都要回传给模型而非崩溃；加步数上限防死循环。
  - 记忆：每轮结束后检查是否需要压缩上下文。
  - 可观测：把「模型在想什么、调了什么工具、结果如何」打印出来（阶段六会升级为结构化 trace）。
"""
from __future__ import annotations

import os
from typing import Any

import anthropic

from .memory import ConversationMemory
from .tools import ToolRegistry, build_default_registry

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", '')

SYSTEM_PROMPT = (
    "你是一个会使用工具的助理。遵循 ReAct 思路：先想清楚要不要用工具、用哪个，"
    "需要外部能力（精确计算、当前时间、联网搜索、读写本地文件）时主动调用工具，"
    "拿到结果后再继续推理，最终用简洁中文回答。不要编造工具能查到的事实。"
)


class Agent:
    """封装 Anthropic 客户端、工具注册表、记忆与 Agent Loop。

    Args:
        model: 使用的模型 id。
        max_steps: 单次 run 内最多的「模型↔工具」往返步数，防死循环。
        verbose: 是否打印每一步的思考/工具调用过程（教学用，建议开）。
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_steps: int = 8,
        verbose: bool = True,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            base_url=os.environ.get("ANTHROPIC_BASE_URL") or None,
        )
        self.model = model
        self.max_steps = max_steps
        self.verbose = verbose
        self.registry = registry or build_default_registry()
        self.memory = ConversationMemory(self.client, model=model)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def run(self, user_input: str) -> str:
        """处理一轮用户输入，内部可能多次往返工具，返回最终文本答案。"""
        self.memory.add("user", user_input)

        for step in range(self.max_steps):
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=self.registry.anthropic_schemas(),
                messages=self.memory.messages,
            )

            # 把模型这一轮输出（可能含 tool_use）原样存入记忆
            self.memory.add("assistant", resp.content)

            # 打印模型这一步说的文字（它的「思考/解释」）
            for block in resp.content:
                if block.type == "text" and block.text.strip():
                    self._log(f"💭 {block.text.strip()}")

            # stop_reason != tool_use → 模型给出了最终答案，结束循环
            if resp.stop_reason != "tool_use":
                final = "".join(b.text for b in resp.content if b.type == "text")
                # 一轮对话结束后，按需压缩历史
                if self.memory.maybe_compact():
                    self._log("🗜️  上下文较长，已自动摘要压缩早期对话")
                return final.strip()

            # 否则：逐个执行被请求的工具，收集 tool_result
            tool_results: list[dict[str, Any]] = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                tool = self.registry.get(block.name)
                if tool is None:
                    # 健壮性：模型请求了不存在的工具，回传错误让它换路
                    result = f"未知工具：{block.name}"
                    self._log(f"⚠️  {result}")
                else:
                    self._log(f"🔧 调用 {block.name}({block.input})")
                    result = tool.run(block.input)  # base.run 已内含参数校验与异常兜底
                    self._log(f"   ↳ {result[:200]}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

            # 工具结果作为一条 user 消息回填，进入下一轮
            self.memory.add("user", tool_results)

        return "（已达到最大步数上限，未能得出最终答案——可能陷入工具循环，建议检查工具描述或提高 max_steps）"
