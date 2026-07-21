"""Agent 核心：手写的 Agent Loop（不依赖任何 Agent 框架）。

循环持续把对话发送给模型，执行模型请求的工具，再把 tool_result 回填，
直到模型给出最终文本。工程边界包括步数上限、工具权限、上下文压缩和结构化事件。
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable, Iterable
from uuid import uuid4

import anthropic

from .events import (
    AgentEvent,
    EventListener,
    EventType,
    console_event_listener,
    summarize_for_event,
)
from .memory import ConversationMemory
from .policy import PolicyAction, ToolPolicy
from .tools import ToolPermission, ToolRegistry, build_default_registry

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "")

SYSTEM_PROMPT = (
    "你是一个会使用工具的助理。遵循 ReAct 思路：先想清楚要不要用工具、用哪个，"
    "需要外部能力（精确计算、当前时间、联网搜索、读写本地文件）时主动调用工具，"
    "拿到结果后再继续推理，最终用简洁中文回答。不要编造工具能查到的事实。"
)

RunIdFactory = Callable[[], str]
Clock = Callable[[], datetime]
Timer = Callable[[], float]


def _default_run_id() -> str:
    return uuid4().hex


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Agent:
    """封装 Anthropic 客户端、工具、权限策略、记忆与 Agent Loop。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_steps: int = 8,
        verbose: bool = True,
        registry: ToolRegistry | None = None,
        client: anthropic.Anthropic | None = None,
        policy: ToolPolicy | None = None,
        listeners: Iterable[EventListener] | None = None,
        run_id_factory: RunIdFactory | None = None,
        clock: Clock | None = None,
        timer: Timer | None = None,
    ) -> None:
        self.client = (
            client
            if client is not None
            else anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                base_url=os.environ.get("ANTHROPIC_BASE_URL") or None,
            )
        )
        self.model = model
        self.max_steps = max_steps
        self.verbose = verbose
        self.registry = registry if registry is not None else build_default_registry()
        self.policy = policy if policy is not None else ToolPolicy()
        self.memory = ConversationMemory(self.client, model=model)
        configured_listeners = list(listeners or ())
        if verbose:
            configured_listeners.append(console_event_listener)
        self.listeners = tuple(configured_listeners)
        self.run_id_factory = run_id_factory or _default_run_id
        self.clock = clock or _utc_now
        self.timer = timer or perf_counter

    def _emit(
        self,
        event_type: EventType,
        run_id: str,
        *,
        step: int | None = None,
        tool_name: str | None = None,
        tool_use_id: str | None = None,
        input_summary: str | None = None,
        output_summary: str | None = None,
        error: str | None = None,
        duration_ms: float | None = None,
    ) -> AgentEvent:
        event = AgentEvent(
            event_type=event_type,
            run_id=run_id,
            timestamp=self.clock(),
            step=step,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            input_summary=input_summary,
            output_summary=output_summary,
            error=error,
            duration_ms=duration_ms,
        )
        for listener in self.listeners:
            try:
                listener(event)
            except Exception:
                # 可观测组件不能改变 Agent 的业务控制流。
                continue
        return event

    def _duration_ms(self, started_at: float) -> float:
        return max(0.0, (self.timer() - started_at) * 1000)

    def _finish(self, run_id: str, started_at: float, answer: str) -> str:
        self._emit(
            EventType.RUN_FINISHED,
            run_id,
            output_summary=summarize_for_event(answer),
            duration_ms=self._duration_ms(started_at),
        )
        return answer

    def run(self, user_input: str) -> str:
        """处理一轮用户输入，内部可能多次往返工具，返回最终文本答案。"""
        run_id = self.run_id_factory()
        run_started_at = self.timer()
        self._emit(
            EventType.RUN_STARTED,
            run_id,
            input_summary=summarize_for_event(user_input),
        )
        try:
            return self._run_loop(user_input, run_id, run_started_at)
        except Exception as exc:
            self._emit(
                EventType.RUN_FAILED,
                run_id,
                error=summarize_for_event(f"{type(exc).__name__}: {exc}"),
                duration_ms=self._duration_ms(run_started_at),
            )
            raise

    def _run_loop(self, user_input: str, run_id: str, run_started_at: float) -> str:
        self.memory.add("user", user_input)

        for step_index in range(self.max_steps):
            step = step_index + 1
            self._emit(EventType.MODEL_REQUESTED, run_id, step=step)
            model_started_at = self.timer()
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=self.registry.anthropic_schemas(),
                messages=self.memory.messages,
            )
            self._emit(
                EventType.MODEL_RESPONSE_RECEIVED,
                run_id,
                step=step,
                output_summary=f"stop_reason={resp.stop_reason}",
                duration_ms=self._duration_ms(model_started_at),
            )

            # 模型输出可能同时包含文字和多个 tool_use，必须原样保存在历史中。
            self.memory.add("assistant", resp.content)
            for block in resp.content:
                if block.type == "text" and block.text.strip():
                    self._emit(
                        EventType.MODEL_TEXT_RECEIVED,
                        run_id,
                        step=step,
                        output_summary=summarize_for_event(block.text.strip()),
                    )

            if resp.stop_reason != "tool_use":
                final = "".join(
                    b.text for b in resp.content if b.type == "text"
                ).strip()
                compact_started_at = self.timer()
                if self.memory.maybe_compact():
                    self._emit(
                        EventType.CONTEXT_COMPACTED,
                        run_id,
                        step=step,
                        duration_ms=self._duration_ms(compact_started_at),
                    )
                return self._finish(run_id, run_started_at, final)

            tool_results: list[dict[str, Any]] = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                tool_started_at = self.timer()
                tool = self.registry.get(block.name)
                input_summary = summarize_for_event(block.input)
                is_error = False

                if tool is None:
                    result = f"未知工具：{block.name}"
                    self._emit(
                        EventType.TOOL_CALL_FAILED,
                        run_id,
                        step=step,
                        tool_name=block.name,
                        tool_use_id=block.id,
                        input_summary=input_summary,
                        error=result,
                        duration_ms=self._duration_ms(tool_started_at),
                    )
                else:
                    sensitive = tool.permission == ToolPermission.SENSITIVE
                    input_summary = summarize_for_event(
                        block.input, sensitive=sensitive
                    )
                    action = self.policy.action_for(tool.permission)
                    approval_started_at = self.timer()
                    if action == PolicyAction.ASK:
                        self._emit(
                            EventType.APPROVAL_REQUESTED,
                            run_id,
                            step=step,
                            tool_name=block.name,
                            tool_use_id=block.id,
                            input_summary=input_summary,
                        )
                    decision = self.policy.authorize(
                        tool_use_id=block.id,
                        tool_name=block.name,
                        permission=tool.permission,
                        arguments=block.input,
                    )
                    if action == PolicyAction.ASK:
                        self._emit(
                            EventType.APPROVAL_RESOLVED,
                            run_id,
                            step=step,
                            tool_name=block.name,
                            tool_use_id=block.id,
                            output_summary=summarize_for_event(decision.reason),
                            duration_ms=self._duration_ms(approval_started_at),
                        )

                    if not decision.allowed:
                        result = f"工具执行被拒绝：{decision.reason}"
                        is_error = True
                        self._emit(
                            EventType.TOOL_CALL_FAILED,
                            run_id,
                            step=step,
                            tool_name=block.name,
                            tool_use_id=block.id,
                            input_summary=input_summary,
                            error=summarize_for_event(result),
                            duration_ms=self._duration_ms(tool_started_at),
                        )
                    else:
                        self._emit(
                            EventType.TOOL_CALL_STARTED,
                            run_id,
                            step=step,
                            tool_name=block.name,
                            tool_use_id=block.id,
                            input_summary=input_summary,
                        )
                        execution_started_at = self.timer()
                        outcome = tool.execute(block.input)
                        result = outcome.content
                        event_type = (
                            EventType.TOOL_CALL_FINISHED
                            if outcome.ok
                            else EventType.TOOL_CALL_FAILED
                        )
                        summary = summarize_for_event(result, sensitive=sensitive)
                        self._emit(
                            event_type,
                            run_id,
                            step=step,
                            tool_name=block.name,
                            tool_use_id=block.id,
                            output_summary=summary if outcome.ok else None,
                            error=None if outcome.ok else summary,
                            duration_ms=self._duration_ms(execution_started_at),
                        )

                tool_result: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
                if is_error:
                    tool_result["is_error"] = True
                tool_results.append(tool_result)

            if not tool_results:
                error = "模型声称调用工具但未实际请求，提示其重新决定"
                self._emit(
                    EventType.MODEL_RESPONSE_INVALID,
                    run_id,
                    step=step,
                    error=error,
                )
                self.memory.add(
                    "user",
                    "系统提示：你上一步返回了 stop_reason=tool_use，但没有实际发起任何工具调用。"
                    "请重新决定：如果需要用工具就正确发起调用，如果不需要就直接给出文字回答。",
                )
                continue

            self.memory.add("user", tool_results)

        answer = "（已达到最大步数上限，未能得出最终答案——可能陷入工具循环，建议检查工具描述或提高 max_steps）"
        self._emit(
            EventType.RUN_FAILED,
            run_id,
            error="达到最大步数上限",
            output_summary=summarize_for_event(answer),
            duration_ms=self._duration_ms(run_started_at),
        )
        return answer
