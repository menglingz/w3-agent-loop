"""Agent 核心：带可靠性控制的手写 Agent Loop。"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
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
from .reliability import (
    CancellationToken,
    ReliabilityConfig,
    RunResult,
    TerminationReason,
    is_retryable_model_error,
    response_token_usage,
)
from .tools import (
    Tool,
    ToolExecutionResult,
    ToolFailureKind,
    ToolPermission,
    ToolRegistry,
    build_default_registry,
)

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "")

SYSTEM_PROMPT = (
    "你是一个会使用工具的助理。遵循 ReAct 思路：先想清楚要不要用工具、用哪个，"
    "需要外部能力（精确计算、当前时间、联网搜索、读写本地文件）时主动调用工具，"
    "拿到结果后再继续推理，最终用简洁中文回答。不要编造工具能查到的事实。"
)

RunIdFactory = Callable[[], str]
Clock = Callable[[], datetime]
Timer = Callable[[], float]
Waiter = Callable[[float, CancellationToken], bool]
ToolExecutor = Callable[
    [Tool[Any], dict[str, Any], float, CancellationToken],
    ToolExecutionResult,
]


@dataclass
class _RunState:
    run_id: str
    started_at: float
    deadline: float
    cancellation: CancellationToken
    steps: int = 0
    model_attempts: int = 0
    tool_calls: int = 0
    total_tokens: int = 0


class _ControlledTermination(Exception):
    def __init__(
        self,
        answer: str,
        reason: TerminationReason,
        error: str,
    ) -> None:
        super().__init__(error)
        self.answer = answer
        self.reason = reason
        self.error = error


def _default_run_id() -> str:
    return uuid4().hex


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _default_waiter(delay_s: float, cancellation: CancellationToken) -> bool:
    return cancellation.wait(delay_s)


def _default_tool_executor(
    tool: Tool[Any],
    raw_input: dict[str, Any],
    timeout_s: float,
    cancellation: CancellationToken,
) -> ToolExecutionResult:
    """在线程中执行工具；超时只停止等待，不能强制杀死 Python 线程。"""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{tool.name}")
    future = executor.submit(tool.execute, raw_input)
    deadline = perf_counter() + timeout_s
    try:
        while True:
            if cancellation.is_cancelled:
                future.cancel()
                return ToolExecutionResult(
                    False,
                    "工具执行已取消；若工具已经开始运行，其最终状态可能未知。",
                    ToolFailureKind.CANCELLED,
                )
            remaining = deadline - perf_counter()
            if remaining <= 0:
                future.cancel()
                return ToolExecutionResult(
                    False,
                    f"工具执行超过 {timeout_s:g} 秒；已停止等待，执行状态未知。",
                    ToolFailureKind.TIMEOUT,
                )
            try:
                return future.result(timeout=min(0.05, remaining))
            except FutureTimeoutError:
                continue
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


class Agent:
    """封装 Anthropic 客户端、工具、权限、可靠性、记忆与 Agent Loop。"""

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
        reliability: ReliabilityConfig | None = None,
        waiter: Waiter | None = None,
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        # 超时、重试、step 上限等可靠性参数
        self.reliability = reliability or ReliabilityConfig()
        # Anthropic SDK 客户端，负责向模型发请求
        self.client = (
            client
            if client is not None
            else anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                base_url=os.environ.get("ANTHROPIC_BASE_URL") or None,
                max_retries=0,
            )
        )
        # 使用的模型 ID
        self.model = model
        # 单次 run 最多执行的工具调用轮数，防止无限循环
        self.max_steps = max_steps
        # 是否向控制台打印事件日志
        self.verbose = verbose
        # 工具注册表，管理所有可用工具
        self.registry = registry if registry is not None else build_default_registry()
        # 工具调用权限策略，决定哪些工具需要用户审批
        self.policy = policy if policy is not None else ToolPolicy()
        # 对话记忆，管理历史消息并在超长时自动压缩
        self.memory = ConversationMemory(self.client, model=model)
        configured_listeners = list(listeners or ())
        if verbose:
            configured_listeners.append(console_event_listener)
        # 事件监听器列表，用于可观测性（日志、监控等）
        self.listeners = tuple(configured_listeners)
        # 生成每次 run 唯一 ID 的工厂函数
        self.run_id_factory = run_id_factory or _default_run_id
        # 获取当前时间的函数，便于测试时注入 mock
        self.clock = clock or _utc_now
        # 计时器函数，用于统计耗时
        self.timer = timer or perf_counter
        # 重试等待函数，便于测试时注入 mock 跳过实际等待
        self.waiter = waiter or _default_waiter
        # 工具执行器，在线程池中运行工具并支持超时控制
        self.tool_executor = tool_executor or _default_tool_executor
        # 保存最近一次 run 的结果，方便调用方读取
        self.last_result: RunResult | None = None

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
        attempt: int | None = None,
        termination_reason: TerminationReason | None = None,
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
            attempt=attempt,
            termination_reason=(
                termination_reason.value if termination_reason is not None else None
            ),
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

    def _remaining_s(self, state: _RunState) -> float:
        return state.deadline - self.timer()

    def _finish_result(
        self,
        state: _RunState,
        answer: str,
        reason: TerminationReason,
        *,
        error: str | None = None,
        cause: BaseException | None = None,
    ) -> RunResult:
        duration_ms = self._duration_ms(state.started_at)
        if reason == TerminationReason.SUCCESS:
            self._emit(
                EventType.RUN_FINISHED,
                state.run_id,
                output_summary=summarize_for_event(answer),
                duration_ms=duration_ms,
                termination_reason=reason,
            )
        elif reason == TerminationReason.CANCELLED:
            self._emit(
                EventType.RUN_CANCELLED,
                state.run_id,
                output_summary=summarize_for_event(answer),
                error=error,
                duration_ms=duration_ms,
                termination_reason=reason,
            )
        else:
            if reason != TerminationReason.SYSTEM_ERROR:
                self._emit(
                    EventType.RUN_LIMIT_REACHED,
                    state.run_id,
                    output_summary=summarize_for_event(answer),
                    error=error,
                    duration_ms=duration_ms,
                    termination_reason=reason,
                )
            self._emit(
                EventType.RUN_FAILED,
                state.run_id,
                output_summary=summarize_for_event(answer),
                error=error,
                duration_ms=duration_ms,
                termination_reason=reason,
            )
        result = RunResult(
            answer=answer,
            termination_reason=reason,
            run_id=state.run_id,
            steps=state.steps,
            model_attempts=state.model_attempts,
            tool_calls=state.tool_calls,
            total_tokens=state.total_tokens,
            duration_ms=duration_ms,
            error=error,
            cause=cause,
        )
        self.last_result = result
        return result

    def run(
        self,
        user_input: str,
        *,
        cancellation: CancellationToken | None = None,
    ) -> str:
        """兼容原接口；系统异常仍重新抛出，其余终止返回中文答案。"""
        result = self.run_with_result(user_input, cancellation=cancellation)
        if result.termination_reason == TerminationReason.SYSTEM_ERROR and result.cause:
            raise result.cause
        return result.answer

    def run_with_result(
        self,
        user_input: str,
        *,
        cancellation: CancellationToken | None = None,
    ) -> RunResult:
        """运行一轮对话并返回机器可读的终止原因和资源统计。"""
        started_at = self.timer()
        state = _RunState(
            run_id=self.run_id_factory(),
            started_at=started_at,
            deadline=started_at + self.reliability.run_timeout_s,
            cancellation=cancellation or CancellationToken(),
        )
        self._emit(
            EventType.RUN_STARTED,
            state.run_id,
            input_summary=summarize_for_event(user_input),
        )
        try:
            answer, reason, error = self._run_loop(user_input, state)
            return self._finish_result(state, answer, reason, error=error)
        except KeyboardInterrupt as exc:
            state.cancellation.cancel()
            return self._finish_result(
                state,
                "（运行已由用户取消）",
                TerminationReason.CANCELLED,
                error="用户取消运行",
                cause=exc,
            )
        except Exception as exc:
            error = summarize_for_event(f"{type(exc).__name__}: {exc}")
            return self._finish_result(
                state,
                "（Agent 运行失败，请检查错误日志）",
                TerminationReason.SYSTEM_ERROR,
                error=error,
                cause=exc,
            )

    def _check_before_model(
        self, state: _RunState
    ) -> tuple[str, TerminationReason, str] | None:
        if state.cancellation.is_cancelled:
            return "（运行已取消）", TerminationReason.CANCELLED, "收到取消信号"
        if self._remaining_s(state) <= 0:
            return (
                "（已达到总运行时间限制）",
                TerminationReason.TIMEOUT,
                "达到总运行时间限制",
            )
        return None

    def _request_with_retries(
        self,
        state: _RunState,
        step: int,
        create_request: Callable[[float], Any],
    ) -> Any:
        for retry_index in range(self.reliability.model_max_retries + 1):
            check = self._check_before_model(state)
            if check is not None:
                answer, reason, error = check
                raise _ControlledTermination(answer, reason, error)

            attempt = retry_index + 1
            state.model_attempts += 1
            self._emit(
                EventType.MODEL_REQUESTED,
                state.run_id,
                step=step,
                attempt=attempt,
            )
            model_started_at = self.timer()
            timeout_s = min(
                self.reliability.model_timeout_s,
                max(0.001, self._remaining_s(state)),
            )
            try:
                response = create_request(timeout_s)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                can_retry = (
                    is_retryable_model_error(exc)
                    and retry_index < self.reliability.model_max_retries
                )
                if not can_retry:
                    raise
                delay_s = self.reliability.retry_delay(retry_index + 1)
                if delay_s >= self._remaining_s(state):
                    raise _ControlledTermination(
                        "（模型请求重试前已达到总运行时间限制）",
                        TerminationReason.TIMEOUT,
                        "模型重试将超过总运行时间限制",
                    ) from exc
                self._emit(
                    EventType.MODEL_RETRY_SCHEDULED,
                    state.run_id,
                    step=step,
                    attempt=attempt + 1,
                    error=summarize_for_event(f"{type(exc).__name__}: {exc}"),
                    output_summary=f"{delay_s:g} 秒后重试",
                    duration_ms=self._duration_ms(model_started_at),
                )
                if self.waiter(delay_s, state.cancellation):
                    raise _ControlledTermination(
                        "（模型重试等待期间运行被取消）",
                        TerminationReason.CANCELLED,
                        "收到取消信号",
                    ) from exc
                continue

            self._emit(
                EventType.MODEL_RESPONSE_RECEIVED,
                state.run_id,
                step=step,
                output_summary=f"stop_reason={response.stop_reason}",
                duration_ms=self._duration_ms(model_started_at),
                attempt=attempt,
            )
            state.total_tokens += response_token_usage(response)
            return response
        raise AssertionError("模型重试循环不应执行到这里")

    def _request_model(self, state: _RunState, step: int) -> Any:
        return self._request_with_retries(
            state,
            step,
            lambda timeout_s: self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=self.registry.anthropic_schemas(),
                messages=self.memory.messages,
                timeout=timeout_s,
            ),
        )

    def _summarize_memory(
        self,
        transcript: str,
        state: _RunState,
        step: int,
    ) -> str:
        response = self._request_with_retries(
            state,
            step,
            lambda timeout_s: self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=(
                    "你是对话摘要器。把给定对话浓缩成要点，保留关键事实、结论、"
                    "用户偏好与未完成事项，去掉寒暄。"
                ),
                messages=[{"role": "user", "content": transcript}],
                timeout=timeout_s,
            ),
        )
        parts = [block.text for block in response.content if block.type == "text"]
        return "\n".join(parts).strip() or "（无可摘要内容）"

    def _execute_tool_once(
        self,
        tool: Tool[Any],
        raw_input: dict[str, Any],
        state: _RunState,
        step: int,
        tool_use_id: str,
        input_summary: str,
        attempt: int,
    ) -> ToolExecutionResult:
        if state.cancellation.is_cancelled:
            return ToolExecutionResult(
                False, "工具执行已取消。", ToolFailureKind.CANCELLED
            )
        if self._remaining_s(state) <= 0:
            return ToolExecutionResult(
                False,
                "已达到总运行时间限制，未执行工具。",
                ToolFailureKind.TIMEOUT,
            )
        if state.tool_calls >= self.reliability.max_tool_calls:
            return ToolExecutionResult(
                False,
                "已达到本次运行的最大工具调用次数。",
                ToolFailureKind.LIMIT,
            )

        state.tool_calls += 1
        self._emit(
            EventType.TOOL_CALL_STARTED,
            state.run_id,
            step=step,
            tool_name=tool.name,
            tool_use_id=tool_use_id,
            input_summary=input_summary,
            attempt=attempt,
        )
        execution_started_at = self.timer()
        timeout_s = min(
            self.reliability.tool_timeout_s,
            max(0.001, self._remaining_s(state)),
        )
        outcome = self.tool_executor(tool, raw_input, timeout_s, state.cancellation)
        sensitive = tool.permission == ToolPermission.SENSITIVE
        summary = summarize_for_event(outcome.content, sensitive=sensitive)
        if outcome.ok:
            event_type = EventType.TOOL_CALL_FINISHED
        elif outcome.failure_kind == ToolFailureKind.TIMEOUT:
            event_type = EventType.TOOL_CALL_TIMED_OUT
        else:
            event_type = EventType.TOOL_CALL_FAILED
        self._emit(
            event_type,
            state.run_id,
            step=step,
            tool_name=tool.name,
            tool_use_id=tool_use_id,
            output_summary=summary if outcome.ok else None,
            error=None if outcome.ok else summary,
            duration_ms=self._duration_ms(execution_started_at),
            attempt=attempt,
        )
        return outcome

    def _execute_tool_with_retries(
        self,
        tool: Tool[Any],
        raw_input: dict[str, Any],
        state: _RunState,
        step: int,
        tool_use_id: str,
        input_summary: str,
    ) -> ToolExecutionResult:
        for retry_index in range(self.reliability.tool_max_retries + 1):
            attempt = retry_index + 1
            outcome = self._execute_tool_once(
                tool,
                raw_input,
                state,
                step,
                tool_use_id,
                input_summary,
                attempt,
            )
            can_retry = (
                not outcome.ok
                and outcome.failure_kind == ToolFailureKind.EXECUTION
                and tool.idempotent
                and tool.retryable
                and retry_index < self.reliability.tool_max_retries
            )
            if not can_retry:
                return outcome
            delay_s = self.reliability.retry_delay(retry_index + 1)
            if delay_s >= self._remaining_s(state):
                return ToolExecutionResult(
                    False,
                    "工具重试将超过总运行时间限制。",
                    ToolFailureKind.TIMEOUT,
                )
            self._emit(
                EventType.TOOL_RETRY_SCHEDULED,
                state.run_id,
                step=step,
                tool_name=tool.name,
                tool_use_id=tool_use_id,
                error=summarize_for_event(outcome.content),
                output_summary=f"{delay_s:g} 秒后重试",
                attempt=attempt + 1,
            )
            if self.waiter(delay_s, state.cancellation):
                return ToolExecutionResult(
                    False,
                    "工具重试等待期间运行被取消。",
                    ToolFailureKind.CANCELLED,
                )
        raise AssertionError("工具重试循环不应执行到这里")

    def _limited_tool_content(self, content: str) -> str:
        limit = self.reliability.max_tool_result_chars
        if len(content) <= limit:
            return content
        return content[:limit] + f"\n…（工具结果已截断，原始长度 {len(content)} 字符）"

    @staticmethod
    def _tool_result(
        tool_use_id: str,
        content: str,
        *,
        is_error: bool,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if is_error:
            result["is_error"] = True
        return result

    def _run_loop(
        self,
        user_input: str,
        state: _RunState,
    ) -> tuple[str, TerminationReason, str | None]:
        self.memory.add("user", user_input)

        for step_index in range(self.max_steps):
            step = step_index + 1
            state.steps = step
            try:
                response = self._request_model(state, step)
            except _ControlledTermination as stop:
                return stop.answer, stop.reason, stop.error

            self.memory.add("assistant", response.content)
            text = "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    self._emit(
                        EventType.MODEL_TEXT_RECEIVED,
                        state.run_id,
                        step=step,
                        output_summary=summarize_for_event(block.text.strip()),
                    )

            tool_blocks = [
                block for block in response.content if block.type == "tool_use"
            ]
            blocked_reason: TerminationReason | None = None
            blocked_error: str | None = None

            if state.total_tokens > self.reliability.max_total_tokens:
                blocked_reason = TerminationReason.BUDGET_EXHAUSTED
                blocked_error = (
                    f"token 用量 {state.total_tokens} 超过预算 "
                    f"{self.reliability.max_total_tokens}"
                )
            elif len(text) > self.reliability.max_model_output_chars:
                blocked_reason = TerminationReason.OUTPUT_LIMIT
                blocked_error = (
                    f"模型输出 {len(text)} 字符，超过限制 "
                    f"{self.reliability.max_model_output_chars}"
                )

            if response.stop_reason != "tool_use":
                if blocked_reason == TerminationReason.BUDGET_EXHAUSTED:
                    return (
                        "（本次运行的 token 预算已耗尽）",
                        blocked_reason,
                        blocked_error,
                    )
                if blocked_reason == TerminationReason.OUTPUT_LIMIT:
                    limit = self.reliability.max_model_output_chars
                    answer = text[:limit] + "\n…（模型输出超过限制，已截断）"
                    return answer, blocked_reason, blocked_error
                compact_started_at = self.timer()
                try:
                    compacted = self.memory.maybe_compact(
                        lambda transcript: self._summarize_memory(
                            transcript,
                            state,
                            step,
                        )
                    )
                except _ControlledTermination as stop:
                    return stop.answer, stop.reason, stop.error
                if compacted:
                    self._emit(
                        EventType.CONTEXT_COMPACTED,
                        state.run_id,
                        step=step,
                        duration_ms=self._duration_ms(compact_started_at),
                    )
                if state.total_tokens > self.reliability.max_total_tokens:
                    error = (
                        f"token 用量 {state.total_tokens} 超过预算 "
                        f"{self.reliability.max_total_tokens}"
                    )
                    return (
                        "（上下文压缩后，本次运行的 token 预算已耗尽）",
                        TerminationReason.BUDGET_EXHAUSTED,
                        error,
                    )
                return text, TerminationReason.SUCCESS, None

            if not tool_blocks:
                error = "模型声称调用工具但未实际请求，提示其重新决定"
                self._emit(
                    EventType.MODEL_RESPONSE_INVALID,
                    state.run_id,
                    step=step,
                    error=error,
                )
                self.memory.add(
                    "user",
                    "系统提示：你上一步返回了 stop_reason=tool_use，但没有实际发起任何工具调用。"
                    "请重新决定：如果需要用工具就正确发起调用，如果不需要就直接给出文字回答。",
                )
                continue

            tool_results: list[dict[str, Any]] = []
            for block in tool_blocks:
                tool_started_at = self.timer()
                tool = self.registry.get(block.name)
                input_summary = summarize_for_event(block.input)

                if blocked_reason is not None:
                    tool_results.append(
                        self._tool_result(
                            block.id,
                            f"工具未执行：{blocked_error}",
                            is_error=True,
                        )
                    )
                    continue
                if state.cancellation.is_cancelled:
                    blocked_reason = TerminationReason.CANCELLED
                    blocked_error = "收到取消信号"
                    tool_results.append(
                        self._tool_result(
                            block.id, "工具未执行：运行已取消。", is_error=True
                        )
                    )
                    continue
                if self._remaining_s(state) <= 0:
                    blocked_reason = TerminationReason.TIMEOUT
                    blocked_error = "达到总运行时间限制"
                    tool_results.append(
                        self._tool_result(
                            block.id,
                            "工具未执行：已达到总运行时间限制。",
                            is_error=True,
                        )
                    )
                    continue
                if tool is None:
                    result = f"未知工具：{block.name}"
                    self._emit(
                        EventType.TOOL_CALL_FAILED,
                        state.run_id,
                        step=step,
                        tool_name=block.name,
                        tool_use_id=block.id,
                        input_summary=input_summary,
                        error=result,
                        duration_ms=self._duration_ms(tool_started_at),
                    )
                    tool_results.append(
                        self._tool_result(block.id, result, is_error=True)
                    )
                    continue

                sensitive = tool.permission == ToolPermission.SENSITIVE
                input_summary = summarize_for_event(block.input, sensitive=sensitive)
                action = self.policy.action_for(tool.permission)
                approval_started_at = self.timer()
                if action == PolicyAction.ASK:
                    self._emit(
                        EventType.APPROVAL_REQUESTED,
                        state.run_id,
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
                        state.run_id,
                        step=step,
                        tool_name=block.name,
                        tool_use_id=block.id,
                        output_summary=summarize_for_event(decision.reason),
                        duration_ms=self._duration_ms(approval_started_at),
                    )
                if not decision.allowed:
                    result = f"工具执行被拒绝：{decision.reason}"
                    self._emit(
                        EventType.TOOL_CALL_FAILED,
                        state.run_id,
                        step=step,
                        tool_name=block.name,
                        tool_use_id=block.id,
                        input_summary=input_summary,
                        error=summarize_for_event(result),
                        duration_ms=self._duration_ms(tool_started_at),
                    )
                    tool_results.append(
                        self._tool_result(block.id, result, is_error=True)
                    )
                    continue

                outcome = self._execute_tool_with_retries(
                    tool,
                    block.input,
                    state,
                    step,
                    block.id,
                    input_summary,
                )
                result = self._limited_tool_content(outcome.content)
                tool_results.append(
                    self._tool_result(block.id, result, is_error=not outcome.ok)
                )
                if outcome.failure_kind == ToolFailureKind.CANCELLED:
                    blocked_reason = TerminationReason.CANCELLED
                    blocked_error = outcome.content
                elif outcome.failure_kind == ToolFailureKind.TIMEOUT:
                    blocked_reason = TerminationReason.TIMEOUT
                    blocked_error = outcome.content
                elif outcome.failure_kind == ToolFailureKind.LIMIT:
                    blocked_reason = TerminationReason.TOOL_CALL_LIMIT
                    blocked_error = outcome.content

            self.memory.add("user", tool_results)
            if blocked_reason is not None:
                answers = {
                    TerminationReason.CANCELLED: "（运行已取消）",
                    TerminationReason.TIMEOUT: "（运行或工具执行已超时）",
                    TerminationReason.TOOL_CALL_LIMIT: "（已达到最大工具调用次数）",
                    TerminationReason.BUDGET_EXHAUSTED: "（本次运行的 token 预算已耗尽）",
                    TerminationReason.OUTPUT_LIMIT: "（模型输出超过大小限制）",
                }
                return answers[blocked_reason], blocked_reason, blocked_error

        answer = "（已达到最大步数上限，未能得出最终答案——可能陷入工具循环，建议检查工具描述或提高 max_steps）"
        return answer, TerminationReason.MAX_STEPS, "达到最大步数上限"
