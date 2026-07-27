"""Agent 运行可靠性配置、取消和结构化终止结果。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import Event
from typing import Any

import anthropic


class TerminationReason(str, Enum):
    """一次 Agent 运行结束的机器可读原因。"""

    SUCCESS = "success"
    CANCELLED = "cancelled"
    MAX_STEPS = "max_steps"
    TIMEOUT = "timeout"
    TOOL_CALL_LIMIT = "tool_call_limit"
    OUTPUT_LIMIT = "output_limit"
    BUDGET_EXHAUSTED = "budget_exhausted"
    SYSTEM_ERROR = "system_error"


@dataclass(frozen=True, slots=True)
class ReliabilityConfig:
    """模型、工具和整次运行的可靠性边界。"""

    model_timeout_s: float = 30.0
    model_max_retries: int = 2
    tool_timeout_s: float = 20.0
    tool_max_retries: int = 1
    retry_base_delay_s: float = 0.25
    retry_multiplier: float = 2.0
    retry_max_delay_s: float = 4.0
    run_timeout_s: float = 120.0
    max_tool_calls: int = 24
    max_model_output_chars: int = 16_000
    max_tool_result_chars: int = 8_000
    max_total_tokens: int = 100_000

    def __post_init__(self) -> None:
        positive = {
            "model_timeout_s": self.model_timeout_s,
            "tool_timeout_s": self.tool_timeout_s,
            "retry_max_delay_s": self.retry_max_delay_s,
            "run_timeout_s": self.run_timeout_s,
            "max_tool_calls": self.max_tool_calls,
            "max_model_output_chars": self.max_model_output_chars,
            "max_tool_result_chars": self.max_tool_result_chars,
            "max_total_tokens": self.max_total_tokens,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if self.model_max_retries < 0:
            raise ValueError("model_max_retries 不能小于 0")
        if self.tool_max_retries < 0:
            raise ValueError("tool_max_retries 不能小于 0")
        if self.retry_base_delay_s < 0:
            raise ValueError("retry_base_delay_s 不能小于 0")
        if self.retry_multiplier < 1:
            raise ValueError("retry_multiplier 必须大于等于 1")

    def retry_delay(self, retry_index: int) -> float:
        """返回第 retry_index 次重试前的指数退避时间。"""
        if retry_index < 1:
            raise ValueError("retry_index 必须大于等于 1")
        delay = self.retry_base_delay_s * self.retry_multiplier ** (retry_index - 1)
        return min(delay, self.retry_max_delay_s)


class CancellationToken:
    """允许其它线程安全地取消一次 Agent 运行。"""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float) -> bool:
        """等待 timeout 秒；若等待期间被取消则返回 True。"""
        return self._event.wait(timeout)


@dataclass(frozen=True, slots=True)
class RunResult:
    """一次 Agent 运行的结构化结果。"""

    answer: str
    termination_reason: TerminationReason
    run_id: str
    steps: int
    model_attempts: int
    tool_calls: int
    total_tokens: int
    duration_ms: float
    error: str | None = None
    cause: BaseException | None = field(default=None, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return self.termination_reason == TerminationReason.SUCCESS


def is_retryable_model_error(error: BaseException) -> bool:
    """只把明确的临时模型请求错误视为可重试。"""
    if isinstance(
        error,
        (
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
            TimeoutError,
            ConnectionError,
        ),
    ):
        return True
    if isinstance(error, anthropic.APIStatusError):
        return error.status_code >= 500
    return False


def response_token_usage(response: Any) -> int:
    """兼容真实 SDK 响应和测试桩，提取本轮输入输出 token。"""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    fields = (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    )
    return sum(int(getattr(usage, field, 0) or 0) for field in fields)
