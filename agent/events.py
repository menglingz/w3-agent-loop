"""Agent 运行事件：为终端、日志、SSE 等观察者提供统一数据。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Mapping


class EventType(str, Enum):
    RUN_STARTED = "run_started"
    MODEL_REQUESTED = "model_requested"
    MODEL_RESPONSE_RECEIVED = "model_response_received"
    MODEL_TEXT_RECEIVED = "model_text_received"
    MODEL_RESPONSE_INVALID = "model_response_invalid"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RESOLVED = "approval_resolved"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_FINISHED = "tool_call_finished"
    TOOL_CALL_FAILED = "tool_call_failed"
    CONTEXT_COMPACTED = "context_compacted"
    RUN_FINISHED = "run_finished"
    RUN_FAILED = "run_failed"


@dataclass(frozen=True, slots=True)
class AgentEvent:
    event_type: EventType
    run_id: str
    timestamp: datetime
    step: int | None = None
    tool_name: str | None = None
    tool_use_id: str | None = None
    input_summary: str | None = None
    output_summary: str | None = None
    error: str | None = None
    duration_ms: float | None = None


EventListener = Callable[[AgentEvent], None]

_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
)
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[a-z0-9._~+/=-]+")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|password|secret|token)\s*[:=]\s*([^\s,;]+)"
)
_SK_KEY_RE = re.compile(r"\bsk-[a-zA-Z0-9_-]{6,}\b")


def _redact_text(text: str) -> str:
    text = _BEARER_RE.sub(r"\1[REDACTED]", text)
    text = _SECRET_ASSIGNMENT_RE.sub(r"\1=[REDACTED]", text)
    return _SK_KEY_RE.sub("[REDACTED]", text)


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _is_sensitive_key(key) else _sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def summarize_for_event(
    value: Any,
    *,
    sensitive: bool = False,
    limit: int = 200,
) -> str:
    """先脱敏再截断，且不修改原始对象。"""
    if sensitive:
        return "[REDACTED]"
    sanitized = _sanitize(value)
    text = sanitized if isinstance(sanitized, str) else repr(sanitized)
    if len(text) <= limit:
        return text
    return text[:limit] + "… [truncated]"


def console_event_listener(event: AgentEvent) -> None:
    """把结构化事件渲染成原有教学风格的终端日志。"""
    if event.event_type == EventType.MODEL_TEXT_RECEIVED and event.output_summary:
        print(f"💭 {event.output_summary}")
    elif event.event_type == EventType.TOOL_CALL_STARTED:
        print(f"🔧 工具调用开始 {event.tool_name}({event.input_summary})")
    elif event.event_type == EventType.TOOL_CALL_FINISHED:
        print(f"  工具调用完成 ↳ {event.output_summary}")
    elif event.event_type == EventType.TOOL_CALL_FAILED:
        icon = "🔒" if event.error and "拒绝" in event.error else "⚠️ "
        print(f"{icon}工具调用失败： {event.error or event.output_summary}")
    elif event.event_type == EventType.MODEL_RESPONSE_INVALID:
        print(f"⚠️  {event.error}")
    elif event.event_type == EventType.CONTEXT_COMPACTED:
        print("🗜️  上下文较长，已自动摘要压缩早期对话")
    elif event.event_type == EventType.MODEL_REQUESTED:
        print(f"当前执行轮数: {event.step}")
