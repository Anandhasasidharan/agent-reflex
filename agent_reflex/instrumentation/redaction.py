from __future__ import annotations

import re
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.sdk.trace.export import SpanProcessor

SENSITIVE_PATTERNS: list[re.Pattern] = [
    re.compile(r"(api[_-]?key|apikey|secret|password|auth)['\"]?\s*[:=]\s*['\"][^'\"]+['\"]", re.IGNORECASE),
    re.compile(r"sk-[a-zA-Z0-9]{10,}", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"),
    re.compile(r"\bghp_[a-zA-Z0-9]{10,}\b"),
    re.compile(r"token['\"]?\s*[:=]\s*['\"][a-zA-Z0-9_]{10,}['\"]", re.IGNORECASE),
]


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        for pattern in SENSITIVE_PATTERNS:
            value = pattern.sub("[REDACTED]", value)
        return value
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    return value


class RedactingSpanProcessor(SpanProcessor):
    def on_start(self, span: Span, parent_context: Any | None = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        ctx = span._span if hasattr(span, "_span") else None
        if ctx is None:
            return
        if hasattr(ctx, "attributes") and ctx.attributes:
            for key, value in list(ctx.attributes.items()):
                ctx.set_attribute(key, _redact_value(value))
        if hasattr(ctx, "events") and ctx.events:
            for event in ctx.events:
                if hasattr(event, "attributes") and event.attributes:
                    for key, value in list(event.attributes.items()):
                        event.attributes[key] = _redact_value(value)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
