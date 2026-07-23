from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from agent_reflex.instrumentation.otel_setup import setup_otel

_tracer: trace.Tracer | None = None


def _get_tracer() -> trace.Tracer:
    global _tracer
    if _tracer is None:
        _tracer = setup_otel()
    return _tracer


def instrument_agent(name: str | None = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            agent_name = name or func.__name__
            tracer = _get_tracer()
            with tracer.start_as_current_span(
                f"agent.{agent_name}",
                kind=SpanKind.INTERNAL,
                attributes={
                    "gen_ai.request.model": agent_name,
                    "agent.task": agent_name,
                    "agent.type": "agent",
                },
            ) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


def instrument_tool(tool_name: str | None = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tname = tool_name or func.__name__
            tracer = _get_tracer()
            with tracer.start_as_current_span(
                f"tool.{tname}",
                kind=SpanKind.CLIENT,
                attributes={
                    "agent.action": tname,
                    "agent.action.type": "tool",
                },
            ) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("agent.artifact", str(result)[:2000])
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


def instrument_chat(
    model: str = "gpt-4o",
    stream: bool | None = None,
    cache_read_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
    time_to_first_chunk_ms: float | None = None,
) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = _get_tracer()
            attrs: dict[str, Any] = {
                "gen_ai.request.model": model,
                "gen_ai.system": "openai",
            }
            if stream is not None:
                attrs["gen_ai.request.stream"] = stream
            if cache_read_tokens is not None:
                attrs["gen_ai.usage.cache_read.input_tokens"] = cache_read_tokens
            if cache_creation_tokens is not None:
                attrs["gen_ai.usage.cache_creation.input_tokens"] = cache_creation_tokens
            if time_to_first_chunk_ms is not None:
                attrs["gen_ai.response.time_to_first_chunk"] = time_to_first_chunk_ms
            with tracer.start_as_current_span(
                f"chat.{model}",
                kind=SpanKind.CLIENT,
                attributes=attrs,
            ) as span:
                try:
                    start = time.monotonic()
                    result = func(*args, **kwargs)
                    duration = time.monotonic() - start
                    span.set_attribute("gen_ai.usage.completion_tokens", _count_tokens(result))
                    span.set_attribute("gen_ai.usage.prompt_tokens", _count_tokens(args))
                    span.set_attribute("execution_time_ms", duration * 1000)
                    span.add_event("gen_ai.completion", {"content": str(result)[:10000]})
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


def _count_tokens(content: Any) -> int:
    text = str(content)
    return len(text) // 4
