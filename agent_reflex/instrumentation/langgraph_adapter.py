from __future__ import annotations

from typing import Any

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from .base import BaseAgentAdapter


class LangGraphAdapter(BaseAgentAdapter):
    def __init__(self) -> None:
        self._tracer = trace.get_tracer("agent_reflex.langgraph")

    def get_framework_name(self) -> str:
        return "langgraph"

    def instrument_agent_executor(self, executor: Any) -> Any:
        original_invoke = executor.invoke

        def traced_invoke(inputs: Any, *args: Any, **kwargs: Any) -> Any:
            with self._tracer.start_as_current_span(
                "agent.langgraph.invoke",
                kind=SpanKind.CLIENT,
                attributes={
                    "gen_ai.request.model": "langgraph",
                    "agent.framework": "langgraph",
                    "agent.type": "team",
                },
            ) as span:
                try:
                    result = original_invoke(inputs, *args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        executor.invoke = traced_invoke
        return executor

    def extract_agent_id(self, executor: Any) -> str:
        return getattr(executor, "name", "langgraph_agent")

    def extract_tools(self, executor: Any) -> list[dict[str, Any]]:
        tools = []
        for node in getattr(executor, "nodes", []) or []:
            tool_attr = getattr(node, "tools", []) or getattr(node, "functions", []) or []
            for t in tool_attr:
                tools.append({"name": getattr(t, "name", str(t)), "type": "tool"})
        return tools

    def get_task_graph(self, executor: Any) -> list[dict[str, Any]]:
        nodes = getattr(executor, "nodes", []) or []
        edges = getattr(executor, "edges", []) or getattr(executor, "transitions", []) or []
        return [
            {"node": getattr(n, "name", str(n)), "edges": [str(e) for e in edges]}
            for n in nodes
        ]
