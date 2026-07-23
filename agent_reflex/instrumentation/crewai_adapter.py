from __future__ import annotations

from typing import Any

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from .base import BaseAgentAdapter


class CrewAIAdapter(BaseAgentAdapter):
    def __init__(self) -> None:
        self._tracer = trace.get_tracer("agent_reflex.crewai")

    def get_framework_name(self) -> str:
        return "crewai"

    def instrument_agent_executor(self, executor: Any) -> Any:
        original_kickoff = executor.kickoff

        def traced_kickoff(*args: Any, **kwargs: Any) -> Any:
            with self._tracer.start_as_current_span(
                "agent.crewai.kickoff",
                kind=SpanKind.CLIENT,
                attributes={
                    "gen_ai.request.model": "crewai",
                    "agent.framework": "crewai",
                    "agent.type": "team",
                },
            ) as span:
                try:
                    result = original_kickoff(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        executor.kickoff = traced_kickoff
        return executor

    def extract_agent_id(self, executor: Any) -> str:
        return getattr(executor, "name", "crewai_crew")

    def extract_tools(self, executor: Any) -> list[dict[str, Any]]:
        tools = []
        for agent in getattr(executor, "agents", []) or []:
            for t in getattr(agent, "tools", []) or []:
                tools.append({"name": getattr(t, "name", str(t)), "type": "tool"})
        return tools

    def get_task_graph(self, executor: Any) -> list[dict[str, Any]]:
        tasks = getattr(executor, "tasks", []) or []
        return [
            {"node": getattr(t, "description", str(t))[:80], "edges": []}
            for t in tasks
        ]
