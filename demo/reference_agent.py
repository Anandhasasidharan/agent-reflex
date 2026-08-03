"""Reference OTel producer for AgentReflex ingestion.

Framework-agnostic: uses only the raw OpenTelemetry Python SDK (no
LangGraph/CrewAI), emitting spans that follow the canonical schema in
docs/otel_ingestion_schema.md. Runs a 3-step multi-agent flow with a
deliberately injected failure in step 2, then a verification step that
rejects the failed output.

Run:
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
        python demo/reference_agent.py

Then confirm the trace landed in AgentReflex/Postgres:
    SELECT session_id, failure_type, cause_node_id
    FROM sessions ORDER BY id DESC LIMIT 1;
"""

from __future__ import annotations

import os
import sys
import time

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind, Status, StatusCode

OTEL_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
LLM_MODEL = os.environ.get("REFERENCE_LLM_MODEL", "deepseek-v4-flash")


def setup_tracer() -> trace.Tracer:
    provider = TracerProvider(resource=Resource.create({
        "service.name": "reference-agent",
        "service.version": "0.1.0",
    }))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{OTEL_ENDPOINT}/v1/traces"))
    )
    trace.set_tracer_provider(provider)
    return provider.get_tracer("reference-agent", "0.1.0")


def emit_step(
    tracer: trace.Tracer,
    *,
    name: str,
    agent_id: str,
    subtask_id: str,
    operation: str,
    observation: str,
    result: str,
    thought: str = "",
    error: str | None = None,
    parent_ctx: trace.Context | None = None,
) -> trace.Span:
    """Emit one step span following the canonical schema."""
    attrs = {
        "gen_ai.system": "reference-agent",
        "gen_ai.operation.name": operation,
        "gen_ai.request.model": LLM_MODEL,
        "agent_reflex.agent.id": agent_id,
        "agent_reflex.subtask.id": subtask_id,
    }
    if thought:
        attrs["agent_reflex.agent.thought"] = thought

    span = tracer.start_span(
        f"agent.{name}",
        kind=SpanKind.CLIENT,
        attributes=attrs,
        context=parent_ctx,
    )
    with trace.use_span(span, end_on_exit=True):
        span.add_event("gen_ai.prompt", {"content": observation})
        span.add_event("gen_ai.completion", {"content": result})
        if error:
            span.set_status(Status(StatusCode.ERROR, error))
        else:
            span.set_status(Status(StatusCode.OK))
    return span


def main() -> int:
    scenario = os.environ.get("REFERENCE_SCENARIO", "timeout")
    if scenario not in ("timeout", "success"):
        print(f"unknown REFERENCE_SCENARIO {scenario!r} (use 'timeout' or 'success')", file=sys.stderr)
        return 2

    tracer = setup_tracer()
    session_ctx = trace.set_span_in_context(tracer.start_span("agent.session"))

    print(f"reference-agent: emitting {scenario} trace to {OTEL_ENDPOINT} (model={LLM_MODEL})", flush=True)

    # Step 1: planner drafts the plan (succeeds in both scenarios).
    emit_step(
        tracer,
        name="planner_draft",
        agent_id="planner",
        subtask_id="task_1",
        operation="chat",
        observation="Draft a 2-step plan to ship the release: freeze scope, then deploy.",
        result="Plan: 1) freeze scope 2) deploy to production",
        thought="The user wants a concrete shipping plan with two clear phases.",
        parent_ctx=session_ctx,
    )

    if scenario == "timeout":
        # Step 2: researcher fetches a required input — deliberately fails
        # (simulates an infra timeout). The trace must mark this step ERROR.
        emit_step(
            tracer,
            name="researcher_fetch",
            agent_id="researcher",
            subtask_id="task_1",
            operation="call_tool",
            observation="Fetch the release checklist from the ops store.",
            result="",
            thought="Querying the ops store for the release checklist.",
            error="tool timeout after 30s: ops store unreachable",
            parent_ctx=session_ctx,
        )
    else:
        # Step 2 (success scenario): the checklist fetch succeeds.
        emit_step(
            tracer,
            name="researcher_fetch",
            agent_id="researcher",
            subtask_id="task_1",
            operation="call_tool",
            observation="Fetch the release checklist from the ops store.",
            result="checklist retrieved: scope frozen, deploys gated on sign-off",
            thought="Querying the ops store for the release checklist.",
            parent_ctx=session_ctx,
        )

    # Step 3: verifier rejects the plan when the checklist fetch failed —
    # the final error that surfaces to the user. In the success scenario the
    # verification passes.
    verifier_error = None if scenario == "success" else "verification failed: missing release checklist"
    verifier_result = (
        "Plan APPROVED: scope frozen and checklist complete."
        if scenario == "success"
        else "Plan REJECTED: cannot proceed without the release checklist."
    )
    emit_step(
        tracer,
        name="verifier_check",
        agent_id="verifier",
        subtask_id="task_2",
        operation="chat",
        observation="Verify the plan against the release checklist.",
        result=verifier_result,
        thought="The checklist fetch failed, so the plan cannot be verified."
        if scenario == "timeout" else "The plan is complete and can be verified.",
        error=verifier_error,
        parent_ctx=session_ctx,
    )

    # Allow the batch processor to flush before exit.
    trace.get_tracer_provider().force_flush()  # type: ignore[attr-defined]
    time.sleep(1.0)

    print("reference-agent: emitted 3 spans.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
