"""Verify OTAR parsing and graph ingestion against the REAL OTel SDK export
format — not a hand-built dict mimicking one.

The OTel SDK builds spans, a real SpanExporter captures them, and the
serialized ReadableSpan.to_json() output (the actual wire representation) is
fed through OTARParser and spans_to_graph.
"""

from __future__ import annotations

import json

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode

from agent_reflex.graph.models import OTARParser
from agent_reflex.graph.span_ingest import parse_otlp_json, spans_to_graph


class CaptureExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list = []

    def export(self, spans: list) -> None:  # type: ignore[override]
        self.spans.extend(spans)

    def shutdown(self) -> None:
        pass


def _make_tracer():
    exporter = CaptureExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "test-producer"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, provider.get_tracer("test", "0.1.0"), exporter


def test_real_otlp_span_wire_format_parses_to_otar():
    provider, tracer, exporter = _make_tracer()

    with tracer.start_as_current_span(
        "agent.plan",
        kind=SpanKind.CLIENT,
        attributes={
            "gen_ai.system": "reference-agent",
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": "deepseek-v4-flash",
            "agent_reflex.agent.id": "planner",
            "agent_reflex.subtask.id": "task_1",
        },
    ) as span:
        span.add_event("gen_ai.prompt", {"content": "Plan the release steps"})
        span.add_event("gen_ai.completion", {"content": "1) freeze scope 2) ship"})
        span.set_status(Status(StatusCode.OK))

    provider.force_flush()
    assert len(exporter.spans) == 1

    wire = json.loads(exporter.spans[0].to_json())
    attrs = wire["attributes"]
    events = [
        {"name": e["name"], "attributes": e.get("attributes", {})}
        for e in wire["events"]
    ]

    otar = OTARParser.parse(attrs, events=events, span_name=wire["name"])
    assert otar.observation == "Plan the release steps"
    assert otar.result == "1) freeze scope 2) ship"
    assert otar.action == "chat"
    assert wire["status"]["status_code"] == "OK"


def test_real_otlp_span_error_status_sets_error_flag():
    provider, tracer, exporter = _make_tracer()

    with tracer.start_as_current_span(
        "agent.step_2",
        kind=SpanKind.CLIENT,
        attributes={
            "gen_ai.system": "reference-agent",
            "gen_ai.operation.name": "call_tool",
            "agent_reflex.agent.id": "worker",
            "agent_reflex.subtask.id": "task_1",
        },
    ) as span:
        span.set_status(Status(StatusCode.ERROR, "tool timeout after 30s"))

    provider.force_flush()
    wire = json.loads(exporter.spans[0].to_json())

    graph = spans_to_graph([wire])
    node = graph.get_all_nodes()[0]
    assert node.error_flag is True
    assert node.otar.action == "call_tool"


def test_spans_to_graph_reconstructs_parentage_and_subtasks():
    provider, tracer, exporter = _make_tracer()

    parent_ctx = None
    with tracer.start_as_current_span(
        "agent.plan",
        kind=SpanKind.CLIENT,
        attributes={
            "gen_ai.system": "reference-agent",
            "gen_ai.operation.name": "chat",
            "agent_reflex.agent.id": "planner",
            "agent_reflex.subtask.id": "task_1",
        },
    ) as parent:
        parent_ctx = trace.set_span_in_context(parent)
        with tracer.start_as_current_span(
            "agent.verify",
            kind=SpanKind.CLIENT,
            context=parent_ctx,
            attributes={
                "gen_ai.system": "reference-agent",
                "gen_ai.operation.name": "check",
                "agent_reflex.agent.id": "verifier",
                "agent_reflex.subtask.id": "task_2",
            },
        ) as child:
            child.add_event("gen_ai.prompt", {"content": "verify the plan"})
            child.add_event("gen_ai.completion", {"content": "plan is wrong: no deadline"})
            child.set_status(Status(StatusCode.ERROR, "verification failed"))

    provider.force_flush()
    assert len(exporter.spans) == 2

    wires = [json.loads(s.to_json()) for s in exporter.spans]
    graph = spans_to_graph(wires)
    nodes = graph.get_all_nodes()
    assert len(nodes) == 2

    ordered = sorted(nodes, key=lambda n: n.step_index)
    assert ordered[0].subtask_id == "task_1"
    assert ordered[1].subtask_id == "task_2"
    assert ordered[1].error_flag is True
    assert graph.get_parent(ordered[1].node_id) is not None


def test_parse_otlp_json_groups_by_trace():
    provider, tracer, exporter = _make_tracer()
    for i, name in enumerate(["agent.a1", "agent.b1"]):
        with tracer.start_as_current_span(
            name,
            kind=SpanKind.CLIENT,
            attributes={
                "gen_ai.system": "reference-agent",
                "gen_ai.operation.name": "chat",
                "agent_reflex.agent.id": f"agent_{i}",
            },
        ) as span:
            span.set_status(Status(StatusCode.OK))

    provider.force_flush()

    payload = {
        "resourceSpans": [
            {
                "resource": {"attributes": []},
                "scopeSpans": [
                    {
                        "scope": {"name": "test"},
                        "spans": [json.loads(s.to_json()) for s in exporter.spans],
                    }
                ],
            }
        ]
    }
    graphs = parse_otlp_json(payload)
    assert len(graphs) == 2
    for trace_id, graph in graphs:
        assert trace_id != ""
        assert len(graph.get_all_nodes()) == 1
