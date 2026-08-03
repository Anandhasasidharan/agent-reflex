"""Build CausalGraphs from real OTLP span exports.

Accepts both the OTLP/HTTP JSON wire format (`ExportTraceServiceRequest`:
{"resourceSpans": [...]}) and the OTel Python SDK's `ReadableSpan.to_json()`
format. This is the ingestion-side entry point — no hand-built graph JSON
ever reaches the pipeline from here.
"""

from __future__ import annotations

import json
from typing import Any

from agent_reflex.common.types import CausalGraphNode, StepOTAR
from agent_reflex.graph.models import CausalGraph, OTARParser

_T0_NS = 1_700_000_000_000_000_000


def _str_of(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _extract_status(status: Any) -> tuple[bool, str]:
    """Return (error_flag, message) from an OTLP status object.

    Handles both the OTLP wire format ({"code": 2}) and the OTel SDK's
    ReadableSpan.to_json() format ({"status_code": "ERROR", "description"}).
    """
    if isinstance(status, dict):
        code = status.get("code")
        if code in (2, "2"):
            return True, _str_of(status.get("message"))
        sdk_code = status.get("status_code")
        if isinstance(sdk_code, str) and "ERROR" in sdk_code.upper():
            return True, _str_of(status.get("description") or status.get("message"))
    return False, ""


def _extract_events(span: dict[str, Any]) -> list[dict[str, Any]]:
    events = span.get("events", [])
    if not isinstance(events, list):
        return []
    normalized = []
    for event in events:
        if isinstance(event, dict):
            attrs = event.get("attributes", {})
            if isinstance(attrs, list):
                attrs = {
                    a.get("key"): _otlp_value_str(a.get("value"))
                    for a in attrs
                    if isinstance(a, dict)
                }
            elif not isinstance(attrs, dict):
                attrs = {}
            normalized.append({"name": event.get("name", ""), "attributes": attrs})
    return normalized


def _otlp_value_str(value: Any) -> str:
    """Unwrap an OTLP AnyValue dict ({'stringValue': ...}) to a plain string."""
    if isinstance(value, dict):
        for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
            if value.get(key) is not None:
                return str(value[key])
        return ""
    return _str_of(value)


def _span_identity(span: dict[str, Any]) -> tuple[str, str, str | None]:
    context = span.get("context", {})
    trace_id = context.get("trace_id", span.get("traceId", ""))
    span_id = context.get("span_id", span.get("spanId", ""))
    parent_id = span.get("parent_id") or span.get("parentSpanId") or context.get("parent_id")
    if isinstance(parent_id, str) and (parent_id == "null" or not parent_id):
        parent_id = None
    return _str_of(trace_id), _str_of(span_id), (str(parent_id) if parent_id else None)


def _start_time_ns(span: dict[str, Any]) -> int:
    raw = span.get("start_time") or span.get("startTimeUnixNano")
    if raw is None:
        return _T0_NS
    if isinstance(raw, (int, float)) and raw > 1e15:
        return int(raw)
    if isinstance(raw, str):
        try:
            from datetime import datetime

            return int(datetime.fromisoformat(raw.replace("Z", "+00:00"))
                       .timestamp() * 1e9)
        except ValueError:
            pass
    return _T0_NS


def _duration_ms(start_ns: int, span: dict[str, Any]) -> float:
    end_raw = span.get("end_time") or span.get("endTimeUnixNano")
    end_ns = start_ns
    if isinstance(end_raw, (int, float)) and end_raw > 1e15:
        end_ns = int(end_raw)
    elif isinstance(end_raw, str):
        try:
            from datetime import datetime

            end_ns = int(datetime.fromisoformat(end_raw.replace("Z", "+00:00"))
                         .timestamp() * 1e9)
        except ValueError:
            end_ns = start_ns
    return max(0.0, (end_ns - start_ns) / 1e6)


def _attributes_dict(span: dict[str, Any]) -> dict[str, Any]:
    attrs = span.get("attributes", {})
    if isinstance(attrs, list):
        normalized: dict[str, Any] = {}
        for item in attrs:
            if isinstance(item, dict) and "key" in item:
                value = item.get("value", {})
                if isinstance(value, dict):
                    normalized[item["key"]] = (
                        value.get("stringValue")
                        or value.get("intValue")
                        or value.get("doubleValue")
                        or value.get("boolValue")
                        or ""
                    )
                else:
                    normalized[item["key"]] = value
        return normalized
    if isinstance(attrs, dict):
        return attrs
    return {}


def spans_to_graph(spans: list[dict[str, Any]], trace_id: str = "") -> CausalGraph:
    """Build a CausalGraph from a list of span dicts (SDK to_json or OTLP).

    Spans are ordered by start time; parent relationships are reconstructed
    from parent span ids; subtasks group by agent_reflex.subtask.id.
    """
    graph = CausalGraph()
    ordered = sorted(spans, key=lambda s: _start_time_ns(s))
    nodes: dict[str, CausalGraphNode] = {}

    for position, span in enumerate(ordered, start=1):
        span_trace, span_id, parent_id = _span_identity(span)
        attrs = _attributes_dict(span)
        events = _extract_events(span)
        otar: StepOTAR = OTARParser.parse(attrs, events, span_name=_str_of(span.get("name")))
        error_flag, error_message = _extract_status(span.get("status", {}))

        node_id = span_id or _str_of(span.get("name", "")) or f"step_{position}"
        subtask = _str_of(attrs.get("agent_reflex.subtask.id")) or _str_of(span.get("name"))
        explicit_index = attrs.get("agent_reflex.step.index")
        step_index = position
        if isinstance(explicit_index, (int, float)):
            step_index = int(explicit_index)

        node = CausalGraphNode(
            node_id=node_id,
            agent_id=_str_of(attrs.get("agent_reflex.agent.id")) or _str_of(attrs.get("gen_ai.system")) or "unknown",
            step_index=step_index,
            otar=otar,
            parent_id=parent_id,
            subtask_id=subtask or None,
            execution_time_ms=_duration_ms(_start_time_ns(span), span),
            error_flag=error_flag,
            raw_span_attributes=dict(attrs),
        )
        if error_message:
            node.raw_span_attributes["error.message"] = error_message
        nodes[span_id or node_id] = node
        graph.add_step(node)

    # Re-link parents when parent spans are present in the same export.
    for span in ordered:
        _, span_id, parent_id = _span_identity(span)
        if parent_id and parent_id in nodes:
            graph.add_dependency(parent_id, span_id or "")
    graph.infer_data_dependencies()
    return graph


def parse_otlp_json(payload: dict[str, Any]) -> list[tuple[str, CausalGraph]]:
    """Parse an OTLP ExportTraceServiceRequest JSON payload.

    Returns a list of (trace_id, CausalGraph) — one graph per trace.
    """
    graphs: list[tuple[str, CausalGraph]] = []
    for resource_span in payload.get("resourceSpans", []):
        for scope_span in resource_span.get("scopeSpans", []):
            spans = scope_span.get("spans", [])
            if not spans:
                continue
            by_trace: dict[str, list[dict[str, Any]]] = {}
            for span in spans:
                if not isinstance(span, dict):
                    continue
                context = span.get("context", {})
                trace_id = _str_of(context.get("trace_id") or span.get("traceId"))
                by_trace.setdefault(trace_id or "unknown", []).append(span)
            for trace_id, trace_spans in by_trace.items():
                graphs.append((trace_id, spans_to_graph(trace_spans, trace_id=trace_id)))
    return graphs


def parse_otlp_json_str(raw: str) -> list[tuple[str, CausalGraph]]:
    return parse_otlp_json(json.loads(raw))


def _proto_spans_to_payload_dict(request: Any) -> dict[str, Any]:
    """Convert an ExportTraceServiceRequest protobuf message into the
    OTLP JSON dict shape used by parse_otlp_json."""
    payload: dict[str, Any] = {"resourceSpans": []}
    for rs in request.resource_spans:
        rs_dict: dict[str, Any] = {"resource": {"attributes": []}, "scopeSpans": []}
        if rs.resource is not None:
            rs_dict["resource"]["attributes"] = [
                {"key": a.key, "value": _proto_anyvalue(a.value)}
                for a in rs.resource.attributes
            ]
        for ss in rs.scope_spans:
            ss_dict: dict[str, Any] = {"scope": {"name": ss.scope.name if ss.scope else ""}, "spans": []}
            for span in ss.spans:
                span_dict: dict[str, Any] = {
                    "traceId": span.trace_id.hex(),
                    "spanId": span.span_id.hex(),
                    "parentSpanId": span.parent_span_id.hex() if span.parent_span_id else "",
                    "name": span.name,
                    "kind": span.kind,
                    "startTimeUnixNano": str(span.start_time_unix_nano),
                    "endTimeUnixNano": str(span.end_time_unix_nano),
                    "attributes": [
                        {"key": a.key, "value": _proto_anyvalue(a.value)}
                        for a in span.attributes
                    ],
                    "events": [
                        {
                            "name": e.name,
                            "attributes": [
                                {"key": a.key, "value": _proto_anyvalue(a.value)}
                                for a in e.attributes
                            ],
                        }
                        for e in span.events
                    ],
                    "status": {"code": span.status.code},
                }
                ss_dict["spans"].append(span_dict)
            rs_dict["scopeSpans"].append(ss_dict)
        payload["resourceSpans"].append(rs_dict)
    return payload


def _proto_anyvalue(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    kind = value.WhichOneof("value")
    if kind is None:
        return {}
    if kind in ("string_value",):
        return {"stringValue": value.string_value}
    if kind == "int_value":
        return {"intValue": value.int_value}
    if kind == "double_value":
        return {"doubleValue": value.double_value}
    if kind == "bool_value":
        return {"boolValue": value.bool_value}
    return {}


def parse_otlp_protobuf(raw: bytes) -> list[tuple[str, CausalGraph]]:
    """Parse a real OTLP/HTTP protobuf payload (ExportTraceServiceRequest)."""
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    request = ExportTraceServiceRequest()
    request.ParseFromString(raw)
    payload = _proto_spans_to_payload_dict(request)
    return parse_otlp_json(payload)
