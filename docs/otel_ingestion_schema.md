# AgentReflex OTel Ingestion Schema (Contract)

This document is the **contract** between any instrumented agent producer and
AgentReflex's ingestion pipeline. It is written so that someone can instrument
a brand-new agent framework from this document alone, without reading
AgentReflex source.

AgentReflex is framework-agnostic: it consumes spans that follow the
OpenTelemetry **GenAI semantic conventions**
(`gen_ai.*`, see
[OpenTelemetry GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/)),
plus a small documented set of `agent_reflex.*` extension attributes for the
OTAR (Observation–Thought–Action–Result) fields that GenAI semconv does not
define. Any producer emitting these attributes works — LangGraph, CrewAI, a
raw SDK script, or a bespoke framework.

Transport: **OTLP/HTTP JSON** (`POST /v1/traces`) or OTLP/gRPC. The reference
path used by the demo is: producer → OTel SDK → OTel Collector → AgentReflex.

---

## 1. Span model

Each **step** of an agent run must be exported as one **span** of
`kind = CLIENT` (or INTERNAL). Spans of one run share a `traceId`; the run is
identified by `traceId` at ingestion.

| OTLP span field | Meaning in AgentReflex | Required |
|-----------------|------------------------|----------|
| `traceId` | session identifier | yes |
| `spanId` | node identifier | yes |
| `parentSpanId` | topology edge to previous step | no (root if absent) |
| `name` | human-readable step name (e.g. `agent.step_2`) | yes |
| `startTimeUnixNano` / `endTimeUnixNano` | step duration → `execution_time_ms` | yes |
| `status.code` | `2` (ERROR) ⇒ `error_flag = true`; `1`/`0` ⇒ false | yes |
| `status.message` | error summary (optional) | no |
| `attributes` | see §2 | see §2 |
| `events` | `gen_ai.prompt` / `gen_ai.completion` events (§3) | no |

`error_flag` drives root-cause candidates in attribution: a step whose span
has `status.code == 2` is a failure candidate.

---

## 2. Span attributes

AgentReflex reads the following **standard GenAI semconv** attributes
(`gen_ai.*`). Attribute keys follow the semconv naming exactly; values follow
semconv types.

| Attribute | Type | Required | Maps to |
|-----------|------|----------|---------|
| `gen_ai.system` | string | yes | provider/framework name (e.g. `openai`, `langgraph`, `crewai`) |
| `gen_ai.operation.name` | string | yes | OTAR **action** (what the step does: `chat`, `call_tool`, `retrieve`, …) |
| `gen_ai.request.model` | string | no | model name (e.g. `deepseek-v4-flash`) |
| `gen_ai.request.temperature` | double | no | sampling temperature |
| `gen_ai.input.messages` | JSON string (array of message objects) | no* | OTAR **observation** |
| `gen_ai.output.messages` | JSON string (array of message objects) | no* | OTAR **result** |
| `gen_ai.request.prompt` | string | no* | OTAR **observation** (single-prompt form) |
| `gen_ai.completion` | string | no* | OTAR **result** (single-completion form) |
| `gen_ai.usage.input_tokens` | int | no | token accounting |
| `gen_ai.usage.output_tokens` | int | no | token accounting |

\* Either `gen_ai.input.messages` / `gen_ai.output.messages` **or** the
corresponding `gen_ai.prompt`/`gen_ai.completion` span events (§3) must be
present for a step to carry content. Steps without any input/output are still
ingested (structural-only nodes) but contribute no OTAR signal.

### AgentReflex extension attributes (`agent_reflex.*`)

These cover OTAR fields the GenAI semconv does not define. They are optional
except where noted; every one is namespaced so producers can ignore them and
still produce a valid ingestible trace.

| Attribute | Type | Required | Maps to |
|-----------|------|----------|---------|
| `agent_reflex.agent.id` | string | yes | OTAR/agent identity (`agent_id`) |
| `agent_reflex.agent.thought` | string | no | OTAR **thought** (reasoning text) |
| `agent_reflex.subtask.id` | string | no | subtask grouping; steps with the same value form one subtask (default: span `name`) |
| `agent_reflex.step.index` | int | no | explicit step ordering (else by start time) |

Producers that only emit standard GenAI semconv and no `agent_reflex.*`
attributes are still accepted; the pipeline infers `agent_id` from
`gen_ai.system` and subtasks from span names.

---

## 3. Span events

For content, producers may alternatively (or additionally) emit two standard
semconv event names on the step span:

| Event name | Event attribute | Maps to |
|------------|-----------------|---------|
| `gen_ai.prompt` | `content` | OTAR **observation** |
| `gen_ai.completion` | `content` | OTAR **result** |

Event attributes take precedence over span attributes when both carry content
for the same OTAR field.

---

## 4. OTAR reconciliation

The OTAR model is built per span as:

| OTAR field | Source priority |
|------------|-----------------|
| `observation` | event `gen_ai.prompt.content` → attr `gen_ai.input.messages` → attr `gen_ai.request.prompt` |
| `thought` | attr `agent_reflex.agent.thought` (no semconv equivalent — extension only) |
| `action` | attr `gen_ai.operation.name` (falls back to span `name`) |
| `result` | event `gen_ai.completion.content` → attr `gen_ai.output.messages` → attr `gen_ai.completion` |

Message-array attributes (`gen_ai.input.messages`, `gen_ai.output.messages`)
are transported as JSON strings of the form
`[{"role":"user","content":"..."}, ...]`; the parser extracts the concatenated
`content` of the array.

---

## 5. Causal graph construction (ingestion side)

1. Spans are grouped by `traceId` → one session per trace.
2. Nodes are created in start-time order; `step_index` is the order position
   (overridden by `agent_reflex.step.index` if present).
3. Edges: `parentSpanId` → `control_flow` edge; additionally
   `CausalGraph.infer_data_dependencies()` adds `data_dependency` edges.
4. `subtask_id` groups spans; the attribution engine's oracle verifies each
   subtask independently.
5. `error_flag` = `status.code == 2`.

## 6. Verified wire-format test

`tests/test_otel_ingestion.py` builds a span with the OTel Python SDK
(`TracerProvider` + `SimpleSpanProcessor` + in-memory exporter), exports the
real `ReadableSpan`, serializes it with `ReadableSpan.to_json()`, and feeds
that wire representation through `OTARParser` — proving the parser handles
the actual SDK export format, not a hand-built dict.

## 7. Example minimal span (illustrative)

```json
{
  "traceId": "aa00112233445566778899aabbccddee",
  "spanId": "bb00112233445566",
  "parentSpanId": "",
  "name": "agent.plan",
  "kind": 2,
  "startTimeUnixNano": "1720000000000000000",
  "endTimeUnixNano": "1720000002500000000",
  "attributes": [
    {"key": "gen_ai.system", "value": {"stringValue": "langgraph"}},
    {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
    {"key": "gen_ai.request.model", "value": {"stringValue": "deepseek-v4-flash"}},
    {"key": "agent_reflex.agent.id", "value": {"stringValue": "planner"}},
    {"key": "agent_reflex.subtask.id", "value": {"stringValue": "task_1"}}
  ],
  "events": [
    {"name": "gen_ai.prompt", "attributes": [{"key": "content", "value": {"stringValue": "Plan the release"}}]},
    {"name": "gen_ai.completion", "attributes": [{"key": "content", "value": {"stringValue": "Plan: 1) Freeze scope 2) Ship"}}]}
  ],
  "status": {"code": 1}
}
```
