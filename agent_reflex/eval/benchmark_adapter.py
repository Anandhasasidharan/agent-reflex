"""
Generic benchmark adapter for attribution evaluation.

Defines the common interface: load traces → convert to CausalGraph →
run attribution → report accuracy. Specific benchmark runners
(Who&When, TraceElephant) implement load_benchmark_traces.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_reflex.attribution.engine import AttributionEngine
from agent_reflex.common.types import CausalGraphNode, StepOTAR
from agent_reflex.graph.models import CausalGraph

BenchmarkTrace = dict[str, Any]


def build_graph_from_trace(trace: BenchmarkTrace) -> CausalGraph:
    cg = CausalGraph()
    for step in trace.get("steps", []):
        node = CausalGraphNode(
            node_id=step["id"],
            agent_id=step.get("agent", "unknown"),
            step_index=int(step["id"].split("_")[-1]) if "_" in step["id"] else 0,
            otar=StepOTAR(
                observation=step.get("input", step.get("observation", "")),
                thought=step.get("thought", ""),
                action=step.get("action", ""),
                result=step.get("result", ""),
            ),
            parent_id=step.get("parent"),
            subtask_id=step.get("subtask", "default"),
            execution_time_ms=step.get("execution_time_ms", 100.0),
            error_flag=step.get("error", step.get("error_flag", False)),
        )
        cg.add_step(node)
    cg.infer_data_dependencies()
    return cg


def run_benchmark(
    traces: list[BenchmarkTrace],
    engine: AttributionEngine,
    label: str = "benchmark",
) -> dict[str, Any]:
    mode_correct = 0
    step_correct = 0
    total = len(traces)

    details: list[dict[str, Any]] = []
    for trace in traces:
        graph = build_graph_from_trace(trace)
        result = engine.attribute(
            session_id=f"{label}_{trace.get('name', trace.get('trace_id', 'unknown'))}",
            graph=graph,
            task_context=trace.get("task_context", ""),
        )
        is_mode = result.failure_type.value == trace.get("true_mode", "")
        is_step = result.cause_node_id == trace.get("true_cause", "")
        if is_mode:
            mode_correct += 1
        if is_step:
            step_correct += 1
        details.append({
            "name": trace.get("name", trace.get("trace_id", "")),
            "true_mode": trace.get("true_mode", ""),
            "predicted_mode": result.failure_type.value,
            "mode_correct": is_mode,
            "true_cause": trace.get("true_cause", ""),
            "predicted_cause": result.cause_node_id,
            "step_correct": is_step,
            "crs": round(result.causal_responsibility_score, 2),
        })

    return {
        "label": label,
        "total": total,
        "mode_accuracy_pct": round(mode_correct / total * 100, 1) if total else 0.0,
        "step_accuracy_pct": round(step_correct / total * 100, 1) if total else 0.0,
        "mode_correct": mode_correct,
        "step_correct": step_correct,
        "details": details,
    }


def load_benchmark_traces(path: str, loader_fn: Callable[[str], list[BenchmarkTrace]]) -> list[BenchmarkTrace]:
    return loader_fn(path)
