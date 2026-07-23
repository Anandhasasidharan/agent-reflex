"""
Who&When-style evaluation for AgentReflex attribution accuracy.

This module provides a framework for evaluating attribution accuracy
on a labeled test set. It supports both synthetic traces (where the
ground-truth cause is known by construction) and annotated traces
following the Who&When benchmark format.

Usage:
    python -m agent_reflex.eval.benchmark --traces data/eval_traces.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class EvalExample:
    trace_id: str
    graph_json: str
    true_failure_mode: str
    true_cause_node_id: str
    task_context: str = ""


@dataclass
class EvalResult:
    mode_correct: bool
    step_correct: bool
    predicted_crs: float
    true_mode: str
    predicted_mode: str
    predicted_cause: str


def load_eval_set(path: str) -> list[EvalExample]:
    with open(path) as f:
        data = json.load(f)
    return [EvalExample(**item) for item in data]


def build_synthetic_eval_set() -> list[EvalExample]:
    from agent_reflex.common.types import CausalGraphNode, StepOTAR
    from agent_reflex.graph.models import CausalGraph

    examples: list[EvalExample] = []

    cg = CausalGraph()
    cg.add_step(CausalGraphNode(
        node_id="s1", agent_id="a", step_index=1,
        otar=StepOTAR("input", "think", "search", "wrong_column_name"),
        parent_id=None, subtask_id="t1", execution_time_ms=100.0, error_flag=True,
    ))
    cg.add_step(CausalGraphNode(
        node_id="s2", agent_id="b", step_index=2,
        otar=StepOTAR("use wrong_column_name", "think", "query", "ERROR"),
        parent_id="s1", subtask_id="t1", execution_time_ms=200.0, error_flag=True,
    ))
    examples.append(EvalExample(
        trace_id="synth_coord_misalignment",
        graph_json=cg.to_json(),
        true_failure_mode="coord_misaligned_assumptions",
        true_cause_node_id="s1",
        task_context="Find and query the revenue data",
    ))

    cg2 = CausalGraph()
    cg2.add_step(CausalGraphNode(
        node_id="s1", agent_id="a", step_index=1,
        otar=StepOTAR("user query", "think", "respond", "fabricated fact"),
        parent_id=None, subtask_id="t1", execution_time_ms=100.0, error_flag=True,
    ))
    examples.append(EvalExample(
        trace_id="synth_hallucination",
        graph_json=cg2.to_json(),
        true_failure_mode="task_hallucination",
        true_cause_node_id="s1",
        task_context="Answer the user's question factually",
    ))

    cg3 = CausalGraph()
    cg3.add_step(CausalGraphNode(
        node_id="s1", agent_id="a", step_index=1,
        otar=StepOTAR("api call", "think", "http_request", "429 Too Many Requests"),
        parent_id=None, subtask_id="t1", execution_time_ms=50.0, error_flag=True,
    ))
    examples.append(EvalExample(
        trace_id="synth_rate_limit",
        graph_json=cg3.to_json(),
        true_failure_mode="infra_rate_limit",
        true_cause_node_id="s1",
        task_context="Fetch data from external API",
    ))

    return examples
