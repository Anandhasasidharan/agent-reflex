"""
Eval runner for AgentReflex attribution accuracy.

Generates synthetic traces with known ground-truth root causes,
runs the full attribution pipeline, and reports accuracy metrics.
"""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agent_reflex.attribution.engine import AttributionEngine
from agent_reflex.common.types import CausalGraphNode, StepOTAR
from agent_reflex.graph.models import CausalGraph

SYNTHETIC_SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "coord_misaligned_assumptions",
        "true_mode": "coord_misaligned_assumptions",
        "true_cause": "step_1",
        "task_context": "Find revenue data and produce a report",
        "steps": [
            {"id": "step_1", "agent": "worker_a", "input": "Find the revenue table schema",
             "thought": "I'll look for a 'revenue' table", "action": "sql_query",
             "result": "Found table with columns [id, amount, region, date]",
             "error": True, "parent": None, "subtask": "schema_discovery"},
            {"id": "step_2", "agent": "worker_b", "input": "Query revenue using the discovered schema",
             "thought": "Using column 'amount' from schema discovery", "action": "sql_query",
             "result": "ERROR: column 'amount' does not exist. Columns are [revenue_usd]",
             "error": True, "parent": "step_1", "subtask": "execute_query"},
            {"id": "step_3", "agent": "worker_b", "input": "Retry with correct column",
             "thought": "Using 'revenue_usd' instead", "action": "sql_query",
             "result": "Revenue: $4.2M", "error": False, "parent": "step_2", "subtask": "execute_query"},
        ],
    },
    {
        "name": "spec_ambiguous",
        "true_mode": "spec_ambiguous",
        "true_cause": "step_1",
        "task_context": "Query the database for user growth metrics",
        "steps": [
            {"id": "step_1", "agent": "planner", "input": "Query the database",
             "thought": "User said 'query the database' without specifying SQL or API",
             "action": "choose_method", "result": "I'll use the REST API instead of SQL",
             "error": True, "parent": None, "subtask": "planning"},
            {"id": "step_2", "agent": "worker", "input": "Call REST API endpoint",
             "thought": "Calling /api/users endpoint", "action": "http_request",
             "result": "API returns 200 with user list", "error": False,
             "parent": "step_1", "subtask": "execution"},
        ],
    },
    {
        "name": "task_hallucination",
        "true_mode": "task_hallucination",
        "true_cause": "step_1",
        "task_context": "Find the CEO's name from the company financial report",
        "steps": [
            {"id": "step_1", "agent": "analyst", "input": "Who is the CEO?",
             "thought": "I recall the CEO is Satya Nadella from a previous conversation",
             "action": "generate_response",
             "result": "The CEO is Satya Nadella (Microsoft was mentioned in the chat history, but the report is about a different company)",
             "error": True, "parent": None, "subtask": "research"},
        ],
    },
    {
        "name": "infra_rate_limit",
        "true_mode": "infra_rate_limit",
        "true_cause": "step_1",
        "task_context": "Fetch user data from external API for 1000 users",
        "steps": [
            {"id": "step_1", "agent": "fetcher", "input": "Fetch user batch 1/10",
             "thought": "Making parallel requests to /api/users?page=1", "action": "http_request",
             "result": "429 Too Many Requests — rate limit exceeded",
             "error": True, "parent": None, "subtask": "data_fetch"},
            {"id": "step_2", "agent": "fetcher", "input": "Retry with backoff",
             "thought": "Waiting 60s then retrying", "action": "http_request",
             "result": "429 again — still rate limited", "error": True,
             "parent": "step_1", "subtask": "data_fetch"},
        ],
    },
    {
        "name": "verif_overconfident",
        "true_mode": "verif_overconfident",
        "true_cause": "step_1",
        "task_context": "Calculate total revenue for Q1 2026",
        "steps": [
            {"id": "step_1", "agent": "calculator", "input": "Q1 revenue = Jan + Feb + Mar",
             "thought": "I remember the numbers: Jan=1.2M, Feb=1.3M, Mar=1.5M. Total=4.0M",
             "action": "calculate", "result": "Total revenue: $4.0M (confidence: 99%)",
             "error": True, "parent": None, "subtask": "calculation"},
        ],
    },
    {
        "name": "infra_context_window",
        "true_mode": "infra_context_window",
        "true_cause": "step_3",
        "task_context": "Analyze this 500-page document and summarize key findings",
        "steps": [
            {"id": "step_1", "agent": "reader", "input": "Read document pages 1-50",
             "thought": "Extracting key info from first 50 pages", "action": "read",
             "result": "Content from pages 1-50 extracted", "error": False,
             "parent": None, "subtask": "reading"},
            {"id": "step_2", "agent": "reader", "input": "Read pages 51-100",
             "thought": "Continuing extraction", "action": "read",
             "result": "Content extracted", "error": False,
             "parent": "step_1", "subtask": "reading"},
            {"id": "step_3", "agent": "reader", "input": "Read pages 101-150",
             "thought": "Context is getting very long...", "action": "read",
             "result": "ERROR: maximum context length exceeded (180K tokens)",
             "error": True, "parent": "step_2", "subtask": "reading"},
        ],
    },
]


def build_graph_from_scenario(scenario: dict[str, Any]) -> CausalGraph:
    cg = CausalGraph()
    for step in scenario["steps"]:
        node = CausalGraphNode(
            node_id=step["id"],
            agent_id=step["agent"],
            step_index=int(step["id"].split("_")[1]),
            otar=StepOTAR(
                observation=step["input"],
                thought=step["thought"],
                action=step["action"],
                result=step["result"],
            ),
            parent_id=step.get("parent"),
            subtask_id=step.get("subtask"),
            execution_time_ms=100.0,
            error_flag=step.get("error", False),
        )
        cg.add_step(node)
    cg.infer_data_dependencies()
    return cg


def run_eval(api_key: str | None = None) -> dict[str, Any]:
    if api_key:
        os.environ["AGENT_REFLEX_OPENAI_API_KEY"] = api_key
    elif not os.environ.get("AGENT_REFLEX_OPENAI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: No OpenAI API key found. Set OPENAI_API_KEY env var.")
        print("Running in structure-only mode (no LLM calls).")
        return {"error": "no_api_key", "note": "Set OPENAI_API_KEY to run actual eval"}

    engine = AttributionEngine()

    results: list[dict[str, Any]] = []
    mode_correct = 0
    step_correct = 0
    total = len(SYNTHETIC_SCENARIOS)

    for scenario in SYNTHETIC_SCENARIOS:
        print(f"  Running: {scenario['name']}...", end=" ")
        graph = build_graph_from_scenario(scenario)
        result = engine.attribute(
            session_id=f"eval_{scenario['name']}",
            graph=graph,
            task_context=scenario["task_context"],
        )

        is_mode_correct = result.failure_type.value == scenario["true_mode"]
        is_step_correct = result.cause_node_id == scenario["true_cause"]

        results.append({
            "scenario": scenario["name"],
            "true_mode": scenario["true_mode"],
            "predicted_mode": result.failure_type.value,
            "mode_correct": is_mode_correct,
            "true_cause": scenario["true_cause"],
            "predicted_cause": result.cause_node_id,
            "step_correct": is_step_correct,
            "crs": round(result.causal_responsibility_score, 2),
            "evidence": result.evidence,
        })

        if is_mode_correct:
            mode_correct += 1
        if is_step_correct:
            step_correct += 1

        print(f"{'✓' if is_mode_correct else '✗'} mode, {'✓' if is_step_correct else '✗'} step")

    mode_accuracy = mode_correct / total * 100
    step_accuracy = step_correct / total * 100

    return {
        "total_scenarios": total,
        "mode_accuracy_pct": round(mode_accuracy, 1),
        "step_accuracy_pct": round(step_accuracy, 1),
        "mode_correct": mode_correct,
        "step_correct": step_correct,
        "details": results,
    }


def print_confusion_matrix(results: list[dict[str, Any]]) -> None:
    modes = sorted(set(r["true_mode"] for r in results) | set(r["predicted_mode"] for r in results))
    print(f"\n{'':30s}", end="")
    for m in modes:
        print(f"{m[:20]:20s}", end="")
    print()
    for true in modes:
        print(f"{true[:30]:30s}", end="")
        for pred in modes:
            count = sum(1 for r in results if r["true_mode"] == true and r["predicted_mode"] == pred)
            print(f"{str(count):20s}", end="")
        print()


def main() -> None:
    print("=" * 60)
    print("AgentReflex Eval — Attribution Accuracy Report")
    print("=" * 60)
    print()
    print(f"Test set: {len(SYNTHETIC_SCENARIOS)} synthetic scenarios")
    print()

    results = run_eval()
    if "error" in results:
        print(f"\n{results['error']}: {results['note']}")
        return

    print("\nResults:")
    print(f"  Mode-level accuracy:  {results['mode_accuracy_pct']:.1f}% ({results['mode_correct']}/{results['total_scenarios']})")
    print(f"  Step-level accuracy:  {results['step_accuracy_pct']:.1f}% ({results['step_correct']}/{results['total_scenarios']})")
    print()

    print("Per-scenario breakdown:")
    print(f"{'Scenario':30s} {'True Mode':25s} {'Predicted':25s} {'CRS':6s} {'Match':6s}")
    print("-" * 100)
    for d in results["details"]:
        match = "✓" if d["mode_correct"] else "✗"
        print(f"{d['scenario']:30s} {d['true_mode']:25s} {d['predicted_mode']:25s} {d['crs']:<6.2f} {match:6s}")

    print_confusion_matrix(results["details"])

    print()
    print("=" * 60)
    print("Note: These are synthetic scenarios designed to test specific")
    print("failure modes. Real-world Who&When benchmark results will differ.")
    print("=" * 60)


if __name__ == "__main__":
    main()
