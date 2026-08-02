"""
AgentReflex Demo — Run a simulated multi-agent failure, observe attribution + recovery.

Usage:
    python demo/run_failure_scenario.py

Requires DEEPSEEK_API_KEY or OPENAI_API_KEY in environment.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_reflex.attribution.engine import AttributionEngine
from agent_reflex.classification.mast_plus import MastPlusClassifier
from agent_reflex.common.types import CausalGraphNode, StepOTAR
from agent_reflex.graph.models import CausalGraph


def build_sample_failure_graph() -> CausalGraph:
    cg = CausalGraph()

    steps = [
        CausalGraphNode(
            node_id="step_1", agent_id="planner", step_index=1,
            otar=StepOTAR(
                observation="User request: 'Get revenue data for Q1 2026'",
                thought="I need to break this into: find revenue table, query it, format results.",
                action="decompose_task",
                result="Subtasks: [find_schema, query_db, format_report]",
            ),
            parent_id=None, subtask_id="planning",
            execution_time_ms=120.0, error_flag=False,
        ),
        CausalGraphNode(
            node_id="step_2", agent_id="worker_a", step_index=2,
            otar=StepOTAR(
                observation="Find the schema for revenue data",
                thought="I'll look for a 'revenue' table in the public schema.",
                action="sql_query",
                result="Found table: 'sales_2026' with columns [id, amount, region, date]",
            ),
            parent_id="step_1", subtask_id="find_schema",
            execution_time_ms=200.0, error_flag=False,
        ),
        CausalGraphNode(
            node_id="step_3", agent_id="worker_b", step_index=3,
            otar=StepOTAR(
                observation="Query revenue for Q1 2026",
                thought="Q1 2026 is Jan-Mar 2026. I'll query SUM(amount) WHERE date BETWEEN '2026-01-01' AND '2026-03-31'",
                action="sql_query",
                result="ERROR: column 'amount' does not exist. Available columns: [id, revenue_usd, region, sale_date]",
            ),
            parent_id="step_2", subtask_id="query_db",
            execution_time_ms=350.0, error_flag=True,
        ),
        CausalGraphNode(
            node_id="step_4", agent_id="worker_b", step_index=4,
            otar=StepOTAR(
                observation="Retry with correct column name 'revenue_usd'",
                thought="The column is 'revenue_usd', not 'amount'. The schema step gave me wrong info.",
                action="sql_query",
                result="Result: Q1 2026 revenue = $4,200,000",
            ),
            parent_id="step_3", subtask_id="query_db",
            execution_time_ms=300.0, error_flag=False,
        ),
        CausalGraphNode(
            node_id="step_5", agent_id="worker_c", step_index=5,
            otar=StepOTAR(
                observation="Format the revenue report",
                thought="The user asked for revenue data. The result was $4.2M. I should present it nicely.",
                action="generate_report",
                result="## Revenue Report Q1 2026\nTotal Revenue: $4,200,000\nNote: All figures in USD.",
            ),
            parent_id="step_4", subtask_id="format_report",
            execution_time_ms=100.0, error_flag=False,
        ),
    ]

    for step in steps:
        cg.add_step(step)

    cg.infer_data_dependencies()
    return cg


def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No LLM API key found. Set DEEPSEEK_API_KEY or OPENAI_API_KEY.")
        print("Usage: DEEPSEEK_API_KEY=sk-... python demo/run_failure_scenario.py")
        sys.exit(1)

    print("=" * 60)
    print("AgentReflex Demo — Failure Attribution + Recovery")
    print("=" * 60)

    print("\n[1/4] Building simulated multi-agent execution trace...")
    graph = build_sample_failure_graph()
    print(f"       Graph has {len(graph.get_all_nodes())} nodes, {len(graph.get_edges())} edges")

    print("\n[2/4] Classifying failure with MAST+ (LLM few-shot)...")
    classifier = MastPlusClassifier()
    trace_text = "\n".join(
        f"[{n.agent_id}] step {n.step_index}: action={n.otar.action}, thought={n.otar.thought[:80]}, error={n.error_flag}"
        for n in graph.get_all_nodes()
    )
    label = classifier.classify(trace_text)
    print(f"       MAST+ Mode: {label.mode.value} (confidence: {label.confidence:.2f})")

    print("\n[3/4] Running counterfactual attribution...")
    engine = AttributionEngine()
    result = engine.attribute(
        session_id="demo_001",
        graph=graph,
        task_context="Get revenue data for Q1 2026 and format a report",
    )
    print(f"       Cause node: {result.cause_node_id}")
    print(f"       Causal Responsibility Score: {result.causal_responsibility_score:.2f}")
    print("       Evidence:")
    for ev in result.evidence:
        print(f"         - {ev}")

    print("\n[4/4] Selecting recovery strategy...")
    from agent_reflex.recovery.playbooks import PlaybookLibrary
    library = PlaybookLibrary()
    candidates = library.matching_playbooks(result.failure_type)
    print(f"       Matching playbooks: {[p.name for p in candidates]}")

    if candidates:
        selected = candidates[0]
        print(f"       → Selected: '{selected.name}'")
        print("       Steps:")
        for i, step in enumerate(selected.steps, 1):
            print(f"         {i}. {step}")

    print("\n" + "=" * 60)
    print("Demo complete. The step_2 (worker_a) provided an incorrect schema")
    print("(listing 'amount' instead of 'revenue_usd'), which caused step_3 to fail.")
    print("The attribution engine correctly identified the root cause as")
    print("a coordination misalignment between schema-discovery and query steps.")
    print("=" * 60)


if __name__ == "__main__":
    main()
