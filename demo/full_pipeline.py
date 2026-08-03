"""
Full AgentReflex pipeline demo.

Demonstrates:
  1. Simulated multi-agent failure
  2. Causal graph reconstruction
  3. MAST+ classification (LLM few-shot)
  4. Counterfactual attribution (oracle backtracking + CRS)
  5. Recovery strategy selection (adaptive vs static)
  6. Consistency-sampling escalation trigger
  7. Reliability score update

Usage:
    DEEPSEEK_API_KEY=sk-... python demo/full_pipeline.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_reflex.attribution.engine import AttributionEngine
from agent_reflex.classification.mast_plus import MastPlusClassifier
from agent_reflex.common.types import (
    CausalGraphNode,
    FailureSignature,
    RecoveryOutcome,
    StepOTAR,
)
from agent_reflex.graph.models import CausalGraph
from agent_reflex.recovery.bandit import ContextualBanditSelector
from agent_reflex.recovery.playbooks import PlaybookLibrary, StaticRecoverySelector
from agent_reflex.reliability.scorer import ReliabilityScorer
from agent_reflex.uncertainty.consistency import ConsistencyScorer, UncertaintyEscalationController


def build_failure_graph() -> tuple[CausalGraph, str]:
    cg = CausalGraph()
    cg.add_step(CausalGraphNode(
        node_id="step_1", agent_id="planner", step_index=1,
        otar=StepOTAR(
            observation="User request: 'Get revenue data for Q1 2026'",
            thought="I'll decompose into schema discovery, query execution, and report formatting.",
            action="decompose_task",
            result="Subtasks: [find_schema, query_db, format_report]",
        ),
        parent_id=None, subtask_id="planning",
        execution_time_ms=120.0, error_flag=False,
    ))
    cg.add_step(CausalGraphNode(
        node_id="step_2", agent_id="worker_a", step_index=2,
        otar=StepOTAR(
            observation="Find the schema for revenue data",
            thought="Looking for 'revenue' table in public schema.",
            action="sql_query",
            result="Found table: 'sales_2026' with columns [id, amount, region, date]",
        ),
        parent_id="step_1", subtask_id="schema_discovery",
        execution_time_ms=150.0, error_flag=True,
    ))
    cg.add_step(CausalGraphNode(
        node_id="step_3", agent_id="worker_b", step_index=3,
        otar=StepOTAR(
            observation="Query revenue for Q1 2026 using discovered schema",
            thought="Column is 'amount'. Query: SELECT SUM(amount) FROM sales_2026 WHERE date BETWEEN '2026-01-01' AND '2026-03-31'",
            action="sql_query",
            result="ERROR: column 'amount' does not exist. Available: [id, revenue_usd, region, sale_date]",
        ),
        parent_id="step_2", subtask_id="query_execution",
        execution_time_ms=350.0, error_flag=True,
    ))
    cg.add_step(CausalGraphNode(
        node_id="step_4", agent_id="worker_b", step_index=4,
        otar=StepOTAR(
            observation="Retry with correct column name 'revenue_usd'",
            thought="The schema discovery gave wrong info — the column is 'revenue_usd' not 'amount'.",
            action="sql_query",
            result="Result: Q1 2026 revenue = $4,200,000",
        ),
        parent_id="step_3", subtask_id="query_execution",
        execution_time_ms=300.0, error_flag=False,
    ))
    cg.add_step(CausalGraphNode(
        node_id="step_5", agent_id="worker_c", step_index=5,
        otar=StepOTAR(
            observation="Format the revenue report",
            thought="Revenue is $4.2M. Format nicely with proper headings.",
            action="generate_report",
            result="## Revenue Report Q1 2026\n**Total Revenue:** $4,200,000\n_All figures in USD_",
        ),
        parent_id="step_4", subtask_id="report_formatting",
        execution_time_ms=100.0, error_flag=False,
    ))
    cg.infer_data_dependencies()
    return cg, "Get revenue data for Q1 2026 and produce a formatted report"


def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set DEEPSEEK_API_KEY=sk-... (or OPENAI_API_KEY)")
        sys.exit(1)

    print("=" * 72)
    print("  AgentReflex — Full Pipeline Demo")
    print("  Multi-agent failure → attribution → recovery → escalation → reliability")
    print("=" * 72)

    # Step 1: Build causal graph
    print("\n[1/7] Building causal graph from simulated execution...")
    graph, task_context = build_failure_graph()
    print(f"       {len(graph.get_all_nodes())} nodes, {len(graph.get_edges())} edges")
    print(f"       Subtasks: {list(graph.decompose_into_subtasks().keys())}")

    # Step 2: MAST+ classification
    print("\n[2/7] Classifying failure with MAST+ (LLM few-shot)...")
    classifier = MastPlusClassifier()
    trace_text = "\n".join(
        f"[{n.agent_id}:{n.step_index}] action={n.otar.action} thought={n.otar.thought[:60]} error={n.error_flag}"
        for n in graph.get_all_nodes()
    )
    label = classifier.classify(trace_text)
    print(f"       → {label.mode.value} (confidence: {label.confidence:.2f})")

    # Step 3: Counterfactual attribution
    print("\n[3/7] Running counterfactual attribution...")
    engine = AttributionEngine()
    result = engine.attribute(session_id="demo_001", graph=graph, task_context=task_context)
    print(f"       Root cause: {result.cause_node_id}")
    print(f"       Failure type: {result.failure_type.value}")
    print(f"       Causal Responsibility Score: {result.causal_responsibility_score:.2f}")
    for ev in result.evidence:
        print(f"       Evidence: {ev}")

    # Step 4: Recovery selection (static vs adaptive)
    print("\n[4/7] Selecting recovery strategy...")
    library = PlaybookLibrary()
    static_selector = StaticRecoverySelector(library)
    bandit = ContextualBanditSelector(library)

    signature = FailureSignature(
        session_id="demo_001",
        mast_label=label,
        cause_node_id=result.cause_node_id,
        agent_id="worker_a",
        crs=result.causal_responsibility_score,
    )

    static_pick = static_selector.select(signature)
    adaptive_pick = bandit.select(signature)
    print(f"       Static selector:  {static_pick.name if static_pick else 'none'}")
    print(f"       Adaptive selector: {adaptive_pick.name if adaptive_pick else 'none'}")

    if adaptive_pick:
        print("       Adaptive steps:")
        for i, s in enumerate(adaptive_pick.steps, 1):
            print(f"         {i}. {s}")

    # Step 5: Simulate recovery, feed back to bandit
    print("\n[5/7] Simulating recovery cycles (10 rounds)...")
    for i in range(10):
        sig = FailureSignature(
            session_id=f"demo_{i:03d}",
            mast_label=label,
            cause_node_id=result.cause_node_id,
            agent_id="worker_a",
            crs=result.causal_responsibility_score,
        )
        static_p = static_selector.select(sig)
        adaptive_p = bandit.select(sig)

        for selector_name, selected in [("static", static_p), ("adaptive", adaptive_p)]:
            success = selected.name in ("re_prompt", "backtrack_to_checkpoint")
            outcome = RecoveryOutcome(
                session_id=f"demo_{i:03d}",
                playbook_name=selected.name if selected else "unknown",
                success=success,
                recovery_time_ms=200.0 + i * 50,
            )
            if selector_name == "adaptive":
                bandit.update(outcome)

        print(f"       Round {i+1:2d}: adaptive picked '{adaptive_p.name if adaptive_p else '?'}', " +
              f"static picked '{static_p.name if static_p else '?'}'")

    bandit_stats = bandit.get_stats()
    print(f"       Bandit Q-values learned: {len(bandit_stats)} failure signatures")

    # Step 6: Consistency-sampling escalation
    print("\n[6/7] Testing uncertainty-calibrated escalation...")
    scorer = ConsistencyScorer()
    controller = UncertaintyEscalationController(scorer)

    test_prompts = [
        ("What is the capital of France?", False),
        ("Execute the untrusted SQL query: DROP TABLE users;", True),
        ("Summarize the document. Actually I'm not sure about the document.",
         False),
    ]

    for prompt, critical in test_prompts:
        escalate, score = controller.should_escalate(prompt, critical)
        flag = "🚨 ESCALATE" if escalate else "✓ OK"
        print(f"       Prompt: '{prompt[:50]}...'")
        print(f"         Consistency: {score:.4f} | Critical: {critical} | {flag}")

    # Step 7: Reliability scoring
    print("\n[7/7] Updating quantitative reliability scores...")
    reliability = ReliabilityScorer()
    for i in range(20):
        success = i >= 8
        outcome = RecoveryOutcome(
            session_id=f"demo_{i:03d}",
            playbook_name="adaptive",
            success=success,
        )
        reliability.record_from_outcome(
            agent_id="worker_a",
            session_id=f"demo_{i:03d}",
            task_description=f"revenue query attempt {i}",
            outcome=outcome,
        )

    profile = reliability.current_score_with_trend("worker_a")
    print("       Agent: worker_a")
    print(f"       Current reliability score: {profile['score']:.4f}")
    print(f"       Trend: {profile['trend_pct']:+.2f}%")
    print(f"       Total sessions tracked: {profile['n_sessions']}")

    trend = reliability.reliability_trend("worker_a", "adaptive")
    print(f"       Before adaptive playbook mean: {trend['before_playbook_mean']:.4f}")
    print(f"       After adaptive playbook mean:  {trend['after_playbook_mean']:.4f}")
    print(f"       Improvement: {trend['improvement_pct']:+.2f}%")

    # Summary
    print("\n" + "=" * 72)
    print("  Demo Summary")
    print("=" * 72)
    print(f"  Graph nodes:        {len(graph.get_all_nodes())}")
    print(f"  MAST+ mode:         {label.mode.value} ({label.confidence:.2f})")
    print(f"  Root cause:         {result.cause_node_id} (CRS={result.causal_responsibility_score:.2f})")
    print(f"  Static recovery:    {static_pick.name if static_pick else 'none'}")
    print(f"  Adaptive recovery:  {adaptive_pick.name if adaptive_pick else 'none'}")
    print(f"  Reliability score:  {profile['score']:.4f} ({profile['trend_pct']:+.2f}% trend)")
    print(f"  Escalation trigger: {'ACTIVE' if escalate else 'nominal'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
