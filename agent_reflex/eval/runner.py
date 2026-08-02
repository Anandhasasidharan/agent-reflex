"""
Eval runner for AgentReflex attribution accuracy.

Generates synthetic traces with known ground-truth root causes,
runs the full attribution pipeline, and reports accuracy metrics.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from agent_reflex.attribution.engine import AttributionEngine
from agent_reflex.common.config import Settings
from agent_reflex.common.llm import resolve_api_key
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
    {
        "name": "spec_incomplete",
        "true_mode": "spec_incomplete",
        "true_cause": "step_1",
        "task_context": "Analyze the Q1 sales data file",
        "steps": [
            {"id": "step_1", "agent": "analyst", "input": "Analyze the Q1 sales data file",
             "thought": "No specific metrics or output format were given, so I'll pick my own",
             "action": "analyze",
             "result": "Produced a chart of raw sales numbers with totals missing",
             "error": True, "parent": None, "subtask": "analysis"},
            {"id": "step_2", "agent": "reviewer", "input": "Validate the analysis output",
             "thought": "The output has no totals, no trend, and no required KPIs — nothing was specified",
             "action": "validate",
             "result": "REJECTED: analysis incomplete against unspecified requirements",
             "error": True, "parent": "step_1", "subtask": "review"},
        ],
    },
    {
        "name": "spec_contradictory",
        "true_mode": "spec_contradictory",
        "true_cause": "step_1",
        "task_context": "Produce a concise executive summary",
        "steps": [
            {"id": "step_1", "agent": "writer", "input": "Write an executive summary",
             "thought": "Instructions said both 'be brief, 1 paragraph' and 'include all 40 metrics with full detail'",
             "action": "write",
             "result": "Summary is 12 pages long but the 1-paragraph requirement cannot both be satisfied",
             "error": True, "parent": None, "subtask": "writing"},
            {"id": "step_2", "agent": "reviewer", "input": "Check summary against requirements",
             "thought": "Length requirement and detail requirement contradict each other",
             "action": "validate",
             "result": "FAILED: cannot satisfy contradictory constraints simultaneously",
             "error": False, "parent": "step_1", "subtask": "review"},
        ],
    },
    {
        "name": "spec_missing",
        "true_mode": "spec_missing",
        "true_cause": "step_1",
        "task_context": "Upload the computed sales figures to the shared dashboard",
        "steps": [
            {"id": "step_1", "agent": "publisher", "input": "Upload sales figures to dashboard",
             "thought": "No instruction defined whether overwriting existing data was allowed",
             "action": "upload",
             "result": "Uploaded new totals and silently replaced last month's published numbers",
             "error": True, "parent": None, "subtask": "publishing"},
            {"id": "step_2", "agent": "consumer", "input": "Read dashboard for monthly report",
             "thought": "Values changed unexpectedly — replaced last month's numbers",
             "action": "read",
             "result": "Tracker inconsistency detected", "error": False,
             "parent": "step_1", "subtask": "reporting"},
        ],
    },
    {
        "name": "coord_misaligned_goals",
        "true_mode": "coord_misaligned_goals",
        "true_cause": "step_2",
        "task_context": "Fetch and finalize the customer export for the billing cycle",
        "steps": [
            {"id": "step_1", "agent": "worker_a", "input": "Fetch customer records",
             "thought": "Goal is to get data out fast for the billing deadline", "action": "fetch",
             "result": "Streamed 100k records without filtering deleted rows",
             "error": False, "parent": None, "subtask": "fetching"},
            {"id": "step_2", "agent": "worker_b", "input": "Finalize export for billing accuracy",
             "thought": "Goal is to bill only active customers precisely",
             "action": "filter",
             "result": "Rejected worker_a's export because rows were unfiltered and attempted a full re-fetch",
             "error": True, "parent": "step_1", "subtask": "finalizing"},
        ],
    },
    {
        "name": "coord_resource_contention",
        "true_mode": "coord_resource_contention",
        "true_cause": "step_2",
        "task_context": "Generate embeddings and summaries for the document corpus",
        "steps": [
            {"id": "step_1", "agent": "embedder", "input": "Embed all documents",
             "thought": "Running the GPU embedding job", "action": "run_job",
             "result": "Job running, 40% GPU occupied", "error": False,
             "parent": None, "subtask": "embedding"},
            {"id": "step_2", "agent": "summarizer", "input": "Summarize all documents",
             "thought": "Running the LLM summarization job that needs the same GPU",
             "action": "run_job",
             "result": "OOM ERROR: GPU exhausted — both jobs allocated the same node",
             "error": True, "parent": "step_1", "subtask": "summarizing"},
        ],
    },
    {
        "name": "coord_deadlock",
        "true_mode": "coord_deadlock",
        "true_cause": "step_1",
        "task_context": "Have the plan approved before shipping",
        "steps": [
            {"id": "step_1", "agent": "planner", "input": "Get plan approval",
             "thought": "Will not release the plan until reviewer acknowledges receipt",
             "action": "wait",
             "result": "Blocked: waiting for reviewer acknowledgement", "error": True,
             "parent": None, "subtask": "planning"},
            {"id": "step_2", "agent": "reviewer", "input": "Review the plan",
             "thought": "Will not acknowledge until plan is released",
             "action": "wait",
             "result": "Blocked: waiting for plan delivery", "error": True,
             "parent": "step_1", "subtask": "review"},
            {"id": "step_3", "agent": "planner", "input": "Re-check plan status",
             "thought": "Cycle detected with reviewer — neither can proceed", "action": "poll",
             "result": "DEADLOCK: mutual wait persists (cycle planner↔reviewer)",
             "error": True, "parent": "step_2", "subtask": "planning"},
        ],
    },
    {
        "name": "verif_underconfident",
        "true_mode": "verif_underconfident",
        "true_cause": "step_1",
        "task_context": "Verify the calculated total before release",
        "steps": [
            {"id": "step_1", "agent": "checker", "input": "Verify total = sum of items",
             "thought": "Sampled the sum twice and got the same correct answer, but flagged it as 'likely wrong' since the wording differed between runs",
             "action": "verify",
             "result": "ESCALATED to human review, though the computed total matched ground truth exactly",
             "error": True, "parent": None, "subtask": "verification"},
            {"id": "step_2", "agent": "human", "input": "Review escalation",
             "thought": "Manual check: total is correct; no action needed", "action": "ack",
             "result": "FALSE ALARM: escalation was unnecessary",
             "error": False, "parent": "step_1", "subtask": "verification"},
        ],
    },
    {
        "name": "verif_wrong_criterion",
        "true_mode": "verif_wrong_criterion",
        "true_cause": "step_1",
        "task_context": "Check whether the reported revenue total is correct",
        "steps": [
            {"id": "step_1", "agent": "verifier", "input": "Check the reported revenue total",
             "thought": "I'll re-add the two line items and confirm the arithmetic checks out",
             "action": "verify",
             "result": "Arithmetic PASSES — but the numbers were stale from an old cache, so the total is still wrong",
             "error": True, "parent": None, "subtask": "verification"},
            {"id": "step_2", "agent": "controller", "input": "Assess verifier result",
             "thought": "Verifier checked formatting of the sum, not the freshness of the source data",
             "action": "assess",
             "result": "Root cause: wrong criterion — verified math, not data validity",
             "error": False, "parent": "step_1", "subtask": "assessment"},
        ],
    },
    {
        "name": "verif_self_inconsistent",
        "true_mode": "verif_self_inconsistent",
        "true_cause": "step_1",
        "task_context": "Report the ship cost for the order",
        "steps": [
            {"id": "step_1", "agent": "agent", "input": "Report the ship cost",
             "thought": "Produced the shipping cost figure", "action": "estimate",
             "result": "Cost: $120", "error": True, "parent": None, "subtask": "quoting"},
            {"id": "step_2", "agent": "agent", "input": "Report the ship cost again for the same order",
             "thought": "Produced a different figure for the identical question",
             "action": "estimate",
             "result": "Cost: $80 — contradicts the earlier $120 answer",
             "error": True, "parent": "step_1", "subtask": "quoting"},
        ],
    },
    {
        "name": "task_derailment",
        "true_mode": "task_derailment",
        "true_cause": "step_1",
        "task_context": "Compute country-by-country user counts",
        "steps": [
            {"id": "step_1", "agent": "analyst", "input": "Compute user count by country",
             "thought": "The requirement is actually 'unique monthly active users by region' — this is a different metric and a different grouping",
             "action": "compute",
             "result": "Counts by country computed correctly and thoroughly — but it complete the wrong task",
             "error": True, "parent": None, "subtask": "analysis"},
            {"id": "step_2", "agent": "reviewer", "input": "Review deliverable",
             "thought": "This answers the wrong question entirely", "action": "validate",
             "result": "REJECTED: complete deliverable for a different request",
             "error": False, "parent": "step_1", "subtask": "review"},
        ],
    },
    {
        "name": "infra_cascade_timeout",
        "true_mode": "infra_cascade_timeout",
        "true_cause": "step_1",
        "task_context": "Run the end-to-end release pipeline",
        "steps": [
            {"id": "step_1", "agent": "service_b", "input": "Receive request",
             "thought": "Waiting for service_c response", "action": "wait",
             "result": "TIMEOUT after 30s", "error": True, "parent": None, "subtask": "inter-call"},
            {"id": "step_2", "agent": "service_c", "input": "Respond to service_b",
             "thought": "Service_c has crashed silently — never responds",
             "action": "wait",
             "result": "TIMEOUT after 30s", "error": True, "parent": "step_1", "subtask": "inter-call"},
            {"id": "step_3", "agent": "service_a", "input": "Propagate to downstream",
             "thought": "Receiving cascading timeouts from both dependents", "action": "propagate",
             "result": "TIMEOUT after 30s — cascade triggered", "error": True,
             "parent": "step_2", "subtask": "inter-call"},
        ],
    },
    {
        "name": "infra_unknown",
        "true_mode": "infra_unknown",
        "true_cause": "step_1",
        "task_context": "Query the external catalog during a routine job",
        "steps": [
            {"id": "step_1", "agent": "worker", "input": "Call external catalog",
             "thought": "Opening connection to upstream", "action": "http_request",
             "result": "ERROR: transport connection reset — no semantic or agentic cause (DNS/van transport)",
             "error": True, "parent": None, "subtask": "external_call"},
            {"id": "step_2", "agent": "worker", "input": "Retry after backoff",
             "thought": "Retrying the same external call", "action": "http_request",
             "result": "ERROR: connection reset again — no fallback path", "error": True,
             "parent": "step_1", "subtask": "external_call"},
        ],
    },
    # --- Scenarios where true_cause is deliberately NOT the first error node ---
    # Early transient failure that self-corrects; the actual root cause comes later.
    {
        "name": "transient_recovered_then_hallucination",
        "true_mode": "task_hallucination",
        "true_cause": "step_3",
        "task_context": "Load the sales report and summarize revenue",
        "steps": [
            {"id": "step_1", "agent": "reader", "input": "Load report chunk 1/3",
             "thought": "Streaming chunks from the source", "action": "load",
             "result": "ERROR: read of chunk 1 timed out (transient network drop)", "error": True,
             "parent": None, "subtask": "loading"},
            {"id": "step_2", "agent": "reader", "input": "Retry chunk load",
             "thought": "Retrying — connection back up", "action": "load",
             "result": "loaded: full document retrieved", "error": False,
             "parent": "step_1", "subtask": "loading"},
            {"id": "step_3", "agent": "analyst", "input": "Summarize revenue from the document",
             "thought": "I remember from a similar report, revenue was 12.8M", "action": "summarize",
             "result": "Revenue was $12.8M (figure fabricated, not from the loaded document)", "error": True,
             "parent": "step_2", "subtask": "analysis"},
            {"id": "step_4", "agent": "reviewer", "input": "Check summary against document",
             "thought": "The loaded data says $9.4M — the figure does not match", "action": "validate",
             "result": "REJECTED: hallucinated revenue figure", "error": False,
             "parent": "step_3", "subtask": "review"},
        ],
    },
    {
        "name": "transient_recovered_then_wrong_criterion",
        "true_mode": "verif_wrong_criterion",
        "true_cause": "step_3",
        "task_context": "Confirm the ledger total is correct before release",
        "steps": [
            {"id": "step_1", "agent": "checker", "input": "Pull ledger entries",
             "thought": "Ledger API flaky right now", "action": "fetch",
             "result": "ERROR: ledger fetch timed out (transient)", "error": True,
             "parent": None, "subtask": "ledger_fetch"},
            {"id": "step_2", "agent": "checker", "input": "Retry ledger fetch",
             "thought": "Retrying; ledger reachable now", "action": "fetch",
             "result": "fetched: full ledger entry set", "error": False,
             "parent": "step_1", "subtask": "ledger_fetch"},
            {"id": "step_3", "agent": "verifier", "input": "Verify total = sum of entries",
             "thought": "I checked the *formatting* of the total, not the freshness of the source entries",
             "action": "verify",
             "result": "Total looks formatted correctly, but it is stale — verified the wrong criterion", "error": True,
             "parent": "step_2", "subtask": "verification"},
            {"id": "step_4", "agent": "controller", "input": "Assess verifier result",
             "thought": "Data was stale, verifier missed it", "action": "assess",
             "result": "Root cause: checked formatting, not data validity", "error": False,
             "parent": "step_3", "subtask": "assessment"},
        ],
    },
    {
        "name": "transient_recovered_then_derailment",
        "true_mode": "task_derailment",
        "true_cause": "step_3",
        "task_context": "Compute unique monthly active users per region",
        "steps": [
            {"id": "step_1", "agent": "worker", "input": "Access the analytics corpus",
             "thought": "The corpus API is momentarily rate-limited", "action": "open",
             "result": "ERROR: rate limit reached on the analytics endpoint (transient)", "error": True,
             "parent": None, "subtask": "access"},
            {"id": "step_2", "agent": "worker", "input": "Retry access",
             "thought": "Backoff finished, retrying", "action": "open",
             "result": "OK: corpus accessible", "error": False,
             "parent": "step_1", "subtask": "access"},
            {"id": "step_3", "agent": "analyst", "input": "Compute per-region MAU",
             "thought": "Actually I derived total session counts, not unique users by region",
             "action": "compute",
             "result": "Computed session totals — engineered the wrong deliverable entirely", "error": True,
             "parent": "step_2", "subtask": "analysis"},
            {"id": "step_4", "agent": "reviewer", "input": "Review deliverable",
             "thought": "This is the wrong task output", "action": "validate",
             "result": "REJECTED: deliverable is for a different request", "error": False,
             "parent": "step_3", "subtask": "review"},
        ],
    },
    {
        "name": "transient_recovered_then_deadlock",
        "true_mode": "coord_deadlock",
        "true_cause": "step_3",
        "task_context": "Get the plan approved and shipped",
        "steps": [
            {"id": "step_1", "agent": "scheduler", "input": "Signal plan readiness",
             "thought": "trying to reach requester; short network flap", "action": "ping",
             "result": "TIMEOUT signalling plan readiness (transient network blip)", "error": True,
             "parent": None, "subtask": "signalling"},
            {"id": "step_2", "agent": "scheduler", "input": "Re-signal plan readiness",
             "thought": "signal re-sent fine", "action": "ping",
             "result": "plan readiness signal acknowledged", "error": False,
             "parent": "step_1", "subtask": "signalling"},
            {"id": "step_3", "agent": "planner", "input": "Wait for reviewer ack",
             "thought": "Will not release until reviewer acknowledges, but reviewer will not ack until release",
             "action": "wait",
             "result": "BLOCKED: mutual wait with reviewer — deadlock cycle", "error": True,
             "parent": "step_2", "subtask": "planning"},
            {"id": "step_4", "agent": "reviewer", "input": "Review the plan",
             "thought": "Cannot ack until plan is delivered", "action": "wait",
             "result": "BLOCKED: awaiting plan delivery — part of the deadlock", "error": True,
             "parent": "step_3", "subtask": "review"},
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
    settings = Settings()
    if api_key:
        os.environ["AGENT_REFLEX_LLM_API_KEY"] = api_key
    elif not resolve_api_key(settings):
        print("ERROR: No LLM API key found. Set DEEPSEEK_API_KEY or OPENAI_API_KEY env var.")
        print("Running in structure-only mode (no LLM calls).")
        return {"error": "no_api_key", "note": "Set DEEPSEEK_API_KEY or OPENAI_API_KEY to run actual eval"}

    engine = AttributionEngine()
    results = _evaluate_scenarios(engine)

    mode_correct = sum(1 for r in results if r["mode_correct"])
    step_correct = sum(1 for r in results if r["step_correct"])
    total = len(results)

    return {
        "total_scenarios": total,
        "mode_accuracy_pct": round(mode_correct / total * 100, 1),
        "step_accuracy_pct": round(step_correct / total * 100, 1),
        "mode_correct": mode_correct,
        "step_correct": step_correct,
        "details": results,
    }


def _evaluate_scenarios(engine: AttributionEngine) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
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

        print(f"{'✓' if is_mode_correct else '✗'} mode, {'✓' if is_step_correct else '✗'} step")
    return results


def run_eval_iterated(api_key: str | None = None, n_runs: int = 3) -> dict[str, Any]:
    """Run the eval n_runs times and report mean/std accuracy across runs."""
    settings = Settings()
    if api_key:
        os.environ["AGENT_REFLEX_LLM_API_KEY"] = api_key
    elif not resolve_api_key(settings):
        return {"error": "no_api_key", "note": "Set DEEPSEEK_API_KEY or OPENAI_API_KEY to run actual eval"}

    engine = AttributionEngine()
    total = len(SYNTHETIC_SCENARIOS)

    run_summaries: list[dict[str, Any]] = []
    mode_accs: list[float] = []
    step_accs: list[float] = []
    last_details: list[dict[str, Any]] = []

    for run in range(n_runs):
        print(f"\n  --- Run {run + 1}/{n_runs} ---")
        details = _evaluate_scenarios(engine)
        mode_accs.append(sum(1 for r in details if r["mode_correct"]) / total * 100)
        step_accs.append(sum(1 for r in details if r["step_correct"]) / total * 100)
        run_summaries.append({
            "run": run + 1,
            "mode_accuracy_pct": round(mode_accs[-1], 1),
            "step_accuracy_pct": round(step_accs[-1], 1),
        })
        last_details = details

    return {
        "n_runs": n_runs,
        "total_scenarios": total,
        "mode_accuracy_mean_pct": round(sum(mode_accs) / len(mode_accs), 1),
        "mode_accuracy_std_pct": round(float(np.std(mode_accs)), 1),
        "step_accuracy_mean_pct": round(sum(step_accs) / len(step_accs), 1),
        "step_accuracy_std_pct": round(float(np.std(step_accs)), 1),
        "runs": run_summaries,
        "details": last_details,
    }


def run_comparison(api_key: str | None = None) -> dict[str, Any]:
    """Run both the real engine and the naive baseline over the eval set.

    Reports step/mode accuracy for the real oracle-guided engine and for
    NaiveEarliestErrorBaseline side by side on the same scenarios.
    Mode accuracy is reported from the same classifier for both (the
    baseline has no classifier of its own), but the step accuracy is the
    meaningful differentiator.
    """
    from .baseline import NaiveEarliestErrorBaseline

    settings = Settings()
    if api_key:
        os.environ["AGENT_REFLEX_LLM_API_KEY"] = api_key
    elif not resolve_api_key(settings):
        return {"error": "no_api_key", "note": "Set DEEPSEEK_API_KEY or OPENAI_API_KEY to run comparison"}

    engine = AttributionEngine()
    baseline = NaiveEarliestErrorBaseline()

    oracle_details: list[dict[str, Any]] = []
    baseline_details: list[dict[str, Any]] = []
    total = len(SYNTHETIC_SCENARIOS)

    for scenario in SYNTHETIC_SCENARIOS:
        graph = build_graph_from_scenario(scenario)
        result = engine.attribute(
            session_id=f"compare_{scenario['name']}",
            graph=graph,
            task_context=scenario["task_context"],
        )
        is_mode_correct = result.failure_type.value == scenario["true_mode"]
        oracle_details.append({
            "scenario": scenario["name"],
            "mode_correct": is_mode_correct,
            "step_correct": result.cause_node_id == scenario["true_cause"],
            "predicted_cause": result.cause_node_id,
            "crs": round(result.causal_responsibility_score, 2),
        })

        cause_node = baseline.attribute(graph)
        baseline_details.append({
            "scenario": scenario["name"],
            "mode_correct": is_mode_correct,
            "step_correct": (cause_node.node_id if cause_node else None) == scenario["true_cause"],
            "predicted_cause": cause_node.node_id if cause_node else None,
            "crs": 0.0,
        })

    oracle_mode = sum(1 for r in oracle_details if r["mode_correct"])
    oracle_step = sum(1 for r in oracle_details if r["step_correct"])
    baseline_step = sum(1 for r in baseline_details if r["step_correct"])

    return {
        "total_scenarios": total,
        "oracle_method": {
            "mode_accuracy_pct": round(oracle_mode / total * 100, 1),
            "step_accuracy_pct": round(oracle_step / total * 100, 1),
            "details": oracle_details,
        },
        "naive_baseline": {
            "mode_accuracy_pct": round(oracle_mode / total * 100, 1),
            "step_accuracy_pct": round(baseline_step / total * 100, 1),
            "details": baseline_details,
        },
    }


def save_results_json(data: dict[str, Any], path: str | None = None) -> str:
    """Write an eval result dict to eval_results/ as structured JSON."""
    import json as _json

    target = path or os.path.join(
        os.path.dirname(__file__), "..", "..", "eval_results",
        f"synthetic_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    target = os.path.abspath(target)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w") as f:
        _json.dump(data, f, indent=2, default=str)
    return target


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
    import argparse

    parser = argparse.ArgumentParser(description="AgentReflex attribution eval")
    parser.add_argument("--runs", type=int, default=1, help="Number of eval runs to aggregate (default 1)")
    parser.add_argument("--save", action="store_true", help="write structured JSON result to eval_results/")
    parser.add_argument("--compare", action="store_true", help="compare real engine vs naive baseline")
    args, _ = parser.parse_known_args()

    print("=" * 60)
    print("AgentReflex Eval — Attribution Accuracy Report")
    print("=" * 60)
    print()
    print(f"Test set: {len(SYNTHETIC_SCENARIOS)} synthetic scenarios")
    print()

    if args.compare:
        results = run_comparison()
        if "error" in results:
            print(f"\n{results['error']}: {results['note']}")
            return
        oracle = results["oracle_method"]
        baseline = results["naive_baseline"]
        print("\nComparison: real oracle-guided engine vs naive earliest-error baseline")
        print(f"{'Method':32s} {'Mode Acc':>10s} {'Step Acc':>10s}")
        print("-" * 54)
        print(f"{'oracle_method (real)':32s} {oracle['mode_accuracy_pct']:>9.1f}% {oracle['step_accuracy_pct']:>9.1f}%")
        print(f"{'naive_baseline':32s} {baseline['mode_accuracy_pct']:>9.1f}% {baseline['step_accuracy_pct']:>9.1f}%")
        print("\nPer-scenario step accuracy:")
        print(f"{'Scenario':34s} {'Oracle':>8s} {'Naive':>8s}")
        print("-" * 52)
        for od, bd in zip(results["oracle_method"]["details"], results["naive_baseline"]["details"]):
            print(f"{od['scenario']:34s} {'✓' if od['step_correct'] else '✗':>8s} {'✓' if bd['step_correct'] else '✗':>8s}")
        if args.save:
            path = save_results_json(
                results,
                path=os.path.join(
                    os.path.dirname(__file__), "..", "..", "eval_results",
                    f"comparison_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                ),
            )
            print(f"\nComparison saved to: {path}")
        return

    if args.runs > 1:
        results = run_eval_iterated(n_runs=args.runs)
    else:
        results = run_eval()
    if "error" in results:
        print(f"\n{results['error']}: {results['note']}")
        return

    if args.runs > 1:
        print("\nResults (across runs):")
        print(f"  Mode-level accuracy:  {results['mode_accuracy_mean_pct']:.1f}% ± {results['mode_accuracy_std_pct']:.1f}")
        print(f"  Step-level accuracy:  {results['step_accuracy_mean_pct']:.1f}% ± {results['step_accuracy_std_pct']:.1f}")
        print("  Per-run:")
        for run in results["runs"]:
            print(f"    Run {run['run']}: mode {run['mode_accuracy_pct']:.1f}%  step {run['step_accuracy_pct']:.1f}%")
    else:
        print("\nResults:")
        print(f"  Mode-level accuracy:  {results['mode_accuracy_pct']:.1f}% ({results['mode_correct']}/{results['total_scenarios']})")
        print(f"  Step-level accuracy:  {results['step_accuracy_pct']:.1f}% ({results['step_correct']}/{results['total_scenarios']})")
    print()

    print("Per-scenario breakdown (last run):")
    print(f"{'Scenario':32s} {'True Mode':28s} {'Predicted':28s} {'CRS':6s} {'Mode':6s} {'Step':6s}")
    print("-" * 112)
    for d in results["details"]:
        m = "✓" if d["mode_correct"] else "✗"
        s = "✓" if d["step_correct"] else "✗"
        print(f"{d['scenario']:32s} {d['true_mode']:28s} {d['predicted_mode']:28s} {d['crs']:<6.2f} {m:6s} {s:6s}")

    print_confusion_matrix(results["details"])

    if args.save:
        path = save_results_json(results)
        print(f"\nStructured results saved to: {path}")

    print()
    print("=" * 60)
    print("Note: These are synthetic scenarios designed to test specific")
    print("failure modes. Real-world Who&When benchmark results will differ.")
    print("=" * 60)


if __name__ == "__main__":
    main()
