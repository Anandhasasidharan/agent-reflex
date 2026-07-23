"""
Redaction-vs-attribution-accuracy ablation.

Runs the attribution engine over the synthetic eval set twice:
- Once with redaction ON  (span content redacted)
- Once with redaction OFF (raw span content)

Reports mode-level and step-level accuracy delta.
"""

from __future__ import annotations

import copy
import os
from typing import Any

from agent_reflex.attribution.engine import AttributionEngine
from agent_reflex.common.config import Settings

from .runner import SYNTHETIC_SCENARIOS, build_graph_from_scenario


def run_ablation(api_key: str | None = None) -> dict[str, Any]:
    settings = Settings()

    resolved_key = api_key or settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    if not resolved_key:
        return {"error": "no_api_key", "note": "Set OPENAI_API_KEY to run ablation"}

    settings_redacted = copy.deepcopy(settings)
    settings_redacted.redaction_enabled = True
    settings_redacted.openai_api_key = resolved_key
    engine_redacted = AttributionEngine(settings=settings_redacted)

    settings_plain = copy.deepcopy(settings)
    settings_plain.redaction_enabled = False
    settings_plain.openai_api_key = resolved_key
    engine_plain = AttributionEngine(settings=settings_plain)

    plain_results = _run_suite(engine_plain, "plain (no redaction)")
    redacted_results = _run_suite(engine_redacted, "redacted")

    return {
        "plain": plain_results,
        "redacted": redacted_results,
        "delta_mode": round(plain_results["mode_accuracy_pct"] - redacted_results["mode_accuracy_pct"], 1),
        "delta_step": round(plain_results["step_accuracy_pct"] - redacted_results["step_accuracy_pct"], 1),
    }


def _run_suite(engine: AttributionEngine, label: str) -> dict[str, Any]:
    mode_correct = 0
    step_correct = 0
    total = len(SYNTHETIC_SCENARIOS)

    for scenario in SYNTHETIC_SCENARIOS:
        graph = build_graph_from_scenario(scenario)
        result = engine.attribute(
            session_id=f"ablation_{scenario['name']}",
            graph=graph,
            task_context=scenario["task_context"],
        )
        if result.failure_type.value == scenario["true_mode"]:
            mode_correct += 1
        if result.cause_node_id == scenario["true_cause"]:
            step_correct += 1

    return {
        "label": label,
        "total_scenarios": total,
        "mode_accuracy_pct": round(mode_correct / total * 100, 1),
        "step_accuracy_pct": round(step_correct / total * 100, 1),
        "mode_correct": mode_correct,
        "step_correct": step_correct,
    }


def main() -> None:
    print("=" * 60)
    print("AgentReflex — Redaction Ablation Report")
    print("=" * 60)

    result = run_ablation()

    if "error" in result:
        print(f"\n{result['error']}: {result.get('note', '')}")
        return

    print()
    print(f"{'Config':30s} {'Mode Acc':>10s} {'Step Acc':>10s} {'Mode#':>6s} {'Step#':>6s}")
    print("-" * 62)
    for key in ("plain", "redacted"):
        r = result[key]
        print(f"{r['label']:30s} {r['mode_accuracy_pct']:>9.1f}% {r['step_accuracy_pct']:>9.1f}% "
              f"{r['mode_correct']:>3d}/{r['total_scenarios']:<2d} {r['step_correct']:>3d}/{r['total_scenarios']:<2d}")
    print("-" * 62)
    print(f"{'Delta (plain - redacted)':30s} {result['delta_mode']:>+9.1f}% {result['delta_step']:>+9.1f}%")
    print()
    print("Note: Redaction ablates span-level sensitive content (API keys,")
    print("secrets, tokens). A positive delta means redaction reduces accuracy.")
    print("=" * 60)


if __name__ == "__main__":
    main()
