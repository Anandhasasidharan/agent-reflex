"""
Cross-benchmark attribution evaluation.

Runs the attribution engine against both Who&When and TraceElephant
(if their datasets are available), then prints a side-by-side comparison.
"""

from __future__ import annotations

import os
from typing import Any

from agent_reflex.attribution.engine import AttributionEngine
from agent_reflex.common.config import Settings

from .traceelephant import run_traceelephant
from .whowhen import run_whowhen


def run_cross_benchmark(api_key: str | None = None) -> dict[str, Any]:
    settings = Settings()
    resolved_key = api_key or settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    if not resolved_key:
        return {"error": "no_api_key", "note": "Set OPENAI_API_KEY to run cross-benchmark eval"}

    settings.openai_api_key = resolved_key
    engine = AttributionEngine(settings=settings)

    whowhen_result = run_whowhen(engine)
    telephant_result = run_traceelephant(engine)

    return {
        "whowhen": whowhen_result,
        "traceelephant": telephant_result,
        "both_available": "error" not in whowhen_result and "error" not in telephant_result,
    }


def main() -> None:
    print("=" * 60)
    print("AgentReflex — Cross-Benchmark Attribution Report")
    print("=" * 60)

    result = run_cross_benchmark()
    if "error" in result:
        print(f"\n{result['error']}: {result.get('note', '')}")
        return

    print()
    for key, label in [("whowhen", "Who&When"), ("traceelephant", "TraceElephant")]:
        r = result[key]
        if "error" in r:
            print(f"  {label}: {r['note']}")
            continue
        print(f"  {label}:")
        print(f"    Mode accuracy:  {r['mode_accuracy_pct']:.1f}% ({r['mode_correct']}/{r['total']})")
        print(f"    Step accuracy:  {r['step_accuracy_pct']:.1f}% ({r['step_correct']}/{r['total']})")
        print()

    if result["both_available"]:
        w = result["whowhen"]
        t = result["traceelephant"]
        print("  Comparison:")
        print(f"    Mode delta (Who&When - TraceElephant):  {w['mode_accuracy_pct'] - t['mode_accuracy_pct']:+.1f}%")
        print(f"    Step delta (Who&When - TraceElephant):  {w['step_accuracy_pct'] - t['step_accuracy_pct']:+.1f}%")
        print()
        print("  Note: Who&When is output-only/partial-observability; TraceElephant")
        print("  provides fully observable execution environments. Differences in")
        print("  accuracy are expected and reflect the observability gap.")

    print("=" * 60)


if __name__ == "__main__":
    main()
