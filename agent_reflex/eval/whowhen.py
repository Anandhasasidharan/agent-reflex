"""
Who&When benchmark runner.

Loads Who&When-format traces, runs attribution, reports accuracy.
Expected data path: data/whowhen/traces.json
"""

from __future__ import annotations

import json
import os
from typing import Any

from agent_reflex.attribution.engine import AttributionEngine

from .benchmark_adapter import run_benchmark

WHO_WHEN_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "whowhen", "traces.json"
)


def load_whowhen_traces(path: str = WHO_WHEN_DATA_PATH) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "traces" in data:
        return data["traces"]
    return []


def run_whowhen(engine: AttributionEngine) -> dict[str, Any]:
    traces = load_whowhen_traces()
    if not traces:
        return {
            "error": "data_not_found",
            "note": (
                "Who&When dataset not found. To run: download traces.json from "
                "the Who&When benchmark repository and place at:\n"
                f"  {WHO_WHEN_DATA_PATH}"
            ),
        }
    return run_benchmark(traces, engine, label="whowhen")
