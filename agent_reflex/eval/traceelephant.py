"""
TraceElephant benchmark runner.

Loads TraceElephant-format traces, runs attribution, reports accuracy.
Expected data: data/traceelephant/ directory with trace JSON files,
or a single traces.json with the TraceElephant schema.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any

from agent_reflex.attribution.engine import AttributionEngine

from .benchmark_adapter import run_benchmark

TRACE_ELEPHANT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "traceelephant"
)


def load_traceelephant_traces(path: str = TRACE_ELEPHANT_DATA_DIR) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []

    combined = os.path.join(path, "traces.json")
    if os.path.exists(combined):
        with open(combined) as f:
            data = json.load(f)
        if isinstance(data, list):
            traces.extend(data)
        elif isinstance(data, dict) and "traces" in data:
            traces.extend(data["traces"])

    for fpath in sorted(glob.glob(os.path.join(path, "*.json"))):
        if fpath == combined:
            continue
        with open(fpath) as f:
            data = json.load(f)
        if isinstance(data, list):
            traces.extend(data)
        elif isinstance(data, dict):
            if "steps" in data or "trace_id" in data:
                traces.append(data)

    return traces


def run_traceelephant(engine: AttributionEngine) -> dict[str, Any]:
    traces = load_traceelephant_traces()
    if not traces:
        return {
            "error": "data_not_found",
            "note": (
                "TraceElephant dataset not found. To run: clone "
                "github.com/TraceElephant/TraceElephant and copy trace files to:\n"
                f"  {TRACE_ELEPHANT_DATA_DIR}"
            ),
        }
    return run_benchmark(traces, engine, label="traceelephant")
