"""Naive baseline for root-cause step attribution.

Exists only to be *compared against* the real attribution engine. It makes
no LLM call and performs no causal reasoning: the root cause is simply the
earliest error-flagged node by step_index. Do not use this inside
AttributionEngine's production path — it is a measurement baseline.
"""

from __future__ import annotations

from agent_reflex.common.types import CausalGraphNode
from agent_reflex.graph.models import CausalGraph


class NaiveEarliestErrorBaseline:
    """Naive baseline: root cause = earliest error-flagged node by
    step_index.

    No LLM call, no causal reasoning. Exists to be compared against the
    real attribution engine, not to replace it.
    """

    def attribute(self, graph: CausalGraph) -> CausalGraphNode | None:
        error_nodes = sorted(
            (n for n in graph.get_all_nodes() if n.error_flag),
            key=lambda n: n.step_index,
        )
        return error_nodes[0] if error_nodes else None
