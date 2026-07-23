from __future__ import annotations

import statistics
from typing import Any


class PredictiveTopologyScorer:
    def __init__(self) -> None:
        self._mode_weights: dict[str, float] = {
            "spec_ambiguous": 0.08,
            "spec_incomplete": 0.07,
            "spec_contradictory": 0.05,
            "spec_missing": 0.04,
            "coord_misaligned_goals": 0.10,
            "coord_misaligned_assumptions": 0.12,
            "coord_resource_contention": 0.06,
            "coord_deadlock": 0.04,
            "verif_overconfident": 0.08,
            "verif_underconfident": 0.03,
            "verif_wrong_criterion": 0.04,
            "verif_self_inconsistent": 0.05,
            "task_derailment": 0.10,
            "task_hallucination": 0.06,
            "infra_rate_limit": 0.03,
            "infra_context_window": 0.03,
            "infra_cascade_timeout": 0.02,
        }

    def score_architecture(self, topology: dict[str, Any]) -> dict[str, float]:
        fan_out = self._calc_fan_out(topology)
        depth = self._calc_depth(topology)
        topology_type = self._classify_topology(topology)
        n_agents = len(topology.get("agents", []))
        n_tools = sum(len(a.get("tools", [])) for a in topology.get("agents", []))
        risk_scores: dict[str, float] = {}
        for mode_str, base_weight in self._mode_weights.items():
            multiplier = 1.0
            if "coord_" in mode_str:
                multiplier *= 1.0 + 0.15 * max(0, fan_out - 2)
                multiplier *= 1.0 + 0.10 * max(0, depth - 3)

            if "infra_" in mode_str:
                multiplier *= 1.0 + 0.05 * n_tools

            if "task_" in mode_str:
                multiplier *= 1.0 + 0.10 * n_agents

            if topology_type == "bag_of_agents":
                multiplier *= 1.25
            elif topology_type == "planner_worker":
                multiplier *= 0.85

            risk_scores[mode_str] = min(1.0, base_weight * multiplier)

        return dict(sorted(risk_scores.items(), key=lambda x: -x[1]))

    def _calc_fan_out(self, topology: dict[str, Any]) -> float:
        edges = topology.get("edges", [])
        out_counts: dict[str, int] = {}
        for edge in edges:
            src = edge.get("source", "")
            out_counts[src] = out_counts.get(src, 0) + 1
        return statistics.mean(out_counts.values()) if out_counts else 1.0

    def _calc_depth(self, topology: dict[str, Any]) -> int:
        edges = topology.get("edges", [])
        if not edges:
            return 1
        children: dict[str, list[str]] = {}
        all_nodes: set[str] = set()
        for edge in edges:
            src = edge.get("source", "")
            dst = edge.get("target", "")
            children.setdefault(src, []).append(dst)
            all_nodes.add(src)
            all_nodes.add(dst)

        def max_depth(node: str, visited: set[str]) -> int:
            if node in visited:
                return 0
            visited.add(node)
            max_d = 1
            for child in children.get(node, []):
                max_d = max(max_d, 1 + max_depth(child, visited))
            visited.remove(node)
            return max_d

        roots = all_nodes - {e.get("target", "") for e in edges}
        return max(max_depth(r, set()) for r in roots) if roots else 1

    def _classify_topology(self, topology: dict[str, Any]) -> str:
        agents = topology.get("agents", [])
        edges = topology.get("edges", [])
        n_agents = len(agents)

        if n_agents <= 1:
            return "single_agent"

        source_nodes: set[str] = {e.get("source", "") for e in edges}
        target_nodes: set[str] = {e.get("target", "") for e in edges}

        planner_nodes = source_nodes - target_nodes

        if len(planner_nodes) == 1 and n_agents > 1:
            return "planner_worker"

        if not edges or len(edges) >= n_agents * (n_agents - 1) * 0.5:
            return "bag_of_agents"

        return "pipeline"
