from __future__ import annotations

from typing import Any

from agent_reflex.graph.models import CausalGraph


def build_viewer_data(graph: CausalGraph, cause_node_id: str | None = None) -> dict[str, Any]:
    nodes_data = []
    for node in graph.get_all_nodes():
        nodes_data.append({
            "id": node.node_id,
            "label": f"{node.agent_id}:{node.step_index}",
            "agent": node.agent_id,
            "step_index": node.step_index,
            "action": node.otar.action,
            "is_root_cause": node.node_id == cause_node_id,
            "error": node.error_flag,
            "subtask": node.subtask_id,
            "parent_id": node.parent_id,
            "execution_time_ms": node.execution_time_ms,
            "otar": {
                "observation": node.otar.observation,
                "thought": node.otar.thought,
                "action": node.otar.action,
                "result": node.otar.result,
            },
        })

    edges_data = [
        {"source": e.source_id, "target": e.target_id, "type": e.edge_type}
        for e in graph.get_edges()
    ]

    return {"nodes": nodes_data, "edges": edges_data}

