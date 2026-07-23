from __future__ import annotations

import json
import uuid
from typing import Any

import networkx as nx

from agent_reflex.common.types import CausalGraphEdge, CausalGraphNode, StepOTAR


class OTARParser:
    @staticmethod
    def parse(span_attributes: dict[str, Any]) -> StepOTAR:
        return StepOTAR(
            observation=span_attributes.get("input", ""),
            thought=span_attributes.get("agent.thought", ""),
            action=span_attributes.get("agent.action", span_attributes.get("agent.action.type", "unknown")),
            result=span_attributes.get("output", span_attributes.get("agent.artifact", "")),
        )

    @staticmethod
    def from_span_events(events: list[dict[str, Any]]) -> StepOTAR:
        parts = {"observation": "", "thought": "", "action": "", "result": ""}
        for event in events:
            name = event.get("name", "")
            attrs = event.get("attributes", {})
            if name == "gen_ai.completion":
                parts["result"] = attrs.get("content", "")
            elif name == "gen_ai.prompt":
                parts["observation"] = attrs.get("content", "")
        return StepOTAR(**parts)


class CausalGraph:
    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: dict[str, CausalGraphNode] = {}

    def add_step(self, node: CausalGraphNode) -> str:
        node_id = node.node_id or str(uuid.uuid4())
        self._nodes[node_id] = node
        self._graph.add_node(
            node_id,
            agent_id=node.agent_id,
            step_index=node.step_index,
            otar=node.otar,
            parent_id=node.parent_id,
            subtask_id=node.subtask_id,
            execution_time_ms=node.execution_time_ms,
            error_flag=node.error_flag,
        )
        if node.parent_id and node.parent_id in self._nodes:
            self._graph.add_edge(node.parent_id, node_id, edge_type="control_flow")
        return node_id

    def add_dependency(self, source_id: str, target_id: str) -> None:
        self._graph.add_edge(source_id, target_id, edge_type="data_dependency")

    def get_node(self, node_id: str) -> CausalGraphNode | None:
        return self._nodes.get(node_id)

    def get_children(self, node_id: str) -> list[CausalGraphNode]:
        return [self._nodes[n] for n in self._graph.successors(node_id) if n in self._nodes]

    def get_parent(self, node_id: str) -> CausalGraphNode | None:
        parents = list(self._graph.predecessors(node_id))
        if parents:
            return self._nodes.get(parents[0])
        return None

    def get_subtask_nodes(self, subtask_id: str) -> list[CausalGraphNode]:
        return [n for n in self._nodes.values() if n.subtask_id == subtask_id]

    def get_all_nodes(self) -> list[CausalGraphNode]:
        return list(self._nodes.values())

    def get_edges(self) -> list[CausalGraphEdge]:
        edges = []
        for u, v, data in self._graph.edges(data=True):
            edges.append(CausalGraphEdge(
                source_id=u,
                target_id=v,
                edge_type=data.get("edge_type", "control_flow"),
            ))
        return edges

    def infer_data_dependencies(self) -> None:
        nodes_sorted = sorted(self._nodes.values(), key=lambda n: n.step_index)
        for i, node in enumerate(nodes_sorted):
            for j in range(i):
                prev = nodes_sorted[j]
                if self._output_feeds_input(prev, node):
                    self.add_dependency(prev.node_id, node.node_id)

    @staticmethod
    def _output_feeds_input(source: CausalGraphNode, target: CausalGraphNode) -> bool:
        src_result = source.otar.result.lower() if source.otar.result else ""
        tgt_input = target.otar.observation.lower() if target.otar.observation else ""
        return bool(src_result and tgt_input and (
            any(word in tgt_input for word in src_result.split()[:5])
            or any(word in src_result for word in tgt_input.split()[:5])
        ))

    def decompose_into_subtasks(self) -> dict[str, list[CausalGraphNode]]:
        subtasks: dict[str, list[CausalGraphNode]] = {}
        for node in self._nodes.values():
            sid = node.subtask_id or "default"
            if sid not in subtasks:
                subtasks[sid] = []
            subtasks[sid].append(node)
        return subtasks

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "agent_id": n.agent_id,
                    "step_index": n.step_index,
                    "otar": {
                        "observation": n.otar.observation,
                        "thought": n.otar.thought,
                        "action": n.otar.action,
                        "result": n.otar.result,
                    },
                    "parent_id": n.parent_id,
                    "subtask_id": n.subtask_id,
                    "execution_time_ms": n.execution_time_ms,
                    "error_flag": n.error_flag,
                }
                for n in self._nodes.values()
            ],
            "edges": [
                {"source_id": e.source_id, "target_id": e.target_id, "edge_type": e.edge_type}
                for e in self.get_edges()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CausalGraph:
        cg = cls()
        for nd in data.get("nodes", []):
            otar = StepOTAR(**nd["otar"])
            node = CausalGraphNode(
                node_id=nd["node_id"],
                agent_id=nd["agent_id"],
                step_index=nd["step_index"],
                otar=otar,
                parent_id=nd["parent_id"],
                subtask_id=nd["subtask_id"],
                execution_time_ms=nd["execution_time_ms"],
                error_flag=nd["error_flag"],
            )
            cg._nodes[node.node_id] = node
            cg._graph.add_node(node.node_id, **nd)
        for ed in data.get("edges", []):
            cg._graph.add_edge(ed["source_id"], ed["target_id"], edge_type=ed["edge_type"])
        return cg

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> CausalGraph:
        return cls.from_dict(json.loads(raw))
