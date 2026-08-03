from __future__ import annotations

import json
import uuid
from typing import Any

import networkx as nx

from agent_reflex.common.types import CausalGraphEdge, CausalGraphNode, StepOTAR


class OTARParser:
    """Parse OTAR (Observation-Thought-Action-Result) fields from OTel spans.

    Follows the canonical schema in docs/otel_ingestion_schema.md: standard
    GenAI semconv attributes (gen_ai.*) plus the agent_reflex.* extension
    namespace. Accepts both the OTel SDK's ReadableSpan.to_json() attribute
    dict format and the OTLP/HTTP JSON wire format (list of {key, value}).
    """

    @staticmethod
    def _normalize_attributes(attrs: Any) -> dict[str, Any]:
        """Accept either an SDK dict or an OTLP list of {key, value} pairs."""
        if isinstance(attrs, dict):
            return attrs
        if isinstance(attrs, list):
            normalized: dict[str, Any] = {}
            for item in attrs:
                if isinstance(item, dict) and "key" in item:
                    value = item.get("value", {})
                    if isinstance(value, dict):
                        # OTLP AnyValue: {"stringValue": ...} | {"intValue": ...}
                        normalized[item["key"]] = (
                            value.get("stringValue")
                            or value.get("intValue")
                            or value.get("doubleValue")
                            or value.get("boolValue")
                            or ""
                        )
                    else:
                        normalized[item["key"]] = value
            return normalized
        return {}

    @staticmethod
    def _messages_content(raw: Any) -> str:
        """Extract concatenated content from a gen_ai.*.messages JSON array."""
        if not raw:
            return ""
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return raw
            if isinstance(parsed, list):
                parts = []
                for message in parsed:
                    if isinstance(message, dict):
                        content = message.get("content", "")
                        if isinstance(content, list):
                            content = " ".join(
                                str(c.get("text", "")) if isinstance(c, dict) else str(c)
                                for c in content
                            )
                        parts.append(str(content))
                return "\n".join(p for p in parts if p)
            return raw
        return str(raw)

    @staticmethod
    def _event_content(events: Any, name: str) -> str:
        if isinstance(events, list):
            for event in events:
                if isinstance(event, dict) and event.get("name") == name:
                    attrs = OTARParser._normalize_attributes(event.get("attributes", {}))
                    return str(attrs.get("content", ""))
        return ""

    @classmethod
    def parse(
        cls,
        span_attributes: dict[str, Any],
        events: Any = None,
        span_name: str = "",
    ) -> StepOTAR:
        attrs = cls._normalize_attributes(span_attributes)
        event_observation = cls._event_content(events, "gen_ai.prompt")
        event_result = cls._event_content(events, "gen_ai.completion")

        observation = (
            event_observation
            or cls._messages_content(attrs.get("gen_ai.input.messages"))
            or str(attrs.get("gen_ai.request.prompt", ""))
        )
        result = (
            event_result
            or cls._messages_content(attrs.get("gen_ai.output.messages"))
            or str(attrs.get("gen_ai.completion", ""))
        )
        action = str(
            attrs.get("gen_ai.operation.name")
            or attrs.get("gen_ai.operation")
            or span_name
            or ""
        )
        thought = str(attrs.get("agent_reflex.agent.thought", ""))
        return StepOTAR(
            observation=observation,
            thought=thought,
            action=action,
            result=result,
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
        self._graph: nx.DiGraph[str] = nx.DiGraph()
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
