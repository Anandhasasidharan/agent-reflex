from __future__ import annotations

from neo4j import Driver, GraphDatabase, Session

from agent_reflex.common.config import Settings

from .models import CausalGraph, CausalGraphNode


class Neo4jGraphStore:
    def __init__(self, settings: Settings | None = None) -> None:
        if settings is None:
            settings = Settings()
        self._driver: Driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_pass),
        )

    def close(self) -> None:
        self._driver.close()

    def save_graph(self, session_id: str, graph: CausalGraph) -> None:
        with self._driver.session() as session:
            session.execute_write(self._create_graph_tx, session_id, graph)

    @staticmethod
    def _create_graph_tx(tx: Session, session_id: str, graph: CausalGraph) -> None:
        tx.run("MERGE (:Session {id: $sid})", sid=session_id)
        for node in graph.get_all_nodes():
            tx.run(
                """
                MERGE (n:Step {node_id: $nid, session_id: $sid})
                SET n.agent_id = $aid,
                    n.step_index = $idx,
                    n.observation = $obs,
                    n.thought = $thought,
                    n.action = $action,
                    n.result = $result,
                    n.subtask_id = $sub,
                    n.error_flag = $err,
                    n.execution_time_ms = $time
                """,
                nid=node.node_id,
                sid=session_id,
                aid=node.agent_id,
                idx=node.step_index,
                obs=node.otar.observation[:10000],
                thought=node.otar.thought[:10000],
                action=node.otar.action,
                result=node.otar.result[:10000],
                sub=node.subtask_id,
                err=node.error_flag,
                time=node.execution_time_ms,
            )
        for edge in graph.get_edges():
            tx.run(
                """
                MATCH (a:Step {node_id: $src, session_id: $sid})
                MATCH (b:Step {node_id: $dst, session_id: $sid})
                MERGE (a)-[r:DEPENDS_ON {type: $etype}]->(b)
                """,
                src=edge.source_id,
                dst=edge.target_id,
                sid=session_id,
                etype=edge.edge_type,
            )

    def load_graph(self, session_id: str) -> CausalGraph | None:
        with self._driver.session() as session:
            result = session.execute_read(self._load_graph_tx, session_id)
            return result

    @staticmethod
    def _load_graph_tx(tx: Session, session_id: str) -> CausalGraph | None:
        nodes_result = tx.run(
            """
            MATCH (n:Step {session_id: $sid})
            RETURN n.node_id AS node_id, n.agent_id AS agent_id,
                   n.step_index AS step_index, n.observation AS observation,
                   n.thought AS thought, n.action AS action,
                   n.result AS result, n.subtask_id AS subtask_id,
                   n.error_flag AS error_flag, n.execution_time_ms AS execution_time_ms
            ORDER BY n.step_index
            """,
            sid=session_id,
        )
        records = list(nodes_result)
        if not records:
            return None

        cg = CausalGraph()
        for record in records:
            from agent_reflex.common.types import StepOTAR
            node = CausalGraphNode(
                node_id=record["node_id"],
                agent_id=record["agent_id"],
                step_index=record["step_index"],
                otar=StepOTAR(
                    observation=record.get("observation", ""),
                    thought=record.get("thought", ""),
                    action=record.get("action", ""),
                    result=record.get("result", ""),
                ),
                parent_id=None,
                subtask_id=record.get("subtask_id"),
                execution_time_ms=record.get("execution_time_ms", 0.0),
                error_flag=record.get("error_flag", False),
            )
            cg._nodes[node.node_id] = node
            cg._graph.add_node(node.node_id)

        edges_result = tx.run(
            """
            MATCH (a:Step {session_id: $sid})-[r:DEPENDS_ON]->(b:Step {session_id: $sid})
            RETURN a.node_id AS src, b.node_id AS dst, r.type AS etype
            """,
            sid=session_id,
        )
        for record in edges_result:
            cg._graph.add_edge(record["src"], record["dst"], edge_type=record["etype"])

        return cg
