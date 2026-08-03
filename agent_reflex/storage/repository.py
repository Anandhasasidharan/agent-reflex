from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import Session, sessionmaker

from agent_reflex.common.config import Settings
from agent_reflex.common.types import (
    AttributionResult,
    CausalGraphNode,
    RecoveryOutcome,
    StepOTAR,
)
from agent_reflex.graph.models import CausalGraph

from .models import (
    Base,
    GraphEdgeRecord,
    RecoveryLogRecord,
    ReliabilityRecord,
    SessionRecord,
    TraceStepRecord,
)


class PostgresRepository:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._engine = create_engine(self._settings.db_url, pool_pre_ping=True)
        self._SessionLocal = sessionmaker(bind=self._engine)

    def init_db(self) -> None:
        Base.metadata.create_all(self._engine)
        # Idempotent migrations for tables created by an older schema
        # (create_all never alters existing tables).
        try:
            with self._session() as db:
                db.execute(text("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS evidence_json TEXT"))
                db.commit()
        except Exception:
            pass

    def _session(self) -> Session:
        return self._SessionLocal()

    def save_session(
        self,
        session_id: str,
        agent_id: str,
        task_description: str,
        graph: CausalGraph,
        attribution: AttributionResult | None = None,
    ) -> None:
        with self._session() as db:
            existing = db.query(SessionRecord).filter_by(session_id=session_id).first()
            if existing:
                return

            session = SessionRecord(
                session_id=session_id,
                agent_id=agent_id,
                task_description=task_description,
                success=0,
                reliability_score=0.0,
                failure_type=attribution.failure_type.value if attribution else None,
                cause_node_id=attribution.cause_node_id if attribution else None,
                causal_responsibility_score=attribution.causal_responsibility_score if attribution else 0.0,
                evidence_json=json.dumps(attribution.evidence) if attribution else None,
            )
            db.add(session)

            for node in graph.get_all_nodes():
                db.add(TraceStepRecord(
                    session_id=session_id,
                    node_id=node.node_id,
                    agent_id=node.agent_id,
                    step_index=node.step_index,
                    observation=node.otar.observation[:10000],
                    thought=node.otar.thought[:10000],
                    action=node.otar.action,
                    result=node.otar.result[:10000],
                    parent_id=node.parent_id,
                    subtask_id=node.subtask_id,
                    execution_time_ms=node.execution_time_ms,
                    error_flag=1 if node.error_flag else 0,
                ))

            for edge in graph.get_edges():
                db.add(GraphEdgeRecord(
                    session_id=session_id,
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    edge_type=edge.edge_type,
                ))

            db.commit()

    def save_recovery_outcome(self, outcome: RecoveryOutcome, selector: str = "adaptive", agent_id: str = "") -> None:
        with self._session() as db:
            db.add(RecoveryLogRecord(
                session_id=outcome.session_id,
                agent_id=agent_id,
                playbook_name=outcome.playbook_name,
                selector=selector,
                success=1 if outcome.success else 0,
                partial=1 if outcome.partial else 0,
                recovery_time_ms=outcome.recovery_time_ms,
            ))
            db.commit()

    def save_reliability_score(self, agent_id: str, session_id: str, score: float) -> None:
        with self._session() as db:
            db.add(ReliabilityRecord(
                agent_id=agent_id,
                session_id=session_id,
                score=score,
            ))
            db.commit()

    def get_heatmap(self, days: int = 30) -> list[dict[str, Any]]:
        cutoff = datetime.now(UTC) - timedelta(days=days)
        with self._session() as db:
            results = (
                db.query(
                    SessionRecord.failure_type,
                    func.date(SessionRecord.created_at).label("date"),
                    func.count(SessionRecord.id).label("count"),
                )
                .filter(SessionRecord.failure_type.isnot(None))
                .filter(SessionRecord.created_at >= cutoff)
                .group_by(SessionRecord.failure_type, func.date(SessionRecord.created_at))
                .all()
            )
            return [
                {"failure_type": r.failure_type, "date": str(r.date), "count": r.count}
                for r in results
            ]

    def get_recovery_breakdown(self) -> list[dict[str, Any]]:
        with self._session() as db:
            results = (
                db.query(
                    RecoveryLogRecord.playbook_name,
                    RecoveryLogRecord.selector,
                    func.count(RecoveryLogRecord.id).label("total"),
                    func.sum(RecoveryLogRecord.success).label("successes"),
                )
                .group_by(RecoveryLogRecord.playbook_name, RecoveryLogRecord.selector)
                .all()
            )
            return [
                {
                    "playbook": r.playbook_name,
                    "selector": r.selector,
                    "total": r.total,
                    "successes": r.successes or 0,
                    "success_rate": round((r.successes or 0) / r.total, 3) if r.total > 0 else 0.0,
                }
                for r in results
            ]

    def get_reliability_history(self, agent_id: str) -> list[dict[str, Any]]:
        with self._session() as db:
            results = (
                db.query(ReliabilityRecord)
                .filter_by(agent_id=agent_id)
                .order_by(ReliabilityRecord.created_at)
                .all()
            )
            return [
                {"session_id": r.session_id, "score": r.score, "created_at": str(r.created_at)}
                for r in results
            ]

    def get_session_count(self) -> int:
        with self._session() as db:
            return db.query(SessionRecord).count()

    def get_failure_type_counts(self) -> dict[str, int]:
        with self._session() as db:
            results = (
                db.query(
                    SessionRecord.failure_type,
                    func.count(SessionRecord.id).label("count"),
                )
                .filter(SessionRecord.failure_type.isnot(None))
                .group_by(SessionRecord.failure_type)
                .all()
            )
            return {r[0]: int(r[1]) for r in results}

    # ------------------------------------------------------------------
    # Session browsing (dashboard / frontend)
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """One session summary (including persisted attribution fields).

        Returns None when the session does not exist. Attribution fields are
        null when LLM attribution failed at ingest time (best-effort path).
        """
        with self._session() as db:
            record = db.query(SessionRecord).filter_by(session_id=session_id).first()
            if record is None:
                return None
            return self._session_summary(record)

    @staticmethod
    def _session_summary(record: SessionRecord) -> dict[str, Any]:
        evidence: list[str] = []
        if record.evidence_json:
            try:
                parsed = json.loads(cast(str, record.evidence_json))
                evidence = parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                evidence = []
        return {
            "session_id": record.session_id,
            "agent_id": record.agent_id,
            "task_description": record.task_description,
            "failure_type": record.failure_type,
            "cause_node_id": record.cause_node_id,
            "causal_responsibility_score": record.causal_responsibility_score,
            "evidence": evidence,
            "created_at": str(record.created_at) if record.created_at else None,
        }

    def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
        agent_id: str | None = None,
        failure_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Paginated session list, newest first. Returns (items, total)."""
        with self._session() as db:
            query = db.query(SessionRecord)
            if agent_id:
                query = query.filter(SessionRecord.agent_id == agent_id)
            if failure_type:
                query = query.filter(SessionRecord.failure_type == failure_type)
            if since is not None:
                query = query.filter(SessionRecord.created_at >= since)
            if until is not None:
                query = query.filter(SessionRecord.created_at <= until)

            total = query.count()
            items = (
                query.order_by(SessionRecord.created_at.desc(), SessionRecord.id.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [self._session_summary(r) for r in items], total

    def get_graph(self, session_id: str) -> CausalGraph | None:
        """Reconstruct the stored CausalGraph for a session, or None."""
        with self._session() as db:
            record = db.query(SessionRecord).filter_by(session_id=session_id).first()
            if record is None:
                return None

            graph = CausalGraph()
            steps = (
                db.query(TraceStepRecord)
                .filter_by(session_id=session_id)
                .order_by(TraceStepRecord.step_index)
                .all()
            )
            for step in steps:
                graph.add_step(CausalGraphNode(
                    node_id=cast(str, step.node_id),
                    agent_id=cast(str, step.agent_id),
                    step_index=cast(int, step.step_index),
                    otar=StepOTAR(
                        observation=cast(str, step.observation),
                        thought=cast(str, step.thought),
                        action=cast(str, step.action),
                        result=cast(str, step.result),
                    ),
                    parent_id=cast("str | None", step.parent_id),
                    subtask_id=cast("str | None", step.subtask_id),
                    execution_time_ms=cast(float, step.execution_time_ms),
                    error_flag=cast(bool, step.error_flag),
                ))

            edges = db.query(GraphEdgeRecord).filter_by(session_id=session_id).all()
            for edge in edges:
                graph.add_dependency(cast(str, edge.source_id), cast(str, edge.target_id))
            return graph

    def get_agent_reliability_summary(self, window: int = 10) -> dict[str, dict[str, Any]]:
        """Per-agent reliability history (last `window` scores) + session counts.

        Scores are recorded at ingest (derived from the trace's error flags)
        and on recovery feedback, so this reflects real persisted data.
        """
        with self._session() as db:
            agents: dict[str, dict[str, Any]] = {}
            records = (
                db.query(ReliabilityRecord)
                .order_by(ReliabilityRecord.agent_id, ReliabilityRecord.created_at.desc())
                .all()
            )
            for record in records:
                entry = agents.setdefault(cast(str, record.agent_id), {"scores": [], "n_sessions": 0})
                if len(entry["scores"]) < window:
                    entry["scores"].append(record.score)

            counts = (
                db.query(SessionRecord.agent_id, func.count(SessionRecord.id).label("count"))
                .group_by(SessionRecord.agent_id)
                .all()
            )
            for agent_id, count in counts:
                agents.setdefault(agent_id, {"scores": [], "n_sessions": 0})["n_sessions"] = int(count)
            for entry in agents.values():
                entry["scores"] = list(reversed(entry["scores"]))
            return agents
