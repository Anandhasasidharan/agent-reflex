from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session, sessionmaker

from agent_reflex.common.config import Settings
from agent_reflex.common.types import (
    AttributionResult,
    RecoveryOutcome,
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
        cutoff = datetime.now(UTC)
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
