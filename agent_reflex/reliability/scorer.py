from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from agent_reflex.common.types import RecoveryOutcome, TrackedSession


class ReliabilityScorer:
    def __init__(self, window_size: int = 20) -> None:
        self._sessions: dict[str, list[TrackedSession]] = defaultdict(list)
        self._window_size = window_size

    def record_session(self, session: TrackedSession) -> None:
        self._sessions[session.agent_id].append(session)

    def record_from_outcome(
        self,
        agent_id: str,
        session_id: str,
        task_description: str,
        outcome: RecoveryOutcome | None = None,
    ) -> None:
        success = outcome.success if outcome else False
        partial = outcome.partial if outcome else False
        session = TrackedSession(
            session_id=session_id,
            agent_id=agent_id,
            task_description=task_description,
            success=success or partial,
            reliability_score=self._compute_single_score(success, partial),
            recovery=outcome,
        )
        self.record_session(session)

    def _compute_single_score(self, success: bool, partial: bool) -> float:
        if success:
            return 1.0
        if partial:
            return 0.5
        return 0.0

    def current_score(self, agent_id: str) -> float:
        sessions = self._sessions.get(agent_id, [])
        if not sessions:
            return 0.0
        return self._exponential_weighted_average(sessions)

    def current_score_with_trend(self, agent_id: str) -> dict[str, Any]:
        sessions = self._sessions.get(agent_id, [])
        if not sessions:
            return {"score": 0.0, "trend_pct": 0.0, "history": [], "n_sessions": 0}

        recent = sessions[-self._window_size:]
        scores = np.array([s.reliability_score for s in recent])
        weights = np.exp(np.linspace(0, 2, len(scores)))
        weights /= weights.sum()

        current = float(np.sum(scores * weights))
        mid_point = len(scores) // 2
        if mid_point > 0:
            first_half = float(np.mean(scores[:mid_point]))
            second_half = float(np.mean(scores[mid_point:]))
            trend_pct = ((second_half - first_half) / max(first_half, 0.01)) * 100
        else:
            trend_pct = 0.0

        return {
            "score": round(current, 4),
            "trend_pct": round(trend_pct, 2),
            "history": [s.reliability_score for s in sessions],
            "n_sessions": len(sessions),
        }

    def _exponential_weighted_average(self, sessions: list[TrackedSession]) -> float:
        recent = sessions[-self._window_size:]
        if not recent:
            return 0.0
        scores = np.array([s.reliability_score for s in recent])
        weights = np.exp(np.linspace(0, 2, len(scores)))
        weights /= weights.sum()
        return float(np.sum(scores * weights))

    def reliability_trend(self, agent_id: str, before_after_playbook: str) -> dict[str, float]:
        sessions = self._sessions.get(agent_id, [])
        before: list[float] = []
        after: list[float] = []
        seen_playbook = False

        for s in sessions:
            if s.recovery and s.recovery.playbook_name == before_after_playbook:
                seen_playbook = True
                after.append(s.reliability_score)
            elif not seen_playbook:
                before.append(s.reliability_score)
            else:
                after.append(s.reliability_score)

        before_mean = float(np.mean(before)) if before else 0.0
        after_mean = float(np.mean(after)) if after else 0.0
        improvement = ((after_mean - before_mean) / max(before_mean, 0.01)) * 100 if before else 0.0

        return {
            "before_playbook_mean": round(before_mean, 4),
            "after_playbook_mean": round(after_mean, 4),
            "improvement_pct": round(improvement, 2),
            "n_before": len(before),
            "n_after": len(after),
        }
