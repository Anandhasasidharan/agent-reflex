from __future__ import annotations

import random
from collections import defaultdict

from agent_reflex.common.config import Settings
from agent_reflex.common.types import FailureSignature, Playbook, RecoveryOutcome

from .playbooks import PlaybookLibrary


class ContextualBanditSelector:
    def __init__(
        self,
        library: PlaybookLibrary | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._library = library or PlaybookLibrary()
        self._settings = settings or Settings()
        self._epsilon = self._settings.bandit_epsilon
        self._epsilon_decay = self._settings.bandit_epsilon_decay

        self._q_values: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def select(self, signature: FailureSignature) -> Playbook | None:
        mode_str = signature.mast_label.mode.value
        candidates = self._library.matching_playbooks(signature.mast_label.mode)
        if not candidates:
            return None

        if random.random() < self._epsilon:
            chosen = random.choice(candidates)
        else:
            best_score = -float("inf")
            chosen = candidates[0]
            for candidate in candidates:
                score = self._q_values[mode_str].get(candidate.name, 0.0)
                if score > best_score:
                    best_score = score
                    chosen = candidate

        return chosen

    def update(self, outcome: RecoveryOutcome) -> None:
        signature_key = self._to_signature_key(outcome)
        reward = 1.0 if outcome.success else (0.5 if outcome.partial else 0.0)

        self._counts[signature_key][outcome.playbook_name] += 1
        n = self._counts[signature_key][outcome.playbook_name]
        current_q = self._q_values[signature_key][outcome.playbook_name]
        self._q_values[signature_key][outcome.playbook_name] = current_q + (
            reward - current_q
        ) / n

        self._epsilon = max(0.01, self._epsilon * self._epsilon_decay)

    def _to_signature_key(self, outcome: RecoveryOutcome) -> str:
        return outcome.session_id.rsplit("_", 1)[0] if "_" in outcome.session_id else "default"

    def get_stats(self) -> dict[str, dict[str, float]]:
        return {
            mode: dict(qs)
            for mode, qs in self._q_values.items()
        }

    @property
    def name(self) -> str:
        return "adaptive"
