from __future__ import annotations

from difflib import SequenceMatcher

import numpy as np
from scipy.spatial.distance import cosine

from agent_reflex.common.config import Settings
from agent_reflex.common.llm import LLMClient


class ConsistencyScorer:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._llm = LLMClient(self._settings)

    def _get_embedding(self, text: str) -> list[float]:
        return self._llm.embed(text)

    def score(self, prompt: str, n_samples: int | None = None) -> float:
        n = n_samples or self._settings.consistency_n_samples
        temperature = self._settings.consistency_temperature

        responses: list[str] = []
        for _ in range(n):
            content = self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            responses.append(content)

        return self._measure_agreement(responses)

    def _measure_agreement(self, responses: list[str]) -> float:
        if len(responses) < 2:
            return 1.0

        try:
            embeddings = [self._get_embedding(r) for r in responses]
            if len(embeddings) < 2:
                return 1.0

            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = 1.0 - cosine(embeddings[i], embeddings[j])
                    similarities.append(sim)

            return float(np.mean(similarities)) if similarities else 1.0
        except Exception:
            return self._lexical_agreement(responses)

    def _lexical_agreement(self, responses: list[str]) -> float:
        """Fallback when no embedding provider is available (e.g. DeepSeek)."""
        if len(responses) < 2:
            return 1.0

        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sim = SequenceMatcher(None, responses[i], responses[j]).ratio()
                similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 1.0


class UncertaintyEscalationController:
    def __init__(
        self,
        scorer: ConsistencyScorer | None = None,
        threshold: float = 0.7,
    ) -> None:
        self._scorer = scorer or ConsistencyScorer()
        self._threshold = threshold

    def should_escalate(self, prompt: str, is_critical: bool = False) -> tuple[bool, float]:
        score = self._scorer.score(prompt)
        if is_critical and score < self._threshold:
            return True, score
        if score < self._threshold * 0.5:
            return True, score
        return False, score

    def calibrate_threshold(
        self,
        labeled_data: list[tuple[str, bool]],
    ) -> float:
        from sklearn.metrics import roc_auc_score

        scores = []
        labels = []
        for prompt, is_failure in labeled_data:
            score = self._scorer.score(prompt)
            scores.append(score)
            labels.append(1 if is_failure else 0)

        if len(set(labels)) < 2:
            return self._threshold

        auroc = roc_auc_score(labels, [-s for s in scores])
        print(f"[UncertaintyEscalation] Calibration AUROC: {auroc:.3f} (random baseline ~0.5)")

        best_threshold = self._threshold
        best_f1 = 0.0
        for t in np.linspace(0.1, 0.95, 18):
            preds = [1 if s < t else 0 for s in scores]
            tp = sum(1 for p, label in zip(preds, labels) if p == 1 and label == 1)
            fp = sum(1 for p, label in zip(preds, labels) if p == 1 and label == 0)
            fn = sum(1 for p, label in zip(preds, labels) if p == 0 and label == 1)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        self._threshold = best_threshold
        return best_threshold
