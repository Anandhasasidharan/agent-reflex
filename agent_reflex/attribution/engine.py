from __future__ import annotations

from typing import Any

from agent_reflex.classification.mast_plus import MastPlusClassifier
from agent_reflex.common.config import Settings
from agent_reflex.common.llm import LLMClient
from agent_reflex.common.types import (
    AttributionResult,
    CausalGraphNode,
)
from agent_reflex.graph.models import CausalGraph

COUNTERFACTUAL_PROMPT = """Given this step that may have caused a failure:

Step input:
{observation}

Step thought:
{thought}

Step action:
{action}

Step output (actual, possibly wrong):
{result}

Hypothetical corrected output:
{corrected_output}

Question: If the step had produced the corrected output instead, would the overall task outcome have been different (i.e., succeeded instead of failed)?
Return JSON: {{"outcome_would_change": boolean, "confidence_pct": 0-100, "reasoning": "short explanation"}}
"""


class AttributionEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        classifier: MastPlusClassifier | None = None,
    ) -> None:
        self._settings = settings or Settings()
        self._classifier = classifier or MastPlusClassifier(settings)
        self._llm = LLMClient(self._settings)

    def _call_llm_json(self, prompt: str) -> dict[str, Any]:
        return self._llm.chat_json(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

    def attribute(
        self,
        session_id: str,
        graph: CausalGraph,
        task_context: str = "",
    ) -> AttributionResult:
        failure_label = self._classifier.classify_from_graph(graph.to_dict())

        reversed_nodes = sorted(
            graph.get_all_nodes(),
            key=lambda n: n.step_index,
            reverse=True,
        )

        cause_node = self._oracle_guided_backtracking(graph, reversed_nodes, task_context)
        if cause_node is None:
            cause_node = reversed_nodes[0]

        crs = self._counterfactual_screening(graph, cause_node, task_context)

        evidence = [
            f"{failure_label.mode.value} detected with confidence {failure_label.confidence:.2f}",
        ]
        if crs > 0.5:
            evidence.append(
                f"Correcting step {cause_node.node_id} would change the outcome (CRS={crs:.2f})"
            )

        return AttributionResult(
            session_id=session_id,
            failure_type=failure_label.mode,
            cause_node_id=cause_node.node_id,
            causal_responsibility_score=crs,
            evidence=evidence,
        )

    def _oracle_guided_backtracking(
        self,
        graph: CausalGraph,
        reversed_nodes: list[CausalGraphNode],
        task_context: str,
    ) -> CausalGraphNode | None:
        """Locate the root cause as the earliest error-flagged step.

        The causal topology carries the signal: a failure cascade starts at
        the most-upstream step that went wrong, and later errors are symptoms
        of that first error. An LLM "is this output correct" oracle is
        unreliable here because steps legitimately produce error-shaped
        outputs (timeouts, 429s, rejected reviews), so it systematically
        flags nothing and defaults to the *last* errored step.
        """
        error_nodes = sorted(
            (n for n in graph.get_all_nodes() if n.error_flag),
            key=lambda n: n.step_index,
        )
        if not error_nodes:
            return None
        return error_nodes[0]

    def _summarize_subtask(self, nodes: list[CausalGraphNode]) -> dict[str, str]:
        return {
            "observation": nodes[0].otar.observation if nodes else "",
            "thought": " | ".join(n.otar.thought for n in nodes if n.otar.thought),
            "action": " -> ".join(n.otar.action for n in nodes if n.otar.action),
            "result": nodes[-1].otar.result if nodes else "",
        }

    def _counterfactual_screening(
        self,
        graph: CausalGraph,
        candidate: CausalGraphNode,
        task_context: str,
    ) -> float:
        corrected_output = self._synthesize_corrected_output(candidate, task_context)

        prompt = COUNTERFACTUAL_PROMPT.format(
            observation=candidate.otar.observation,
            thought=candidate.otar.thought,
            action=candidate.otar.action,
            result=candidate.otar.result,
            corrected_output=corrected_output,
        )

        result = self._call_llm_json(prompt)
        confidence_pct = result.get("confidence_pct")
        if confidence_pct is not None:
            return min(max(confidence_pct / 100.0, 0.0), 1.0)
        return 0.95 if result.get("outcome_would_change", False) else 0.15

    def _synthesize_corrected_output(
        self,
        node: CausalGraphNode,
        task_context: str,
    ) -> str:
        return self._llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": "Given a step that produced a wrong output, synthesize what the correct output should have been. Be concise.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Task: {task_context}\n"
                        f"Step input: {node.otar.observation}\n"
                        f"Step action: {node.otar.action}\n"
                        f"Actual (wrong) output: {node.otar.result}\n"
                        f"What should the correct output have been?"
                    ),
                },
            ],
            temperature=0.1,
        )
