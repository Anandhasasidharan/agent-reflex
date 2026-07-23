from __future__ import annotations

import json
from typing import Any

from agent_reflex.classification.mast_plus import MastPlusClassifier
from agent_reflex.common.config import Settings
from agent_reflex.common.types import (
    AttributionResult,
    CausalGraphNode,
)
from agent_reflex.graph.models import CausalGraph

ORACLE_PROMPT = """You are verifying whether a step in a multi-agent system produced the correct output given its input and the overall task.

Step input (observation):
{observation}

Step thought process:
{thought}

Step action:
{action}

Step output (result):
{result}

Overall task context:
{task_context}

Question: Did this step produce a *correct* output given its input?
Be strict: if the output is irrelevant, factually wrong, or doesn't follow from the input, answer NO.
If the output is reasonable and correct, answer YES.

Return JSON: {{"correct": boolean, "reasoning": "short explanation"}}
"""


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
        self._client: Any | None = None

    @property
    def client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self._settings.openai_api_key)
        return self._client

    def _call_llm_json(self, prompt: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return json.loads(response.choices[0].message.content)

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
        subtasks = graph.decompose_into_subtasks()

        failed_subtasks: list[str] = []
        for sid, nodes in subtasks.items():
            summary = self._summarize_subtask(nodes)
            oracle_prompt = ORACLE_PROMPT.format(
                observation=summary["observation"],
                thought=summary["thought"],
                action=summary["action"],
                result=summary["result"],
                task_context=task_context,
            )
            result = self._call_llm_json(oracle_prompt)
            if not result.get("correct", True):
                failed_subtasks.append(sid)

        if not failed_subtasks:
            return None

        candidates = [
            n for n in reversed_nodes
            if n.subtask_id in failed_subtasks and n.error_flag
        ]
        return candidates[0] if candidates else reversed_nodes[0]

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
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
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
        return response.choices[0].message.content or ""
