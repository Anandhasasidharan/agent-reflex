from __future__ import annotations

from agent_reflex.common.config import Settings
from agent_reflex.common.llm import LLMClient
from agent_reflex.common.types import MastMode, MastPlusLabel

MAST_EXAMPLES: list[dict] = [
    {
        "trace": "Agent A was asked to generate a SQL query but produced Python code instead because the task description said 'query the database' without specifying the query language.",
        "label": "spec_ambiguous",
    },
    {
        "trace": "Agent B was told to 'summarize the report' but the report was 200 pages and no truncation strategy was provided. Agent B exceeded its context window and produced a hallucinated summary of a different document.",
        "label": "spec_incomplete",
    },
    {
        "trace": "Two agents were given conflicting priorities: Agent C was told 'respond within 5 seconds', while Agent D was told 'verify every fact with citations'. Agent C finished first but Agent D rejected its output, causing a deadlock.",
        "label": "spec_contradictory",
    },
    {
        "trace": "Agent E assumed Agent F would validate date formats; Agent F assumed Agent E had already done it. A malformed date '2026-13-01' passed through both undetected.",
        "label": "coord_misaligned_assumptions",
    },
    {
        "trace": "Agent G finished its analysis but marked it with 99% confidence. The result was factually wrong because it never actually checked the source data — it just summarized its own prior output.",
        "label": "verif_overconfident",
    },
    {
        "trace": "Agent H completed every step correctly per its instructions, but the instructions themselves asked for 'user count by country' while the actual requirement was 'unique monthly active users by region'.",
        "label": "task_derailment",
    },
    {
        "trace": "The LLM call returned 'According to the 2023 report, revenue was $2.1B' but no such report existed. The agent generated a plausible-sounding fact that was entirely fabricated.",
        "label": "task_hallucination",
    },
    {
        "trace": "Execution failed because OpenAI returned a 429 rate limit error after 50 rapid-fire consecutive tool calls with no backoff.",
        "label": "infra_rate_limit",
    },
    {
        "trace": "The agent's conversation history grew to 180K tokens and the LLM API returned a 400 error: 'maximum context length exceeded'. The agent had no memory management or summarization layer.",
        "label": "infra_context_window",
    },
    {
        "trace": "A five-agent team was deployed. Agent A timed out waiting for Agent B's output. Agent B was waiting for Agent C. Agent C had crashed silently. All three cascading timeouts triggered within 30 seconds.",
        "label": "infra_cascade_timeout",
    },
]


MAST_FEW_SHOT_PROMPT = """You are a failure-mode classifier for multi-agent AI systems.
Classify the failure into exactly one of these 18 categories:

Specification failures (the task/instruction was the problem):
- spec_ambiguous: instruction can be interpreted multiple ways
- spec_incomplete: instruction missing critical details
- spec_contradictory: instruction contains conflicting requirements
- spec_missing: no instruction existed for a required behavior

Coordination failures (agents conflicted with each other):
- coord_misaligned_goals: agents pursuing incompatible objectives
- coord_misaligned_assumptions: agents assuming different things about each other's behavior
- coord_resource_contention: agents competing for the same resource
- coord_deadlock: agents blocked waiting for each other in a cycle

Verification failures (agent's self-check was wrong):
- verif_overconfident: agent claimed success when it failed
- verif_underconfident: agent flagged false positive errors
- verif_wrong_criterion: agent checked the wrong thing
- verif_self_inconsistent: agent gave different answers to the same question

Task execution failures:
- task_derailment: agent did the wrong task correctly
- task_hallucination: agent fabricated information

Infrastructure/Operational failures:
- infra_rate_limit: API rate limit exceeded
- infra_context_window: context window overflow
- infra_cascade_timeout: cascading timeouts across agents
- infra_unknown: other infrastructure failure

Return only a JSON object with keys:
- "mode": the most specific matching category
- "confidence": a float 0.0 to 1.0
- "reasoning": one-sentence explanation

Examples:
{examples}

Trace to classify:
{trace}
"""


class MastPlusClassifier:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._llm = LLMClient(self._settings)

    def classify(self, trace_text: str) -> MastPlusLabel:
        examples_text = "\n\n".join(
            f"Trace: {ex['trace']}\nLabel: {ex['label']}"
            for ex in MAST_EXAMPLES
        )
        prompt = MAST_FEW_SHOT_PROMPT.format(examples=examples_text, trace=trace_text)

        result = self._llm.chat_json(
            messages=[
                {"role": "system", "content": "You classify multi-agent failures precisely."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )

        mode = MastMode(result["mode"])
        return MastPlusLabel(mode=mode, confidence=result.get("confidence", 0.0))

    def classify_from_graph(self, graph_dict: dict) -> MastPlusLabel:
        steps = graph_dict.get("nodes", [])
        edges = graph_dict.get("edges", [])
        trace_parts = []
        for step in steps:
            trace_parts.append(
                f"[{step['agent_id']}] step {step['step_index']}: "
                f"action={step['otar']['action']}, "
                f"thought={step['otar']['thought'][:200]}, "
                f"result={step['otar']['result'][:200]}, "
                f"error={step['error_flag']}"
            )
        for edge in edges:
            trace_parts.append(f"  {edge['source_id']} -[{edge['edge_type']}]-> {edge['target_id']}")

        return self.classify("\n".join(trace_parts))
