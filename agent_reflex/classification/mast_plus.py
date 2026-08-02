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
        "trace": "Agent G was told to maximize throughput; Agent H was told to maximize correctness on the same shared dataset. Agent G kept writing partial rows while Agent H kept rolling back the transaction it deemed incomplete. Neither goal was wrong alone; the two goals conflicted.",
        "label": "coord_misaligned_goals",
    },
    {
        "trace": "Two agents both wanted a validated answer to proceed. Agent I would not release its intermediate output until Agent J acknowledged it, and Agent J would not acknowledge anything until Agent I released it. Both ended it up blocked forever.",
        "label": "coord_deadlock",
    },
    {
        "trace": "Four agents all fired the same expensive external model call simultaneously to answer overlapping subquestions, and the shared GPU node returned an OOM error. No agent owned the shared resource or serialized access to it.",
        "label": "coord_resource_contention",
    },
    {
        "trace": "Agent G finished its analysis but marked it with 99% confidence. The result was factually wrong because it never validated the source data against the actual ledger — it simply re-signed its own prior inference. Output reads confidently but is wrong.",
        "label": "verif_overconfident",
    },
    {
        "trace": "Agent J computed the correct answer twice, but its two independent draft samples differed in wording, so it flagged the result as 'likely wrong' and escalated it for human review. Its own numeric answer matched ground truth exactly. The check was far too strict.",
        "label": "verif_underconfident",
    },
    {
        "trace": "Agent K was asked 'is the total correct?', so it verified the total against itself, re-adding the two line items and confirming the arithmetic. But the numbers themselves were stale from a week-old cache, and the actual amounts had changed. It checked procedure; the question was really about the data.",
        "label": "verif_wrong_criterion",
    },
    {
        "trace": "Agent L answered 'the cost is 120' to a question in one step, then said 'the cost is 80' in a later step for the same item. No retraction or correction notes were found — the two answers disagree without explanation.",
        "label": "verif_self_inconsistent",
    },
    {
        "trace": "Agent H completed every step correctly per its instructions, but the instruction itself asked for 'user count by country' while the actual business requirement was 'unique monthly active users by region'. The work was valid; the task asked for it was the wrong task.",
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
        "trace": "A five-agent team was deployed. Agent A timed out waiting for Agent B's output. Agent B was waiting Agent C. Agent C had crashed silently. All three timed out in cascade within 30 seconds.",
        "label": "infra_cascade_timeout",
    },
    {
        "trace": "The agent was cut off mid-call by a transparent `transport connection reset` / DNS failure that was not a semantic or agentic error at all, and there was no fallback path to reopen the connection.",
        "label": "infra_unknown",
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
        steps = sorted(steps, key=lambda s: s["step_index"])
        trace_parts = []
        edges = graph_dict.get("edges", [])
        for step in steps:
            flag = "ERROR" if step["error_flag"] else "OK"
            otar = step["otar"]
            trace_parts.append(
                f"[{step['agent_id']}|{flag}] step {step['step_index']} "
                f"(subtask={step.get('subtask_id', '?')}): "
                f"action={otar['action']}; "
                f"thought={otar['thought'][:200]}; "
                f"result={otar['result'][:200]}"
            )
        for edge in edges:
            trace_parts.append(f"  {edge['source_id']} -[{edge['edge_type']}]-> {edge['target_id']}")

        return self.classify("\n".join(trace_parts))
