from __future__ import annotations

from agent_reflex.common.types import FailureSignature, MastMode, Playbook

DEFAULT_PLAYBOOKS: list[Playbook] = [
    Playbook(
        name="re_prompt",
        failure_patterns=[
            "spec_ambiguous", "spec_incomplete", "verif_wrong_criterion",
        ],
        steps=[
            "Clarify the ambiguous or missing instruction",
            "Re-prompt with more specific requirements",
            "Verify the new output against the clarified spec",
        ],
        max_retries=2,
    ),
    Playbook(
        name="backtrack_to_checkpoint",
        failure_patterns=[
            "task_derailment", "task_hallucination", "spec_contradictory",
        ],
        steps=[
            "Identify the last known-good checkpoint step",
            "Roll back execution to that checkpoint",
            "Re-run from checkpoint with corrected context",
        ],
        max_retries=1,
    ),
    Playbook(
        name="swap_agent",
        failure_patterns=[
            "coord_misaligned_assumptions", "coord_deadlock",
            "verif_overconfident", "verif_self_inconsistent",
        ],
        steps=[
            "Route the failing task to a different agent instance",
            "Provide the other agent with full context of prior attempts",
            "Monitor new agent's output for correctness",
        ],
        max_retries=3,
    ),
    Playbook(
        name="escalate_to_human",
        failure_patterns=[
            "coord_misaligned_goals", "spec_missing",
        ],
        steps=[
            "Package the failure context and current state",
            "Route to human-in-the-loop queue",
            "Wait for human decision or override",
        ],
        max_retries=0,
    ),
    Playbook(
        name="circuit_breaker",
        failure_patterns=[
            "infra_rate_limit", "infra_cascade_timeout",
        ],
        steps=[
            "Trip circuit breaker — stop all new requests",
            "Wait for backoff period (exponential: 2^N seconds)",
            "Test with a single health-check request",
            "Gradually restore traffic if health-check passes",
        ],
        max_retries=3,
    ),
    Playbook(
        name="rate_limit_backoff",
        failure_patterns=["infra_rate_limit"],
        steps=[
            "Extract Retry-After header or use default 60s backoff",
            "Implement exponential backoff: 2^retry_count * 30s",
            "Queue pending requests and replay after backoff",
        ],
        max_retries=5,
    ),
    Playbook(
        name="context_window_summarize",
        failure_patterns=["infra_context_window"],
        steps=[
            "Extract the oldest conversation turns exceeding context budget",
            "Summarize extracted turns into a compact memory block",
            "Replace extracted turns with the summary",
            "Re-try the failing call",
        ],
        max_retries=2,
    ),
    Playbook(
        name="tool_fallback",
        failure_patterns=[
            "coord_resource_contention", "infra_unknown",
        ],
        steps=[
            "Identify the failing tool or resource",
            "Switch to a fallback tool with equivalent capability",
            "Log the fallback event for observability",
        ],
        max_retries=3,
    ),
]


class PlaybookLibrary:
    def __init__(self, playbooks: list[Playbook] | None = None) -> None:
        self._playbooks = playbooks or DEFAULT_PLAYBOOKS

    def list_playbooks(self) -> list[Playbook]:
        return list(self._playbooks)

    def matching_playbooks(self, mode: MastMode) -> list[Playbook]:
        mode_str = mode.value
        return [p for p in self._playbooks if mode_str in p.failure_patterns]

    def get(self, name: str) -> Playbook | None:
        for p in self._playbooks:
            if p.name == name:
                return p
        return None


class StaticRecoverySelector:
    def __init__(self, library: PlaybookLibrary | None = None) -> None:
        self._library = library or PlaybookLibrary()

    def select(self, signature: FailureSignature) -> Playbook | None:
        candidates = self._library.matching_playbooks(signature.mast_label.mode)
        return candidates[0] if candidates else None

    @property
    def name(self) -> str:
        return "static"
