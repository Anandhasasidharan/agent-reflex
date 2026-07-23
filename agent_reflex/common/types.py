from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MastMode(Enum):
    SPEC_AMBIGUOUS = "spec_ambiguous"
    SPEC_INCOMPLETE = "spec_incomplete"
    SPEC_CONTRADICTORY = "spec_contradictory"
    SPEC_MISSING = "spec_missing"
    COORD_MISALIGNED_GOALS = "coord_misaligned_goals"
    COORD_MISALIGNED_ASSUMPTIONS = "coord_misaligned_assumptions"
    COORD_RESOURCE_CONTENTION = "coord_resource_contention"
    COORD_DEADLOCK = "coord_deadlock"
    VERIF_OVERCONFIDENT = "verif_overconfident"
    VERIF_UNDERCONFIDENT = "verif_underconfident"
    VERIF_WRONG_CRITERION = "verif_wrong_criterion"
    VERIF_SELF_INCONSISTENT = "verif_self_inconsistent"
    TASK_DERAILMENT = "task_derailment"
    TASK_HALLUCINATION = "task_hallucination"
    INFRA_RATE_LIMIT = "infra_rate_limit"
    INFRA_CONTEXT_WINDOW = "infra_context_window"
    INFRA_CASCADE_TIMEOUT = "infra_cascade_timeout"
    INFRA_UNKNOWN = "infra_unknown"


@dataclass
class MastPlusLabel:
    mode: MastMode
    confidence: float = 0.0


@dataclass
class StepOTAR:
    observation: str
    thought: str
    action: str
    result: str


@dataclass
class CausalGraphNode:
    node_id: str
    agent_id: str
    step_index: int
    otar: StepOTAR
    parent_id: str | None
    subtask_id: str | None
    execution_time_ms: float
    error_flag: bool
    raw_span_attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalGraphEdge:
    source_id: str
    target_id: str
    edge_type: str  # "data_dependency" | "control_flow" | "subtask"


@dataclass
class AttributionResult:
    session_id: str
    failure_type: MastMode
    cause_node_id: str
    causal_responsibility_score: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class FailureSignature:
    session_id: str
    mast_label: MastPlusLabel
    cause_node_id: str
    agent_id: str
    crs: float
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class Playbook:
    name: str
    failure_patterns: list[str]
    steps: list[str]
    max_retries: int = 3


@dataclass
class RecoveryOutcome:
    session_id: str
    playbook_name: str
    success: bool
    partial: bool = False
    recovery_time_ms: float = 0.0


@dataclass
class TrackedSession:
    session_id: str
    agent_id: str
    task_description: str
    success: bool
    reliability_score: float
    attribution: AttributionResult | None = None
    recovery: RecoveryOutcome | None = None
