from agent_reflex.common.types import (
    AttributionResult,
    CausalGraphNode,
    MastMode,
    MastPlusLabel,
    StepOTAR,
)


def test_mast_mode_values():
    assert MastMode.TASK_DERAILMENT.value == "task_derailment"
    assert MastMode.INFRA_RATE_LIMIT.value == "infra_rate_limit"
    assert len(MastMode) == 18


def test_mast_plus_label():
    label = MastPlusLabel(mode=MastMode.INFRA_CASCADE_TIMEOUT, confidence=0.95)
    assert label.mode == MastMode.INFRA_CASCADE_TIMEOUT
    assert label.confidence == 0.95


def test_step_otar():
    otar = StepOTAR(observation="obs", thought="think", action="act", result="res")
    assert otar.observation == "obs"
    assert otar.thought == "think"


def test_causal_graph_node():
    otar = StepOTAR("o", "t", "a", "r")
    node = CausalGraphNode(
        node_id="n1", agent_id="agent_a", step_index=1, otar=otar,
        parent_id=None, subtask_id="task1", execution_time_ms=100.0, error_flag=True,
    )
    assert node.node_id == "n1"
    assert node.error_flag is True


def test_attribution_result():
    result = AttributionResult(
        session_id="sess_1",
        failure_type=MastMode.TASK_DERAILMENT,
        cause_node_id="step_7",
        causal_responsibility_score=0.82,
        evidence=["subtask C oracle failed", "correcting step 7 changes outcome"],
    )
    assert result.causal_responsibility_score == 0.82
    assert len(result.evidence) == 2
