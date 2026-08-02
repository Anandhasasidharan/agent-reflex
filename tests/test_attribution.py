from unittest.mock import MagicMock

from agent_reflex.attribution.engine import AttributionEngine
from agent_reflex.common.types import CausalGraphNode, StepOTAR
from agent_reflex.graph.models import CausalGraph


def _make_node(node_id: str, agent_id: str = "a", step_index: int = 1, error: bool = True) -> CausalGraphNode:
    return CausalGraphNode(
        node_id=node_id, agent_id=agent_id, step_index=step_index,
        otar=StepOTAR("input", "thought", "action", "result"),
        parent_id=None, subtask_id="t1", execution_time_ms=10.0, error_flag=error,
    )


def test_attribution_engine_init():
    engine = AttributionEngine()
    assert engine is not None
    assert engine._llm._client is None


def test_attribution_client_lazy(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    engine = AttributionEngine()
    assert engine._llm._client is None
    client = engine._llm.client
    assert client is not None
    assert engine._llm._client is not None


def test_oracle_backtracking_no_failures():
    graph = CausalGraph()
    graph.add_step(CausalGraphNode(
        node_id="s1", agent_id="a", step_index=1,
        otar=StepOTAR("input", "thinking", "search", "correct result"),
        parent_id=None, subtask_id="t1", execution_time_ms=10.0, error_flag=False,
    ))

    reversed_nodes = sorted(graph.get_all_nodes(), key=lambda n: n.step_index, reverse=True)
    subtasks = graph.decompose_into_subtasks()

    assert len(reversed_nodes) == 1
    assert "t1" in subtasks


def test_subtask_summarize():
    engine = AttributionEngine()
    nodes = [
        CausalGraphNode(
            node_id="s1", agent_id="a", step_index=1,
            otar=StepOTAR("input1", "think1", "search1", "result1"),
            parent_id=None, subtask_id="t1", execution_time_ms=10.0, error_flag=False,
        ),
        CausalGraphNode(
            node_id="s2", agent_id="a", step_index=2,
            otar=StepOTAR("input2", "think2", "search2", "result2"),
            parent_id="s1", subtask_id="t1", execution_time_ms=10.0, error_flag=False,
        ),
    ]
    summary = engine._summarize_subtask(nodes)
    assert summary["observation"] == "input1"
    assert summary["result"] == "result2"
    assert "think1" in summary["thought"]


def _run_counterfactual(engine, llm_response: dict, monkeypatch) -> float:
    monkeypatch.setattr(engine, "_synthesize_corrected_output", lambda _node, _ctx: "corrected output")
    monkeypatch.setattr(engine, "_call_llm_json", lambda _prompt: llm_response)

    graph = MagicMock(spec=CausalGraph)
    graph.decompose_into_subtasks.return_value = {}
    candidate = _make_node("s1")
    return engine._counterfactual_screening(graph, candidate, "some task")


def test_counterfactual_continuous_score(monkeypatch):
    engine = AttributionEngine()
    crs = _run_counterfactual(engine, {"confidence_pct": 85}, monkeypatch)
    assert crs == 0.85


def test_counterfactual_low_confidence(monkeypatch):
    engine = AttributionEngine()
    crs = _run_counterfactual(engine, {"confidence_pct": 15}, monkeypatch)
    assert crs == 0.15


def test_counterfactual_mid_confidence(monkeypatch):
    engine = AttributionEngine()
    crs = _run_counterfactual(engine, {"confidence_pct": 50}, monkeypatch)
    assert crs == 0.50


def test_counterfactual_fallback_would_change(monkeypatch):
    engine = AttributionEngine()
    crs = _run_counterfactual(engine, {"outcome_would_change": True}, monkeypatch)
    assert crs == 0.95


def test_counterfactual_fallback_no_change(monkeypatch):
    engine = AttributionEngine()
    crs = _run_counterfactual(engine, {"outcome_would_change": False}, monkeypatch)
    assert crs == 0.15


def test_counterfactual_clamps_out_of_range(monkeypatch):
    engine = AttributionEngine()
    crs_high = _run_counterfactual(engine, {"confidence_pct": 150}, monkeypatch)
    assert crs_high == 1.0
    crs_low = _run_counterfactual(engine, {"confidence_pct": -10}, monkeypatch)
    assert crs_low == 0.0
