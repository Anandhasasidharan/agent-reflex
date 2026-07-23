import io
import sys

from agent_reflex.eval.ablation import run_ablation
from agent_reflex.eval.runner import SYNTHETIC_SCENARIOS, build_graph_from_scenario, main, run_eval


def test_synthetic_scenarios_defined():
    assert len(SYNTHETIC_SCENARIOS) >= 6


def test_build_graph_from_scenario():
    scenario = SYNTHETIC_SCENARIOS[0]
    graph = build_graph_from_scenario(scenario)
    assert len(graph.get_all_nodes()) == len(scenario["steps"])
    assert graph.get_node("step_1") is not None
    assert graph.get_node("step_2") is not None


def test_all_scenarios_build():
    for scenario in SYNTHETIC_SCENARIOS:
        graph = build_graph_from_scenario(scenario)
        assert len(graph.get_all_nodes()) == len(scenario["steps"])
        assert scenario["true_cause"] in [n.node_id for n in graph.get_all_nodes()]


def test_confusion_matrix_structure():
    results = [
        {"true_mode": "spec_ambiguous", "predicted_mode": "spec_ambiguous"},
        {"true_mode": "task_hallucination", "predicted_mode": "task_hallucination"},
        {"true_mode": "spec_ambiguous", "predicted_mode": "task_hallucination"},
    ]
    from agent_reflex.eval.runner import print_confusion_matrix
    captured = io.StringIO()
    sys.stdout = captured
    print_confusion_matrix(results)
    sys.stdout = sys.__stdout__
    output = captured.getvalue()
    assert "spec_ambiguous" in output


def test_scenario_graph_has_expected_structure():
    for scenario in SYNTHETIC_SCENARIOS:
        graph = build_graph_from_scenario(scenario)
        nodes = graph.get_all_nodes()
        assert any(n.error_flag for n in nodes), f"{scenario['name']} should have error nodes"
        modes_found = set(n.agent_id for n in nodes)
        assert len(modes_found) >= 1


def test_run_eval_no_api_key(monkeypatch):
    monkeypatch.delenv("AGENT_REFLEX_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = run_eval()
    assert "error" in result
    assert result["error"] == "no_api_key"


def test_main_no_api_key(monkeypatch):
    monkeypatch.delenv("AGENT_REFLEX_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured = io.StringIO()
    sys.stdout = captured
    main()
    sys.stdout = sys.__stdout__
    output = captured.getvalue()
    assert "No OpenAI API key" in output


def test_ablation_main_no_api_key(monkeypatch):
    monkeypatch.delenv("AGENT_REFLEX_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = run_ablation()
    assert "error" in result
    assert result["error"] == "no_api_key"
