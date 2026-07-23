import json
import os
import tempfile

from agent_reflex.eval.benchmark import (
    EvalExample,
    EvalResult,
    build_synthetic_eval_set,
    load_eval_set,
)
from agent_reflex.eval.benchmark_adapter import (
    build_graph_from_trace,
    load_benchmark_traces,
    run_benchmark,
)
from agent_reflex.eval.cross_benchmark import run_cross_benchmark
from agent_reflex.eval.traceelephant import load_traceelephant_traces, run_traceelephant
from agent_reflex.eval.whowhen import load_whowhen_traces, run_whowhen


class MockEngine:
    def attribute(self, session_id, graph, task_context=""):
        from agent_reflex.common.types import AttributionResult, MastMode
        return AttributionResult(
            session_id=session_id,
            failure_type=MastMode.TASK_HALLUCINATION,
            cause_node_id="step_1",
            causal_responsibility_score=0.85,
        )

    def __bool__(self):
        return True


def test_build_graph_from_trace():
    trace = {
        "name": "test_trace",
        "true_mode": "task_hallucination",
        "true_cause": "step_1",
        "steps": [
            {"id": "step_1", "agent": "a", "input": "in", "thought": "think",
             "action": "act", "result": "wrong", "error": True, "parent": None,
             "subtask": "t1"},
        ],
    }
    graph = build_graph_from_trace(trace)
    nodes = graph.get_all_nodes()
    assert len(nodes) == 1
    assert nodes[0].node_id == "step_1"


def test_run_benchmark_no_traces():
    result = run_benchmark([], None, label="empty")
    assert result["total"] == 0
    assert result["mode_accuracy_pct"] == 0.0


def test_run_benchmark_with_traces():
    traces = [
        {"name": "t1", "true_mode": "task_hallucination", "true_cause": "step_1",
         "task_context": "test", "steps": [{"id": "step_1", "input": "in", "thought": "think",
                                             "action": "act", "result": "wrong", "error": True}]},
    ]
    result = run_benchmark(traces, MockEngine(), label="mock")
    assert result["total"] == 1
    assert result["mode_accuracy_pct"] == 100.0
    assert result["step_accuracy_pct"] == 100.0
    assert len(result["details"]) == 1


def test_run_benchmark_with_trace_id_fallback():
    traces = [
        {"trace_id": "t1", "true_mode": "task_hallucination", "true_cause": "step_1",
         "steps": [{"id": "step_1", "input": "in", "thought": "think",
                     "action": "act", "result": "wrong", "error": True}]},
    ]
    result = run_benchmark(traces, MockEngine(), label="mock")
    assert result["details"][0]["name"] == "t1"


def test_run_benchmark_without_true_fields():
    traces = [
        {"name": "t1", "steps": [{"id": "step_1", "input": "in", "thought": "think",
                                   "action": "act", "result": "out", "error": True}]},
    ]
    result = run_benchmark(traces, MockEngine(), label="mock")
    assert result["details"][0]["true_mode"] == ""


def test_build_graph_from_trace_no_steps():
    trace = {"name": "empty", "steps": []}
    graph = build_graph_from_trace(trace)
    assert len(graph.get_all_nodes()) == 0


def test_build_graph_from_trace_no_id_underscore():
    trace = {
        "steps": [
            {"id": "step1", "input": "in", "thought": "think",
             "action": "act", "result": "out", "error": True},
        ],
    }
    graph = build_graph_from_trace(trace)
    assert graph.get_all_nodes()[0].step_index == 0


def test_build_graph_from_trace_fallback_fields():
    trace = {
        "steps": [
            {"id": "s1", "agent": "a", "observation": "in", "thought": "think",
             "action": "act", "result": "out", "error_flag": True, "parent": None},
        ],
    }
    graph = build_graph_from_trace(trace)
    node = graph.get_all_nodes()[0]
    assert node.otar.observation == "in"
    assert node.error_flag is True


def test_load_benchmark_traces_delegates():
    def loader(path):
        return [{"name": path}]
    traces = load_benchmark_traces("/fake/path", loader)
    assert len(traces) == 1
    assert traces[0]["name"] == "/fake/path"


def test_whowhen_loader_no_file():
    traces = load_whowhen_traces("/nonexistent/path/traces.json")
    assert traces == []


def test_whowhen_loader_with_temp_file():
    data = [
        {"name": "t1", "true_mode": "spec_ambiguous", "true_cause": "step_1", "steps": []},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        fpath = f.name
    try:
        traces = load_whowhen_traces(fpath)
        assert len(traces) == 1
        assert traces[0]["name"] == "t1"
    finally:
        os.unlink(fpath)


def test_whowhen_loader_dict_with_traces():
    data = {"traces": [{"name": "t1", "true_mode": "spec_ambiguous", "steps": []}]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        fpath = f.name
    try:
        traces = load_whowhen_traces(fpath)
        assert len(traces) == 1
        assert traces[0]["name"] == "t1"
    finally:
        os.unlink(fpath)


def test_whowhen_loader_unknown_dict():
    data = {"other": "value"}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        fpath = f.name
    try:
        traces = load_whowhen_traces(fpath)
        assert traces == []
    finally:
        os.unlink(fpath)


def test_traceelephant_loader_no_file():
    traces = load_traceelephant_traces("/nonexistent/path")
    assert traces == []


def test_traceelephant_loader_with_temp_file():
    data = {"trace_id": "te1", "true_mode": "infra_rate_limit", "true_cause": "step_1",
            "steps": [{"id": "step_1", "agent": "a", "input": "in", "thought": "think",
                       "action": "act", "result": "err", "error": True}]}
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "trace_001.json")
        with open(fpath, "w") as f:
            json.dump(data, f)
        traces = load_traceelephant_traces(tmpdir)
        assert len(traces) >= 1


def test_traceelephant_loader_combined_as_list():
    data = [{"trace_id": "te1", "steps": [{"id": "s1", "input": "in", "thought": "think",
                                            "action": "act", "result": "out", "error": True}]}]
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "traces.json")
        with open(fpath, "w") as f:
            json.dump(data, f)
        traces = load_traceelephant_traces(tmpdir)
        assert len(traces) == 1


def test_traceelephant_loader_combined_as_dict_with_traces():
    data = {"traces": [{"trace_id": "te1", "steps": [{"id": "s1", "input": "in", "thought": "think",
                                                       "action": "act", "result": "out", "error": True}]}]}
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "traces.json")
        with open(fpath, "w") as f:
            json.dump(data, f)
        traces = load_traceelephant_traces(tmpdir)
        assert len(traces) == 1


def test_traceelephant_loader_individual_json_without_steps():
    data = {"some_key": "value"}
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "other.json")
        with open(fpath, "w") as f:
            json.dump(data, f)
        traces = load_traceelephant_traces(tmpdir)
        assert len(traces) == 0




def test_whowhen_runner_no_data(monkeypatch):
    monkeypatch.delenv("AGENT_REFLEX_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = run_whowhen(None)
    assert "error" in result
    assert result["error"] == "data_not_found"


def test_traceelephant_runner_no_data(monkeypatch):
    monkeypatch.delenv("AGENT_REFLEX_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = run_traceelephant(None)
    assert "error" in result
    assert result["error"] == "data_not_found"


def test_cross_benchmark_no_api_key(monkeypatch):
    monkeypatch.delenv("AGENT_REFLEX_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = run_cross_benchmark()
    assert "error" in result
    assert result["error"] == "no_api_key"



def test_eval_example_defaults():
    ex1 = EvalExample(trace_id="t1", graph_json="{}", true_failure_mode="spec_ambiguous", true_cause_node_id="s1")
    assert ex1.task_context == ""
    ex2 = EvalExample(trace_id="t2", graph_json="{}", true_failure_mode="spec_ambiguous", true_cause_node_id="s1", task_context="ctx")
    assert ex2.task_context == "ctx"


def test_eval_result_fields():
    r = EvalResult(mode_correct=True, step_correct=False, predicted_crs=0.8,
                   true_mode="spec_ambiguous", predicted_mode="task_hallucination",
                   predicted_cause="step_2")
    assert r.mode_correct is True
    assert r.step_correct is False
    assert r.predicted_crs == 0.8
    assert r.true_mode == "spec_ambiguous"
    assert r.predicted_mode == "task_hallucination"
    assert r.predicted_cause == "step_2"


def test_build_synthetic_eval_set_count():
    examples = build_synthetic_eval_set()
    assert len(examples) == 3


def test_build_synthetic_eval_set_valid_graph_json():
    examples = build_synthetic_eval_set()
    for ex in examples:
        import json as _json
        data = _json.loads(ex.graph_json)
        assert "nodes" in data


def test_build_synthetic_eval_set_contents():
    examples = build_synthetic_eval_set()
    trace_ids = [ex.trace_id for ex in examples]
    assert "synth_coord_misalignment" in trace_ids
    assert "synth_hallucination" in trace_ids
    assert "synth_rate_limit" in trace_ids


def test_load_eval_set():
    data = [
        {"trace_id": "t1", "graph_json": '{"nodes":[]}', "true_failure_mode": "spec_ambiguous", "true_cause_node_id": "s1"},
        {"trace_id": "t2", "graph_json": '{"nodes":[]}', "true_failure_mode": "task_hallucination", "true_cause_node_id": "s2", "task_context": "ctx"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        fpath = f.name
    try:
        examples = load_eval_set(fpath)
        assert len(examples) == 2
        assert examples[0].trace_id == "t1"
        assert examples[1].task_context == "ctx"
    finally:
        os.unlink(fpath)
