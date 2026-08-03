from unittest.mock import patch

from fastapi.testclient import TestClient

from agent_reflex.dashboard.api import app

client = TestClient(app)


def _key() -> dict:
    return {"x-api-key": "test-key"}


def test_ingest_trace_no_engine():
    response = client.post("/traces", headers=_key(), json={
        "session_id": "test_001",
        "graph_json": '{"nodes": [], "edges": []}',
        "task_context": "test",
    })
    assert response.status_code in (200, 503)


def test_consistency_score():
    with patch("agent_reflex.uncertainty.consistency.ConsistencyScorer.score") as mock_score:
        mock_score.return_value = 0.85
        response = client.post("/consistency/score", headers=_key(), json={
            "prompt": "What is 2+2?",
            "is_critical": False,
        })
    assert response.status_code == 200
    data = response.json()
    assert data["consistency_score"] == 0.85
    assert data["should_escalate"] is False


def test_consistency_score_critical_low():
    with patch("agent_reflex.uncertainty.consistency.ConsistencyScorer.score") as mock_score:
        mock_score.return_value = 0.3
        response = client.post("/consistency/score", headers=_key(), json={
            "prompt": "Ambiguous query?",
            "is_critical": True,
        })
    assert response.status_code == 200
    data = response.json()
    assert data["should_escalate"] is True


def test_recovery_stats_empty():
    response = client.get("/recovery/stats", headers=_key())
    assert response.status_code == 200
    data = response.json()
    assert data["total_trials"] == 0


def test_recovery_feedback():
    response = client.post("/recovery/feedback", headers=_key(), json={
        "session_id": "test_001",
        "playbook_name": "re_prompt",
        "success": True,
    })
    assert response.status_code == 200
    assert response.json()["status"] == "recorded"


def test_reliability_empty():
    response = client.get("/agents/test_agent/reliability", headers=_key())
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert data["n_sessions"] == 0


def test_predictive_endpoint():
    response = client.get("/predictive/score", headers=_key())
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    response = client.get("/health", headers=_key())
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "db" in data


def test_get_attribution_unknown_session_404():
    with patch("agent_reflex.dashboard.api._db", None):
        response = client.get("/traces/test_001/attribution", headers=_key())
    assert response.status_code == 404


def test_get_graph_unknown_session_404():
    with patch("agent_reflex.dashboard.api._db", None):
        response = client.get("/traces/test_001/graph", headers=_key())
    assert response.status_code == 404


class _FakeDB:
    """Stub PostgresRepository for endpoint-shape tests (the real
    repository is exercised against real Postgres in CI's integration job
    and in the manual verification in the build notes)."""

    def __init__(self, sessions=None):
        self.sessions = sessions or [{
            "session_id": "trace_aaa",
            "agent_id": "otlp_receiver",
            "task_description": "Draft a 2-step plan to ship the release.",
            "failure_type": "infra_timeout",
            "cause_node_id": "step_2",
            "causal_responsibility_score": 0.9,
            "evidence": ["step_2 hit an infra timeout"],
            "created_at": "2026-08-03 10:00:00",
        }]

    def get_session(self, session_id):
        return next((s for s in self.sessions if s["session_id"] == session_id), None)

    def list_sessions(self, limit=50, offset=0, agent_id=None, failure_type=None,
                      since=None, until=None):
        items = self.sessions
        if agent_id:
            items = [s for s in items if s["agent_id"] == agent_id]
        if failure_type:
            items = [s for s in items if s["failure_type"] == failure_type]
        return items, len(items)

    def get_graph(self, session_id):
        from agent_reflex.common.types import CausalGraphNode, StepOTAR
        from agent_reflex.graph.models import CausalGraph
        if not self.get_session(session_id):
            return None
        cg = CausalGraph()
        cg.add_step(CausalGraphNode(
            node_id="step_1", agent_id="planner", step_index=1,
            otar=StepOTAR("in", "think", "chat", "plan"),
            parent_id=None, subtask_id="task_1", execution_time_ms=10.0, error_flag=False,
        ))
        cg.add_step(CausalGraphNode(
            node_id="step_2", agent_id="researcher", step_index=2,
            otar=StepOTAR("in2", "think2", "call_tool", ""),
            parent_id="step_1", subtask_id="task_1", execution_time_ms=20.0, error_flag=True,
        ))
        cg.add_dependency("step_1", "step_2")
        return cg

    def get_agent_reliability_summary(self, window=10):
        return {
            "otlp_receiver": {"scores": [0.0, 1.0, 0.0], "n_sessions": 3},
            "empty_agent": {"scores": [], "n_sessions": 2},
        }


def test_list_traces_empty_db():
    with patch("agent_reflex.dashboard.api._db", None):
        response = client.get("/traces", headers=_key())
    assert response.status_code == 200
    data = response.json()
    assert data["items"] == []
    assert data["total"] == 0


def test_list_traces_with_db():
    with patch("agent_reflex.dashboard.api._db", _FakeDB()):
        response = client.get("/traces", headers=_key())
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    item = data["items"][0]
    assert item["session_id"] == "trace_aaa"
    assert item["failure_type"] == "infra_timeout"
    assert item["cause_node_id"] == "step_2"


def test_list_traces_filters_and_pagination():
    db = _FakeDB(sessions=[
        {"session_id": "s1", "agent_id": "a", "task_description": "", "failure_type": "infra_timeout",
         "cause_node_id": "x", "causal_responsibility_score": 0.9, "evidence": [], "created_at": "2026-08-03 10:00:00"},
        {"session_id": "s2", "agent_id": "b", "task_description": "", "failure_type": "spec_ambiguous",
         "cause_node_id": "y", "causal_responsibility_score": 0.8, "evidence": [], "created_at": "2026-08-03 11:00:00"},
    ])
    with patch("agent_reflex.dashboard.api._db", db):
        response = client.get("/traces?agent_id=a", headers=_key())
    assert [i["session_id"] for i in response.json()["items"]] == ["s1"]
    with patch("agent_reflex.dashboard.api._db", db):
        response = client.get("/traces?failure_type=spec_ambiguous&limit=1&offset=1", headers=_key())
    assert response.status_code == 200
    assert response.json()["total"] == 1
    with patch("agent_reflex.dashboard.api._db", db):
        response = client.get("/traces?since=not-a-date", headers=_key())
    assert response.status_code == 422


def test_get_attribution_real_shape():
    with patch("agent_reflex.dashboard.api._db", _FakeDB()):
        response = client.get("/traces/trace_aaa/attribution", headers=_key())
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "trace_aaa"
    att = data["attribution"]
    assert att["failure_type"] == "infra_timeout"
    assert att["cause_node_id"] == "step_2"
    assert att["crs"] == 0.9
    assert att["evidence"] == ["step_2 hit an infra timeout"]


def test_get_attribution_null_when_llm_failed():
    db = _FakeDB(sessions=[{
        "session_id": "trace_llm_fail", "agent_id": "otlp_receiver",
        "task_description": "x", "failure_type": None, "cause_node_id": None,
        "causal_responsibility_score": 0.0, "evidence": [], "created_at": "2026-08-03 10:00:00",
    }])
    with patch("agent_reflex.dashboard.api._db", db):
        response = client.get("/traces/trace_llm_fail/attribution", headers=_key())
    assert response.status_code == 200
    assert response.json()["attribution"] is None


def test_get_graph_real_shape():
    with patch("agent_reflex.dashboard.api._db", _FakeDB()):
        response = client.get("/traces/trace_aaa/graph", headers=_key())
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "trace_aaa"
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1
    root = [n for n in data["nodes"] if n["is_root_cause"]]
    assert root[0]["id"] == "step_2"
    assert root[0]["error"] is True
    # Node detail panel needs the full OTAR content.
    assert "otar" in data["nodes"][0]
    assert data["nodes"][0]["otar"]["observation"] == "in"


def test_agents_status():
    with patch("agent_reflex.dashboard.api._db", _FakeDB()):
        response = client.get("/agents/status", headers=_key())
    assert response.status_code == 200
    data = response.json()
    by_id = {a["agent_id"]: a for a in data}
    assert "otlp_receiver" in by_id
    assert by_id["otlp_receiver"]["n_sessions"] == 3
    assert by_id["otlp_receiver"]["health"] in ("healthy", "degraded", "critical")
    assert by_id["empty_agent"]["score"] is None
    assert by_id["empty_agent"]["health"] == "unknown"


def test_agents_status_empty_db():
    with patch("agent_reflex.dashboard.api._db", None):
        response = client.get("/agents/status", headers=_key())
    assert response.status_code == 200
    assert response.json() == []


def test_eval_results_no_file():
    fake_settings = type("FakeSettings", (), {"eval_results_dir": "/nonexistent"})()
    with patch("agent_reflex.dashboard.api.Settings", return_value=fake_settings):
        with patch("agent_reflex.dashboard.api._db", _FakeDB()):
            response = client.get("/eval/results", headers=_key())
    assert response.status_code == 200
    assert response.json()["available"] is False


def test_eval_results_reads_latest_comparison(tmp_path):
    import json as _json

    (tmp_path / "comparison_run_20260801_000000.json").write_text(_json.dumps({
        "total_scenarios": 22,
        "oracle_method": {"mode_accuracy_pct": 90.9, "step_accuracy_pct": 68.2},
        "naive_baseline": {"mode_accuracy_pct": 90.9, "step_accuracy_pct": 81.8},
    }))
    fake_settings = type("FakeSettings", (), {"eval_results_dir": str(tmp_path)})()
    with patch("agent_reflex.dashboard.api.Settings", return_value=fake_settings):
        with patch("agent_reflex.dashboard.api._db", _FakeDB()):
            response = client.get("/eval/results", headers=_key())
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["oracle"]["step_accuracy_pct"] == 68.2
    assert data["naive_baseline"]["step_accuracy_pct"] == 81.8


def test_reliability_trend_empty():
    response = client.get("/agents/test_agent/reliability/trend/re_prompt", headers=_key())
    assert response.status_code == 200
    data = response.json()
    assert "before_playbook_mean" in data


def test_stats_heatmap_empty():
    response = client.get("/stats/heatmap", headers=_key())
    assert response.status_code == 200
    assert response.json() == []


def test_stats_recovery_breakdown_empty():
    response = client.get("/stats/recovery-breakdown", headers=_key())
    assert response.status_code == 200
    assert response.json() == []


def test_predictive_post():
    response = client.post("/predictive/score", headers=_key(), json={
        "agents": [{"name": "a1", "tools": []}],
        "edges": [],
    })
    assert response.status_code == 200
    data = response.json()
    assert "risk_scores" in data
