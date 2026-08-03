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


def test_get_attribution_placeholder():
    response = client.get("/traces/test_001/attribution", headers=_key())
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test_001"
    assert data["status"] == "pending"


def test_get_graph_placeholder():
    response = client.get("/traces/test_001/graph", headers=_key())
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test_001"
    assert data["graph"] == {}


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
