
from agent_reflex.storage.models import (
    GraphEdgeRecord,
    RecoveryLogRecord,
    ReliabilityRecord,
    SessionRecord,
    TraceStepRecord,
)


def test_model_tables_exist():
    assert hasattr(SessionRecord, "__tablename__")
    assert SessionRecord.__tablename__ == "sessions"
    assert TraceStepRecord.__tablename__ == "trace_steps"
    assert GraphEdgeRecord.__tablename__ == "graph_edges"
    assert RecoveryLogRecord.__tablename__ == "recovery_logs"
    assert ReliabilityRecord.__tablename__ == "reliability_scores"


def test_session_record_columns():
    cols = [c.name for c in SessionRecord.__table__.columns]
    assert "session_id" in cols
    assert "agent_id" in cols
    assert "failure_type" in cols
    assert "cause_node_id" in cols
    assert "causal_responsibility_score" in cols
    assert "task_description" in cols


def test_trace_step_record_columns():
    cols = [c.name for c in TraceStepRecord.__table__.columns]
    assert "node_id" in cols
    assert "observation" in cols
    assert "thought" in cols
    assert "action" in cols
    assert "result" in cols
    assert "error_flag" in cols


def test_recovery_log_record_columns():
    cols = [c.name for c in RecoveryLogRecord.__table__.columns]
    assert "playbook_name" in cols
    assert "selector" in cols
    assert "success" in cols
    assert "partial" in cols


def test_reliability_record_columns():
    cols = [c.name for c in ReliabilityRecord.__table__.columns]
    assert "agent_id" in cols
    assert "score" in cols
    assert "session_id" in cols


def test_graph_edge_record_columns():
    cols = [c.name for c in GraphEdgeRecord.__table__.columns]
    assert "source_id" in cols
    assert "target_id" in cols
    assert "edge_type" in cols


def test_dashboard_stats_heatmap_endpoint():
    from fastapi.testclient import TestClient

    from agent_reflex.dashboard.api import app
    client = TestClient(app)
    response = client.get("/stats/heatmap")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_dashboard_stats_recovery_breakdown_endpoint():
    from fastapi.testclient import TestClient

    from agent_reflex.dashboard.api import app
    client = TestClient(app)
    response = client.get("/stats/recovery-breakdown")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
