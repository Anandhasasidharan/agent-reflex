from agent_reflex.common.types import RecoveryOutcome
from agent_reflex.reliability.scorer import ReliabilityScorer


def test_empty_scorer():
    scorer = ReliabilityScorer()
    result = scorer.current_score_with_trend("nonexistent_agent")
    assert result["score"] == 0.0
    assert result["n_sessions"] == 0


def test_single_success():
    scorer = ReliabilityScorer()
    outcome = RecoveryOutcome(session_id="s1", playbook_name="re_prompt", success=True)
    scorer.record_from_outcome("agent_a", "s1", "do stuff", outcome)
    result = scorer.current_score_with_trend("agent_a")
    assert result["score"] > 0.5
    assert result["n_sessions"] == 1


def test_multiple_sessions_mixed():
    scorer = ReliabilityScorer()
    for i in range(10):
        success = i % 3 != 0
        outcome = RecoveryOutcome(session_id=f"s{i}", playbook_name="re_prompt", success=success)
        scorer.record_from_outcome("agent_b", f"s{i}", f"task {i}", outcome)
    result = scorer.current_score_with_trend("agent_b")
    assert result["n_sessions"] == 10
    assert 0.3 < result["score"] < 1.0


def test_trend_improvement():
    scorer = ReliabilityScorer()
    for i in range(5):
        outcome = RecoveryOutcome(session_id=f"s{i}", playbook_name="re_prompt", success=False)
        scorer.record_from_outcome("agent_c", f"s{i}", f"task {i}", outcome)
    for i in range(5, 10):
        outcome = RecoveryOutcome(session_id=f"s{i}", playbook_name="re_prompt", success=True)
        scorer.record_from_outcome("agent_c", f"s{i}", f"task {i}", outcome)
    result = scorer.current_score_with_trend("agent_c")
    assert result["trend_pct"] > 0


def test_reliability_trend_before_after_playbook():
    scorer = ReliabilityScorer()
    for i in range(5):
        outcome = RecoveryOutcome(session_id=f"s{i}", playbook_name="re_prompt", success=False)
        scorer.record_from_outcome("agent_d", f"s{i}", f"task {i}", outcome)
    for i in range(5, 10):
        outcome = RecoveryOutcome(session_id=f"s{i}", playbook_name="circuit_breaker", success=True)
        scorer.record_from_outcome("agent_d", f"s{i}", f"task {i}", outcome)
    for i in range(10, 12):
        outcome = RecoveryOutcome(session_id=f"s{i}", playbook_name="re_prompt", success=True)
        scorer.record_from_outcome("agent_d", f"s{i}", f"task {i}", outcome)
    trend = scorer.reliability_trend("agent_d", "circuit_breaker")
    assert trend["n_before"] == 5
    assert trend["n_after"] == 7


def test_current_score_with_sessions():
    scorer = ReliabilityScorer()
    for i in range(3):
        outcome = RecoveryOutcome(session_id=f"s{i}", playbook_name="re_prompt", success=True)
        scorer.record_from_outcome("agent_h", f"s{i}", f"task {i}", outcome)
    score = scorer.current_score("agent_h")
    assert score > 0.0


def test_partial_success():
    scorer = ReliabilityScorer()
    outcome = RecoveryOutcome(session_id="s1", playbook_name="re_prompt", success=False, partial=True)
    scorer.record_from_outcome("agent_e", "s1", "partial task", outcome)
    result = scorer.current_score_with_trend("agent_e")
    assert result["score"] < 1.0
    assert result["n_sessions"] == 1


def test_reliability_trend_no_playbook():
    scorer = ReliabilityScorer()
    for i in range(3):
        outcome = RecoveryOutcome(session_id=f"s{i}", playbook_name="re_prompt", success=True)
        scorer.record_from_outcome("agent_f", f"s{i}", f"task {i}", outcome)
    trend = scorer.reliability_trend("agent_f", "circuit_breaker")
    assert trend["n_before"] == 3
    assert trend["n_after"] == 0


def test_reliability_trend_after_only():
    scorer = ReliabilityScorer()
    for i in range(3):
        outcome = RecoveryOutcome(session_id=f"s{i}", playbook_name="circuit_breaker", success=True)
        scorer.record_from_outcome("agent_g", f"s{i}", f"task {i}", outcome)
    trend = scorer.reliability_trend("agent_g", "circuit_breaker")
    assert trend["n_before"] == 0
    assert trend["n_after"] == 3


def test_current_score_no_sessions():
    scorer = ReliabilityScorer()
    score = scorer.current_score("nonexistent")
    assert score == 0.0


def test_record_without_outcome():
    scorer = ReliabilityScorer()
    scorer.record_from_outcome("agent_h", "s1", "task with no outcome")
    result = scorer.current_score_with_trend("agent_h")
    assert result["score"] == 0.0
    assert result["n_sessions"] == 1
