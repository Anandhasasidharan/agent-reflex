from unittest.mock import MagicMock

from agent_reflex.uncertainty.consistency import ConsistencyScorer, UncertaintyEscalationController


def test_scorer_init():
    scorer = ConsistencyScorer()
    assert scorer._client is None
    assert scorer._settings.consistency_n_samples == 5


def test_measure_agreement_same():
    scorer = ConsistencyScorer()
    score = scorer._measure_agreement(["Paris", "Paris", "Paris"])
    assert score >= 0.9


def test_measure_agreement_single():
    scorer = ConsistencyScorer()
    score = scorer._measure_agreement(["Paris"])
    assert score == 1.0


def test_measure_agreement_empty():
    scorer = ConsistencyScorer()
    score = scorer._measure_agreement([])
    assert score == 1.0


def test_escalation_controller_defaults():
    controller = UncertaintyEscalationController(threshold=0.7)
    assert controller._threshold == 0.7


def test_escalation_controller_threshold_logic():
    scorer = MagicMock()
    scorer.score.return_value = 0.3
    controller = UncertaintyEscalationController(scorer=scorer, threshold=0.7)
    should, score = controller.should_escalate("test prompt", is_critical=True)
    assert should is True
    assert score == 0.3


def test_escalation_controller_no_escalate():
    scorer = MagicMock()
    scorer.score.return_value = 0.85
    controller = UncertaintyEscalationController(scorer=scorer, threshold=0.7)
    should, score = controller.should_escalate("test prompt", is_critical=False)
    assert should is False
    assert score == 0.85


def test_calibrate_threshold():
    scorer = MagicMock()
    scorer.score.side_effect = [0.2, 0.3, 0.8, 0.9, 0.1, 0.4]
    controller = UncertaintyEscalationController(scorer=scorer, threshold=0.7)
    labeled = [
        ("bad prompt 1", True),
        ("bad prompt 2", True),
        ("good prompt 1", False),
        ("good prompt 2", False),
        ("bad prompt 3", True),
        ("bad prompt 4", True),
    ]
    threshold = controller.calibrate_threshold(labeled)
    assert 0.1 <= threshold <= 0.95
