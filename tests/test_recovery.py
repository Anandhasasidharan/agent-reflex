from agent_reflex.common.types import FailureSignature, MastMode, MastPlusLabel, RecoveryOutcome
from agent_reflex.recovery.bandit import ContextualBanditSelector
from agent_reflex.recovery.playbooks import (
    DEFAULT_PLAYBOOKS,
    PlaybookLibrary,
    StaticRecoverySelector,
)


def test_playbook_library():
    lib = PlaybookLibrary()
    assert len(lib.list_playbooks()) >= 8


def test_playbook_matching():
    lib = PlaybookLibrary()
    match = lib.matching_playbooks(MastMode.INFRA_RATE_LIMIT)
    assert any(p.name == "rate_limit_backoff" for p in match)
    assert any(p.name == "circuit_breaker" for p in match)

    match = lib.matching_playbooks(MastMode.SPEC_AMBIGUOUS)
    assert any(p.name == "re_prompt" for p in match)


def test_playbook_get():
    lib = PlaybookLibrary()
    p = lib.get("re_prompt")
    assert p is not None
    assert p.name == "re_prompt"
    assert len(p.steps) > 0


def test_playbook_get_nonexistent():
    lib = PlaybookLibrary()
    p = lib.get("nonexistent_playbook")
    assert p is None


def test_static_selector():
    lib = PlaybookLibrary()
    selector = StaticRecoverySelector(lib)
    sig = FailureSignature(
        session_id="test_001",
        mast_label=MastPlusLabel(mode=MastMode.SPEC_AMBIGUOUS, confidence=0.9),
        cause_node_id="step_3",
        agent_id="agent_a",
        crs=0.8,
    )
    playbook = selector.select(sig)
    assert playbook is not None
    assert playbook.name == "re_prompt"

    sig2 = FailureSignature(
        session_id="test_002",
        mast_label=MastPlusLabel(mode=MastMode.INFRA_CASCADE_TIMEOUT, confidence=0.9),
        cause_node_id="step_5",
        agent_id="agent_b",
        crs=0.7,
    )
    playbook2 = selector.select(sig2)
    assert playbook2 is not None
    assert playbook2.name == "circuit_breaker"


def test_static_selector_name():
    selector = StaticRecoverySelector()
    assert selector.name == "static"


def test_static_selector_no_match():
    playbooks_without_backtrack = [p for p in DEFAULT_PLAYBOOKS if p.name != "backtrack_to_checkpoint"]
    lib = PlaybookLibrary(playbooks_without_backtrack)
    selector = StaticRecoverySelector(lib)
    sig = FailureSignature(
        session_id="no_match",
        mast_label=MastPlusLabel(mode=MastMode.TASK_HALLUCINATION, confidence=0.9),
        cause_node_id="step_1",
        agent_id="a",
        crs=0.5,
    )
    playbook = selector.select(sig)
    assert playbook is None


def test_bandit_selector():
    lib = PlaybookLibrary()
    bandit = ContextualBanditSelector(lib)

    sig = FailureSignature(
        session_id="bandit_001",
        mast_label=MastPlusLabel(mode=MastMode.INFRA_RATE_LIMIT, confidence=0.9),
        cause_node_id="step_1",
        agent_id="agent_a",
        crs=0.9,
    )

    playbook = bandit.select(sig)
    assert playbook is not None

    outcome = RecoveryOutcome(
        session_id="bandit_001",
        playbook_name=playbook.name,
        success=True,
        recovery_time_ms=500.0,
    )
    bandit.update(outcome)

    stats = bandit.get_stats()
    assert len(stats) > 0


def test_bandit_learning():
    lib = PlaybookLibrary()
    bandit = ContextualBanditSelector(lib)

    for i in range(20):
        sig = FailureSignature(
            session_id=f"learn_{i}",
            mast_label=MastPlusLabel(mode=MastMode.INFRA_RATE_LIMIT, confidence=0.9),
            cause_node_id="step_1",
            agent_id="agent_a",
            crs=0.9,
        )
        playbook = bandit.select(sig)
        outcome = RecoveryOutcome(
            session_id=f"learn_{i}",
            playbook_name=playbook.name if playbook else "unknown",
            success=True,
        )
        bandit.update(outcome)

    stats = bandit.get_stats()
    assert "learn" in str(stats) or len(stats) > 0


def test_bandit_name():
    bandit = ContextualBanditSelector()
    assert bandit.name == "adaptive"


def test_bandit_update_partial_success():
    lib = PlaybookLibrary()
    bandit = ContextualBanditSelector(lib)
    outcome = RecoveryOutcome(session_id="test_001", playbook_name="re_prompt", success=False, partial=True)
    bandit.update(outcome)
    assert bandit._epsilon > 0
