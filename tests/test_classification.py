from agent_reflex.classification.mast_plus import (
    MAST_EXAMPLES,
    MAST_FEW_SHOT_PROMPT,
    MastPlusClassifier,
)


def test_mast_examples_coverage():
    modes_in_examples = set(ex["label"] for ex in MAST_EXAMPLES)
    assert "spec_ambiguous" in modes_in_examples
    assert "infra_rate_limit" in modes_in_examples
    assert "infra_cascade_timeout" in modes_in_examples
    assert "task_hallucination" in modes_in_examples


def test_prompt_template_renders():
    examples_str = "\n\n".join(
        f"Trace: {ex['trace']}\nLabel: {ex['label']}" for ex in MAST_EXAMPLES
    )
    prompt = MAST_FEW_SHOT_PROMPT.format(
        examples=examples_str,
        trace="Trace: Agent failed due to rate limit",
    )
    assert "spec_ambiguous" in prompt
    assert "infra_rate_limit" in prompt
    assert "Trace:" in prompt
    assert "Return only a JSON object" in prompt


def test_classifier_lazy_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    classifier = MastPlusClassifier()
    assert classifier._client is None
    classifier.client
    assert classifier._client is not None
