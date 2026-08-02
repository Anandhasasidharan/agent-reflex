from agent_reflex.eval.ablation import run_ablation
from agent_reflex.instrumentation.redaction import (
    SENSITIVE_PATTERNS,
    RedactingSpanProcessor,
    _redact_value,
)


def test_redact_api_key():
    result = _redact_value("api_key = sk-abc123def456ghi789jkl012")
    assert "[REDACTED]" in result
    assert "sk-abc123" not in result


def test_redact_token():
    result = _redact_value("token: ghp_abcdefghijklmnopqrstuvwxyz0123456789abcd")
    assert "[REDACTED]" in result


def test_redact_no_false_positive():
    result = _redact_value("plain text with no secrets here")
    assert result == "plain text with no secrets here"


def test_redact_nested_dict():
    result = _redact_value({"key": "sk-abcdef1234567890", "nested": {"secret": "token: ghp_abcdefghijklmnopqrstuvwxyz0123456789abcd"}})
    assert result["key"] == "[REDACTED]"
    assert "[REDACTED]" in result["nested"]["secret"]


def test_redact_list():
    result = _redact_value(["sk-abcdef1234567890", "ok"])
    assert result[0] == "[REDACTED]"
    assert result[1] == "ok"


def test_redact_non_string_value():
    result = _redact_value(42)
    assert result == 42
    result = _redact_value(3.14)
    assert result == 3.14
    result = _redact_value(None)
    assert result is None


def test_ablation_runner_structure(monkeypatch):
    monkeypatch.delenv("AGENT_REFLEX_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_REFLEX_LLM_API_KEY", raising=False)
    result = run_ablation()
    assert "error" in result
    assert result["error"] == "no_api_key"


def test_sensitive_patterns_compiled():
    for pat in SENSITIVE_PATTERNS:
        assert pat.pattern is not None


class MockSpan:
    def __init__(self, attributes=None, events=None):
        self._span = MockSpanContext(attributes, events)


class MockSpanContext:
    def __init__(self, attributes=None, events=None):
        self.attributes = attributes or {}
        self.events = events or []

    def set_attribute(self, key, value):
        self.attributes[key] = value


class MockEvent:
    def __init__(self, attributes=None):
        self.attributes = attributes or {}


def test_redacting_processor_on_end_with_attrs():
    processor = RedactingSpanProcessor()
    span = MockSpan(attributes={"secret": "sk-abc123def456ghi789jkl012", "safe": "hello"})
    processor.on_end(span)
    assert span._span.attributes["secret"] == "[REDACTED]"
    assert span._span.attributes["safe"] == "hello"


def test_redacting_processor_on_end_with_events():
    processor = RedactingSpanProcessor()
    event = MockEvent(attributes={"key": "ghp_token123456789"})
    span = MockSpan(attributes={"safe": "ok"}, events=[event])
    processor.on_end(span)
    assert event.attributes["key"] == "[REDACTED]"


def test_redacting_processor_on_end_no_span_ctx():
    processor = RedactingSpanProcessor()

    class SpanNoCtx:
        pass

    processor.on_end(SpanNoCtx())


def test_redacting_processor_on_start():
    processor = RedactingSpanProcessor()
    processor.on_start(None)


def test_redacting_processor_shutdown():
    processor = RedactingSpanProcessor()
    processor.shutdown()


def test_redacting_processor_force_flush():
    processor = RedactingSpanProcessor()
    assert processor.force_flush() is True
