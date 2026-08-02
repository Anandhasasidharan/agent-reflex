import sys
from unittest.mock import MagicMock

from agent_reflex.common.config import Settings
from agent_reflex.common.llm import LLMClient, extract_json, resolve_api_key


def _mock_openai_module(monkeypatch) -> MagicMock:
    fake_openai = MagicMock()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    return fake_openai


def _fake_response(content: str):
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def test_extract_json_clean():
    assert extract_json('{"mode": "spec_ambiguous", "confidence": 0.9}') == {
        "mode": "spec_ambiguous",
        "confidence": 0.9,
    }


def test_extract_json_fenced():
    result = extract_json('```json\n{"correct": true, "reasoning": "ok"}\n```')
    assert result == {"correct": True, "reasoning": "ok"}


def test_extract_json_prose_wrapped():
    content = 'Here is the answer:\n{"outcome_would_change": false}\nHope that helps.'
    assert extract_json(content) == {"outcome_would_change": False}


def test_extract_json_empty_raises():
    try:
        extract_json("")
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty response")


def test_extract_json_no_object_raises():
    try:
        extract_json("no json here at all")
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-JSON response")


def test_settings_deepseek_defaults():
    settings = Settings(_env_file=None)
    assert settings.llm_base_url == "https://api.deepseek.com"
    assert settings.llm_model == "deepseek-v4-flash"
    assert settings.llm_embedding_base_url == ""


def test_resolve_api_key_prefers_deepseek(monkeypatch):
    monkeypatch.delenv("AGENT_REFLEX_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_REFLEX_LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    assert resolve_api_key(Settings(_env_file=None)) == "sk-deepseek"


def test_resolve_api_key_prefers_explicit_setting():
    settings = Settings(_env_file=None, llm_api_key="sk-llm", openai_api_key="sk-legacy")
    assert resolve_api_key(settings) == "sk-llm"


def test_client_wiring(monkeypatch):
    fake_openai = _mock_openai_module(monkeypatch)
    settings = Settings(_env_file=None, llm_api_key="sk-test")
    client = LLMClient(settings)
    assert client._client is None
    c = client.client
    fake_openai.OpenAI.assert_called_once_with(
        api_key="sk-test",
        base_url="https://api.deepseek.com",
        timeout=120.0,
        max_retries=2,
    )
    assert c is not None


def test_chat_returns_content(monkeypatch):
    fake_openai = _mock_openai_module(monkeypatch)
    fake_openai.OpenAI.return_value.chat.completions.create.return_value = _fake_response("hello")
    client = LLMClient(Settings(_env_file=None, llm_api_key="sk-test"))
    result = client.chat(messages=[{"role": "user", "content": "hi"}])
    assert result == "hello"
    created = fake_openai.OpenAI.return_value.chat.completions.create
    created.assert_called_once()
    kwargs = created.call_args.kwargs
    assert kwargs["model"] == "deepseek-v4-flash"


def test_chat_json_parses(monkeypatch):
    fake_openai = _mock_openai_module(monkeypatch)
    fake_openai.OpenAI.return_value.chat.completions.create.return_value = _fake_response(
        '{"mode": "task_hallucination"}'
    )
    client = LLMClient(Settings(_env_file=None, llm_api_key="sk-test"))
    result = client.chat_json(messages=[{"role": "user", "content": "hi"}])
    assert result == {"mode": "task_hallucination"}


def test_chat_json_retries_without_json_mode(monkeypatch):
    fake_openai = _mock_openai_module(monkeypatch)
    create = fake_openai.OpenAI.return_value.chat.completions.create
    create.side_effect = [
        RuntimeError("json mode not supported"),
        _fake_response('{"correct": true}'),
    ]
    client = LLMClient(Settings(_env_file=None, llm_api_key="sk-test"))
    result = client.chat_json(messages=[{"role": "user", "content": "hi"}])
    assert result == {"correct": True}
    assert create.call_count == 2
    first_kwargs = create.call_args_list[0].kwargs
    second_kwargs = create.call_args_list[1].kwargs
    assert "response_format" in first_kwargs
    assert "response_format" not in second_kwargs


def test_chat_json_retries_on_malformed_json(monkeypatch):
    fake_openai = _mock_openai_module(monkeypatch)
    create = fake_openai.OpenAI.return_value.chat.completions.create
    create.side_effect = [
        _fake_response('{"mode": "spec_contradictory", "confide'),  # truncated
        _fake_response('{"mode": "spec_contradictory"}'),
    ]
    client = LLMClient(Settings(_env_file=None, llm_api_key="sk-test"))
    result = client.chat_json(messages=[{"role": "user", "content": "hi"}])
    assert result == {"mode": "spec_contradictory"}
    assert create.call_count == 2


def test_embed_raises_without_provider():
    client = LLMClient(Settings(_env_file=None, llm_api_key="sk-test"))
    try:
        client.embed("hello")
    except RuntimeError as e:
        assert "embedding" in str(e)
        return
    raise AssertionError("expected RuntimeError when no embedding provider configured")


def test_embed_uses_embedding_settings(monkeypatch):
    fake_openai = _mock_openai_module(monkeypatch)
    fake_openai.OpenAI.return_value.embeddings.create.return_value.data = [MagicMock(embedding=[0.1, 0.2])]
    settings = Settings(
        _env_file=None,
        llm_api_key="sk-test",
        llm_embedding_base_url="https://embeddings.example.com",
        llm_embedding_model="my-embedder",
    )
    client = LLMClient(settings)
    result = client.embed("hello")
    assert result == [0.1, 0.2]
    fake_openai.OpenAI.assert_called_with(
        api_key="sk-test",
        base_url="https://embeddings.example.com",
    )
    fake_openai.OpenAI.return_value.embeddings.create.assert_called_once_with(
        model="my-embedder",
        input="hello",
    )
