import pytest

from agent_reflex.common.config import Settings


def test_dev_defaults_pass_validation():
    settings = Settings()
    assert not settings.is_production
    settings.validate_production()


@pytest.mark.parametrize(
    "env_overrides,missing",
    [
        ({}, "LLM API key"),
        (
            {"llm_api_key": "sk-x"},
            "AGENT_REFLEX_DB_URL still uses dev credentials",
        ),
        (
            {"llm_api_key": "sk-x", "db_url": "postgresql://reflex:real@db:5432/agent_reflex"},
            "AGENT_REFLEX_NEO4J_PASS still uses dev credentials",
        ),
    ],
)
def test_production_fails_fast_without_secrets(env_overrides, missing, monkeypatch):
    # Hermetic: CI sets AGENT_REFLEX_DB_URL etc., which would mask the
    # dev-credential checks — clear every settings env var first.
    for env_name in (
        "AGENT_REFLEX_DB_URL",
        "AGENT_REFLEX_NEO4J_PASS",
        "AGENT_REFLEX_NEO4J_URI",
        "AGENT_REFLEX_NEO4J_USER",
        "AGENT_REFLEX_OTEL_ENDPOINT",
        "AGENT_REFLEX_OPENAI_API_KEY",
        "AGENT_REFLEX_LLM_API_KEY",
        "AGENT_REFLEX_LLM_BASE_URL",
        "AGENT_REFLEX_LLM_MODEL",
    ):
        monkeypatch.delenv(env_name, raising=False)
    kwargs = {"env": "production", **env_overrides}
    with pytest.raises(RuntimeError) as exc:
        Settings(**kwargs).validate_production()
    assert missing in str(exc.value)


def test_production_passes_with_real_secrets():
    settings = Settings(
        env="production",
        llm_api_key="sk-real",
        db_url="postgresql://reflex:supersecret@db:5432/agent_reflex",
        neo4j_pass="supersecret",
        otel_endpoint="http://otel:4318",
    )
    settings.validate_production()
