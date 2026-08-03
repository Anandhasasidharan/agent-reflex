from pydantic_settings import BaseSettings, SettingsConfigDict

_DEV_SECRETS = ("reflex_dev",)


class Settings(BaseSettings):
    """Runtime configuration, loaded from environment variables
    (AGENT_REFLEX_*), an optional .env file, then defaults.

    Defaults are for local development only. In production the `env` field
    must be "production" and every secret must be provided explicitly —
    Settings.validate_production() fails fast on missing or dev credentials.
    """

    model_config = SettingsConfigDict(
        env_prefix="agent_reflex_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = "development"  # development | production
    db_url: str = "postgresql://reflex:reflex_dev@localhost:5432/agent_reflex"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_pass: str = "reflex_dev"
    otel_service_name: str = "agent_reflex"
    otel_endpoint: str = "http://localhost:4318"
    otel_semconv_opt_in: str = "gen_ai_latest_experimental"
    redaction_enabled: bool = True
    openai_api_key: str = ""
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-flash"
    llm_embedding_base_url: str = ""
    llm_embedding_model: str = "text-embedding-3-small"
    consistency_n_samples: int = 5
    consistency_temperature: float = 0.3
    bandit_epsilon: float = 0.3
    bandit_epsilon_decay: float = 0.99

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    def validate_production(self) -> None:
        """Fail fast with a loud, specific error when running in production
        without real secrets. Never silently fall back to dev defaults."""
        if not self.is_production:
            return
        problems: list[str] = []
        if not (self.llm_api_key or self.openai_api_key):
            problems.append(
                "no LLM API key: set AGENT_REFLEX_LLM_API_KEY or AGENT_REFLEX_OPENAI_API_KEY"
            )
        if any(secret in self.db_url for secret in _DEV_SECRETS):
            problems.append("AGENT_REFLEX_DB_URL still uses dev credentials")
        if self.neo4j_pass in _DEV_SECRETS:
            problems.append("AGENT_REFLEX_NEO4J_PASS still uses dev credentials")
        if not self.otel_endpoint:
            problems.append("AGENT_REFLEX_OTEL_ENDPOINT is empty")
        if problems:
            raise RuntimeError(
                "production configuration is incomplete (set AGENT_REFLEX_ENV=development "
                "only for local work):\n  - " + "\n  - ".join(problems)
            )
