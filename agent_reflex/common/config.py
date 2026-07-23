from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "agent_reflex_"}

    db_url: str = "postgresql://reflex:reflex_dev@localhost:5432/agent_reflex"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_pass: str = "reflex_dev"
    otel_service_name: str = "agent_reflex"
    otel_endpoint: str = "http://localhost:4318"
    otel_semconv_opt_in: str = "gen_ai_latest_experimental"
    redaction_enabled: bool = True
    openai_api_key: str = ""
    consistency_n_samples: int = 5
    consistency_temperature: float = 0.3
    bandit_epsilon: float = 0.3
    bandit_epsilon_decay: float = 0.99
