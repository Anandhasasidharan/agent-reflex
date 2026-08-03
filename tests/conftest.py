import os

import pytest

os.environ.setdefault("AGENT_REFLEX_OPENAI_API_KEY", "sk-test-dummy-placeholder")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-placeholder")


@pytest.fixture(autouse=True)
def _fake_api_key_store(monkeypatch):
    """Accept any API key in tests; the key store is a real Postgres
    dependency exercised separately (tests/test_auth.py)."""
    from agent_reflex.api import auth as auth_mod
    from agent_reflex.api.auth import ApiKeyRecord

    def fake_lookup(db, key_hash):
        return ApiKeyRecord(id=999, name="test", key_hash=key_hash, scope="admin")

    monkeypatch.setattr(auth_mod, "_lookup_key", fake_lookup)

