import asyncio
from collections import defaultdict

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from agent_reflex.api import auth as auth_mod
from agent_reflex.api.auth import (
    ApiKeyAuth,
    ApiKeyRecord,
    api_key_auth,
    generate_api_key,
    hash_api_key,
)


def _request(headers: dict[str, str]) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": ("127.0.0.1", 50000),
        "server": ("test", 80),
        "scheme": "http",
    }
    return Request(scope)


@pytest.fixture(autouse=True)
def _no_db(monkeypatch):
    def fake_lookup(db, key_hash):
        table = {
            hash_api_key("write-key"): (1, "ci", "write"),
            hash_api_key("read-key"): (2, "readonly", "read"),
        }
        row = table.get(key_hash)
        if row is None:
            return None
        return ApiKeyRecord(id=row[0], name=row[1], key_hash=key_hash, scope=row[2])

    monkeypatch.setattr(auth_mod, "_lookup_key", fake_lookup)


def test_hash_is_sha256_and_deterministic():
    raw = generate_api_key()
    assert len(raw) >= 40
    h = hash_api_key(raw)
    assert len(h) == 64
    assert h == hash_api_key(raw)
    assert raw not in h


def test_missing_key_is_401():
    with pytest.raises(HTTPException) as exc:
        api_key_auth.require("read")(_request({}))
    assert exc.value.status_code == 401


def test_invalid_key_is_401():
    with pytest.raises(HTTPException) as exc:
        api_key_auth.require("read")(_request({"authorization": "Bearer nope"}))
    assert exc.value.status_code == 401


def test_valid_key_with_header_returns_identity():
    identity = api_key_auth.require("write")(_request({"x-api-key": "write-key"}))
    assert identity["name"] == "ci"
    assert identity["scope"] == "write"
    assert identity["key_id"] == 1


def test_valid_key_with_bearer_token():
    identity = api_key_auth.require("read")(_request({"authorization": "Bearer read-key"}))
    assert identity["scope"] == "read"


def test_scope_mismatch_is_403():
    with pytest.raises(HTTPException) as exc:
        api_key_auth.require("write")(_request({"x-api-key": "read-key"}))
    assert exc.value.status_code == 403


def test_invalid_scope_raises_value_error():
    with pytest.raises(ValueError):
        ApiKeyAuth().require("sudo")


def test_rate_limit_blocks_after_60_per_key(monkeypatch):
    import agent_reflex.dashboard.api as api

    monkeypatch.setattr(api, "_rate_windows", defaultdict(list))
    req = _request({"x-api-key": "k"})
    req.state.api_key_id = 42
    for _ in range(60):
        asyncio.run(api._rate_limit_write(req))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(api._rate_limit_write(req))
    assert exc.value.status_code == 429


def test_rate_limit_is_per_key(monkeypatch):
    import agent_reflex.dashboard.api as api

    monkeypatch.setattr(api, "_rate_windows", defaultdict(list))
    key_a = _request({"x-api-key": "a"})
    key_a.state.api_key_id = 1
    key_b = _request({"x-api-key": "b"})
    key_b.state.api_key_id = 2
    for _ in range(60):
        asyncio.run(api._rate_limit_write(key_a))
    with pytest.raises(HTTPException):
        asyncio.run(api._rate_limit_write(key_a))
    asyncio.run(api._rate_limit_write(key_b))
    assert api._rate_windows["2"]  # key B is unaffected by key A's exhaustion
