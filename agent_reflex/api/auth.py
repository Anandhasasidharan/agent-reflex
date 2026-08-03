"""API-key authentication for AgentReflex.

Keys are generated with secrets.token_urlsafe, stored in Postgres as
SHA-256 hashes (never plaintext), carry a scope (read | write), and can be
revoked individually. Admin management happens via CLI:

    python -m agent_reflex.api.auth create --name deploy --scope write
    python -m agent_reflex.api.auth revoke <key_id>
    python -m agent_reflex.api.auth list
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException, Request, status
from sqlalchemy import Column, DateTime, Integer, String

from agent_reflex.storage.models import Base

API_KEY_HEADER = "x-api-key"
BEARER_PREFIX = "bearer "

VALID_SCOPES = {"read", "write"}


class ApiKeyRecord(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    key_hash = Column(String(64), unique=True, nullable=False, index=True)
    scope = Column(String(20), nullable=False, default="write")
    created_at = Column(DateTime, default=datetime.utcnow)
    revoked_at = Column(DateTime, nullable=True)


def hash_api_key(key: str) -> str:
    """SHA-256 of the raw key — what is stored at rest."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def generate_api_key() -> str:
    """Generate a new API key. The plaintext is shown exactly once at
    creation time; only its hash is ever persisted."""
    return secrets.token_urlsafe(32)


def _extract_key(request: Request) -> str:
    """Pull the API key from the Authorization bearer header or x-api-key."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith(BEARER_PREFIX):
        return auth[len(BEARER_PREFIX):].strip()
    header_key = request.headers.get(API_KEY_HEADER, "")
    if header_key:
        return header_key.strip()
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="missing API key: send Authorization: Bearer <key> or x-api-key header",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _lookup_key(db: Any, key_hash: str) -> ApiKeyRecord | None:
    found = (
        db.query(ApiKeyRecord)
        .filter(ApiKeyRecord.key_hash == key_hash, ApiKeyRecord.revoked_at.is_(None))
        .first()
    )
    return found if isinstance(found, ApiKeyRecord) else None


class ApiKeyAuth:
    """FastAPI dependency that authenticates an API key with a scope.

    Usage:
        @app.post("/traces", dependencies=[Depends(api_key_auth.require("write"))])
    """

    def require(self, scope: str = "read") -> Any:
        if scope not in VALID_SCOPES:
            raise ValueError(f"invalid scope {scope!r}; valid: {sorted(VALID_SCOPES)}")

        def dependency(request: Request) -> dict[str, Any]:
            from agent_reflex.storage.repository import PostgresRepository

            key = _extract_key(request)
            repository = PostgresRepository()
            try:
                with repository._session() as db:
                    record = _lookup_key(db, hash_api_key(key))
                    if record is None:
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="invalid or revoked API key",
                        )
                    if record.scope != scope and record.scope != "admin":
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"this key has scope {record.scope!r}, requires {scope!r}",
                        )
                    request.state.api_key_id = record.id
                    request.state.api_key_name = record.name
                    return {"key_id": record.id, "name": record.name, "scope": record.scope}
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"key store unavailable: {type(exc).__name__}",
                )

        return dependency


api_key_auth = ApiKeyAuth()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _connect() -> Any:
    from agent_reflex.storage.repository import PostgresRepository

    repository = PostgresRepository()
    repository.init_db()
    return repository


def cli_create(name: str, scope: str = "write") -> None:
    if scope not in VALID_SCOPES:
        raise SystemExit(f"invalid scope {scope!r}; valid: {sorted(VALID_SCOPES)}")
    repository = _connect()
    raw = generate_api_key()
    with repository._session() as db:
        db.add(ApiKeyRecord(name=name, key_hash=hash_api_key(raw), scope=scope))
        db.commit()
    print(f"created API key {name!r} (scope={scope})")
    print(f"KEY: {raw}")
    print("store this value now — only its SHA-256 hash is persisted")


def cli_revoke(key_id: str) -> None:
    repository = _connect()
    with repository._session() as db:
        record = db.query(ApiKeyRecord).filter(ApiKeyRecord.id == int(key_id)).first()
        if record is None:
            raise SystemExit(f"no key with id {key_id}")
        if record.revoked_at is not None:
            raise SystemExit(f"key {key_id} already revoked")
        record_name = record.name
        record.revoked_at = datetime.now(UTC)
        db.commit()
    print(f"revoked key id={key_id} ({record_name})")


def cli_list() -> None:
    repository = _connect()
    with repository._session() as db:
        for record in db.query(ApiKeyRecord).order_by(ApiKeyRecord.id).all():
            state = "revoked" if record.revoked_at else "active"
            print(f"id={record.id} name={record.name!r} scope={record.scope} {state}")


def main() -> None:
    import sys

    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(1)
    command, rest = args[0], args[1:]
    if command == "create":
        kwargs: dict[str, str] = {}
        positional: list[str] = []
        for arg in rest:
            if arg.startswith("--name="):
                kwargs["name"] = arg.split("=", 1)[1]
            elif arg.startswith("--scope="):
                kwargs["scope"] = arg.split("=", 1)[1]
            else:
                positional.append(arg)
        if positional:
            kwargs["name"] = positional[0]
        cli_create(kwargs.get("name", "default"), kwargs.get("scope", "write"))
    elif command == "revoke" and rest:
        cli_revoke(rest[0])
    elif command == "list":
        cli_list()
    else:
        raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    main()
