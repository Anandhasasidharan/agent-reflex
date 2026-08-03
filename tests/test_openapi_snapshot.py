"""The frontend's TypeScript types are generated from this checked-in
OpenAPI snapshot. If the backend schema drifts, this test fails until the
snapshot is intentionally regenerated and the frontend types are updated:

    python -m agent_reflex.dashboard.export_openapi
"""

import json
from pathlib import Path

SNAPSHOT = Path("frontend/src/types/openapi.json")


def test_openapi_schema_matches_frontend_snapshot():
    from fastapi.testclient import TestClient

    from agent_reflex.dashboard.api import app

    client = TestClient(app)
    live = client.get("/openapi.json").json()

    if not SNAPSHOT.exists():
        raise AssertionError(
            f"missing openapi snapshot at {SNAPSHOT}; run "
            "python -m agent_reflex.dashboard.export_openapi"
        )

    snapshot = json.loads(SNAPSHOT.read_text())
    assert live == snapshot, (
        "backend OpenAPI schema drifted from frontend/types/openapi.json; "
        "run python -m agent_reflex.dashboard.export_openapi and update "
        "frontend/src/lib/types.ts to match"
    )
