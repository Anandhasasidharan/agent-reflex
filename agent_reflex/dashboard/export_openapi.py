"""Export the FastAPI OpenAPI schema to the frontend's checked-in snapshot.

The frontend's TypeScript types are derived from
frontend/src/types/openapi.json. A pytest (test_openapi_snapshot) fails if
the live schema drifts from that snapshot, so a backend change forces an
intentional, reviewed update:

    python -m agent_reflex.dashboard.export_openapi

Then update frontend/src/lib/types.ts to match.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    from agent_reflex.dashboard.api import app

    snapshot = json.dumps(app.openapi(), indent=2) + "\n"
    target = Path("frontend/src/types/openapi.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(snapshot)
    print(f"wrote openapi snapshot ({len(snapshot)} bytes) to {target}")


if __name__ == "__main__":
    main()
