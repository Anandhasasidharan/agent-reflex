from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

from agent_reflex.api.auth import api_key_auth
from agent_reflex.attribution.engine import AttributionEngine
from agent_reflex.common.types import RecoveryOutcome
from agent_reflex.graph.models import CausalGraph
from agent_reflex.recovery.bandit import ContextualBanditSelector
from agent_reflex.recovery.playbooks import PlaybookLibrary, StaticRecoverySelector
from agent_reflex.reliability.scorer import ReliabilityScorer
from agent_reflex.storage.repository import PostgresRepository
from agent_reflex.uncertainty.consistency import UncertaintyEscalationController

logger = logging.getLogger("agent_reflex.api")

_attribution_engine: AttributionEngine | None = None
_playbook_library = PlaybookLibrary()
_static_selector = StaticRecoverySelector(_playbook_library)
_bandit_selector = ContextualBanditSelector(_playbook_library)
_escalation_controller = UncertaintyEscalationController()
_reliability_scorer = ReliabilityScorer()
_db: PostgresRepository | None = None

_recovery_log: list[dict[str, Any]] = []
_reliability_history: dict[str, list[float]] = {}


def _task_context_for(graph: CausalGraph) -> str:
    """Best-effort task description from the first span's observation."""
    nodes = sorted(graph.get_all_nodes(), key=lambda n: n.step_index)
    if not nodes:
        return ""
    observation = nodes[0].otar.observation
    return (observation or nodes[0].otar.thought or "")[:1000]


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    from agent_reflex.common.config import Settings

    settings = Settings()
    settings.validate_production()
    global _attribution_engine, _db
    _attribution_engine = AttributionEngine()
    try:
        _db = PostgresRepository()
        _db.init_db()
    except Exception:
        _db = None
    yield


app = FastAPI(title="AgentReflex", version="0.1.0", lifespan=lifespan)

# Per-key sliding-window rate limit (in-process; a multi-worker deployment
# should back this with Redis — noted in README).
_MAX_WRITES_PER_MINUTE = 60
_rate_windows: dict[str, list[float]] = defaultdict(list)


async def _rate_limit_write(request: Request) -> None:
    """Rate limit keyed by the authenticated API key (set by the auth dep)."""
    key_id = str(getattr(request.state, "api_key_id", "unauthenticated"))
    now = time.monotonic()
    recent = [t for t in _rate_windows[key_id] if now - t < 60.0]
    if len(recent) >= _MAX_WRITES_PER_MINUTE:
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    recent.append(now)
    _rate_windows[key_id] = recent


@app.get("/health")
async def health() -> dict[str, str]:
    db_status = "connected" if _db is not None else "unavailable"
    return {"status": "ok", "db": db_status}


@app.get("/ready")
async def ready() -> dict[str, Any]:
    """Readiness: the service can actually serve (DB reachable)."""
    from agent_reflex.common.config import Settings

    if _db is None:
        return {"status": "not_ready", "reason": "database_unavailable"}
    try:
        from sqlalchemy import text

        with _db._session() as db:
            db.execute(text("SELECT 1"))
    except Exception as exc:
        return {"status": "not_ready", "reason": f"database_unreachable: {type(exc).__name__}"}
    settings = Settings()
    from agent_reflex.common.llm import resolve_api_key

    has_llm = bool(resolve_api_key(settings))
    return {"status": "ready", "db": "connected", "llm_configured": has_llm}


@app.post(
    "/v1/traces",
    dependencies=[Depends(api_key_auth.require("write")), Depends(_rate_limit_write)],
)
async def otlp_receive(request: Request) -> dict[str, Any]:
    """OTLP/HTTP receiver (ExportTraceServiceRequest).

    Parses real OTel spans forwarded by a collector — protobuf (the actual
    OTLP wire format) or JSON — reconstructs one CausalGraph per trace, runs
    real attribution, and persists to Postgres.
    """
    from agent_reflex.graph.span_ingest import parse_otlp_json, parse_otlp_protobuf

    if _attribution_engine is None:
        raise HTTPException(status_code=503, detail="Attribution engine not initialized")

    raw = await request.body()
    content_type = request.headers.get("content-type", "")
    try:
        if "protobuf" in content_type:
            traces = parse_otlp_protobuf(raw)
        else:
            payload = json.loads(raw)
            traces = parse_otlp_json(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid OTLP payload: {exc}")

    if not traces:
        return {"partialSuccess": {"rejectedSpans": 0, "errorMessage": ""}}

    ingested = 0
    for trace_id, graph in traces:
        if not graph.get_all_nodes():
            continue
        result = None
        try:
            result = _attribution_engine.attribute(
                session_id=trace_id,
                graph=graph,
                task_context=_task_context_for(graph),
            )
        except Exception as exc:
            # LLM attribution is best-effort: a provider outage must not
            # reject the trace — persist it, mark the failure_type unknown.
            logger.warning(
                "attribution skipped for trace %s", trace_id,
                extra={"error": f"{type(exc).__name__}: {exc}"},
            )
            request.app.state.last_attribution_error = f"{type(exc).__name__}: {exc}"
        if _db is not None:
            _db.save_session(
                session_id=trace_id,
                agent_id="otlp_receiver",
                task_description=_task_context_for(graph),
                graph=graph,
                attribution=result,
            )
        ingested += 1

    return {"partialSuccess": {"rejectedSpans": 0, "errorMessage": ""}, "ingested_traces": ingested}


class TraceInput(BaseModel):
    session_id: str
    graph_json: str
    task_context: str = ""
    agent_id: str = "default_agent"


class ConsistencyInput(BaseModel):
    prompt: str
    is_critical: bool = False


class RecoveryFeedback(BaseModel):
    session_id: str
    agent_id: str = "default_agent"
    task_description: str = ""
    playbook_name: str
    success: bool
    partial: bool = False


class TopologyInput(BaseModel):
    agents: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    shared_contexts: int = 0


@app.post(
    "/traces",
    dependencies=[Depends(api_key_auth.require("write")), Depends(_rate_limit_write)],
)
async def ingest_trace(input_data: TraceInput) -> dict[str, Any]:
    graph = CausalGraph.from_json(input_data.graph_json)
    if _attribution_engine is None:
        raise HTTPException(status_code=503, detail="Attribution engine not initialized")

    result = _attribution_engine.attribute(
        session_id=input_data.session_id,
        graph=graph,
        task_context=input_data.task_context,
    )

    if _db is not None:
        _db.save_session(
            session_id=input_data.session_id,
            agent_id=input_data.agent_id,
            task_description=input_data.task_context,
            graph=graph,
            attribution=result,
        )

    return {
        "session_id": input_data.session_id,
        "attribution": {
            "failure_type": result.failure_type.value,
            "cause_node_id": result.cause_node_id,
            "crs": result.causal_responsibility_score,
            "evidence": result.evidence,
        },
    }


@app.get("/traces/{session_id}/attribution", dependencies=[Depends(api_key_auth.require("read"))])
async def get_attribution(session_id: str) -> dict[str, Any]:
    return {"session_id": session_id, "status": "pending"}


@app.get("/traces/{session_id}/graph", dependencies=[Depends(api_key_auth.require("read"))])
async def get_graph(session_id: str) -> dict[str, Any]:
    return {"session_id": session_id, "graph": {}}


@app.get("/agents/{agent_id}/reliability", dependencies=[Depends(api_key_auth.require("read"))])
async def get_reliability(agent_id: str) -> dict[str, Any]:
    return _reliability_scorer.current_score_with_trend(agent_id)


@app.get("/agents/{agent_id}/reliability/trend/{playbook_name}", dependencies=[Depends(api_key_auth.require("read"))])
async def get_reliability_trend(agent_id: str, playbook_name: str) -> dict[str, Any]:
    return _reliability_scorer.reliability_trend(agent_id, playbook_name)


@app.get("/recovery/stats", dependencies=[Depends(api_key_auth.require("read"))])
async def get_recovery_stats() -> dict[str, Any]:
    if not _recovery_log:
        return {"adaptive_success_rate": 0.0, "static_success_rate": 0.0, "total_trials": 0}

    adaptive = [r for r in _recovery_log if r.get("selector") == "adaptive"]
    static = [r for r in _recovery_log if r.get("selector") == "static"]

    adaptive_rate = sum(1 for r in adaptive if r["success"]) / len(adaptive) if adaptive else 0.0
    static_rate = sum(1 for r in static if r["success"]) / len(static) if static else 0.0

    return {
        "adaptive_success_rate": round(adaptive_rate, 3),
        "static_success_rate": round(static_rate, 3),
        "adaptive_trials": len(adaptive),
        "static_trials": len(static),
        "total_trials": len(_recovery_log),
    }


@app.post("/consistency/score", dependencies=[Depends(api_key_auth.require("write")), Depends(_rate_limit_write)])
async def score_consistency(input_data: ConsistencyInput) -> dict[str, Any]:
    should_escalate, score = _escalation_controller.should_escalate(
        prompt=input_data.prompt,
        is_critical=input_data.is_critical,
    )
    return {
        "consistency_score": round(score, 4),
        "should_escalate": should_escalate,
        "threshold": _escalation_controller._threshold,
    }


@app.post("/recovery/feedback", dependencies=[Depends(api_key_auth.require("write")), Depends(_rate_limit_write)])
async def recovery_feedback(feedback: RecoveryFeedback) -> dict[str, str]:
    outcome = RecoveryOutcome(
        session_id=feedback.session_id,
        playbook_name=feedback.playbook_name,
        success=feedback.success,
        partial=feedback.partial,
    )
    _bandit_selector.update(outcome)
    _reliability_scorer.record_from_outcome(
        agent_id=feedback.agent_id,
        session_id=feedback.session_id,
        task_description=feedback.task_description,
        outcome=outcome,
    )

    if _db is not None:
        _db.save_recovery_outcome(outcome, selector="adaptive", agent_id=feedback.agent_id)
        _db.save_reliability_score(
            feedback.agent_id,
            feedback.session_id,
            _reliability_scorer.current_score(feedback.agent_id),
        )

    _recovery_log.append({
        "session_id": feedback.session_id,
        "playbook": feedback.playbook_name,
        "success": feedback.success,
        "partial": feedback.partial,
        "selector": "adaptive",
    })
    return {"status": "recorded"}


@app.get("/stats/heatmap", dependencies=[Depends(api_key_auth.require("read"))])
async def stats_heatmap() -> list[dict[str, Any]]:
    if _db is None:
        return []
    return _db.get_heatmap()


@app.get("/stats/recovery-breakdown", dependencies=[Depends(api_key_auth.require("read"))])
async def stats_recovery_breakdown() -> list[dict[str, Any]]:
    if _db is None:
        return []
    return _db.get_recovery_breakdown()


@app.post("/predictive/score", dependencies=[Depends(api_key_auth.require("write")), Depends(_rate_limit_write)])
async def predict_topology(topology: TopologyInput) -> dict[str, Any]:
    from agent_reflex.predictive.topology_scorer import PredictiveTopologyScorer
    scorer = PredictiveTopologyScorer()
    scores = scorer.score_architecture(topology.model_dump())
    return {"risk_scores": scores}


@app.get("/predictive/score", dependencies=[Depends(api_key_auth.require("read"))])
async def predict_topology_get() -> dict[str, Any]:
    return {"message": "Send topology via POST /predictive/score with JSON body"}
