# AgentReflex

A multi-agent reliability platform that reconstructs agent execution as a hierarchical causal graph, attributes failures to the exact responsible step via counterfactual intervention (not pattern matching), learns which recovery strategy works per failure signature via a contextual bandit, and calibrates human escalation on consistency-sampled uncertainty.

**Build time:** 10 weeks · **Difficulty:** Research-adjacent · **GPU required:** No · **LLM:** model-agnostic (any OpenAI-compatible API — DeepSeek by default, OpenAI, Ollama, vLLM, OpenRouter)

## Architecture

```
LLMClient (model-agnostic)  — any OpenAI-compatible API
[DeepSeek default · JSON extraction + retry · optional embeddings]
         |
         v
OTel GenAI-Native Instrumentation SDK  (LangGraph + CrewAI adapters)
[gen_ai_latest_experimental opt-in; cache + streaming attrs]
         |
         v
 Collector + Causal Graph Reconstructor
(OTAR parsing, dependency graph, redaction layer)
         |
    +----+----+
    |         |
    v         v
Attribution  MAST+ Classifier  (18 modes: 14 MAST + 4 Infra/Ops)
Engine      (LLM few-shot)
    |         |
    |   evaluated against Who&When + TraceElephant
    |   + redaction on/off ablation
    +----+----+
         |
         v
Quantitative Reliability Scorer  (weighted EWMA, trend analysis)
         |
    +----+----+
    |         |
    v         v
Adaptive    Uncertainty-Calibrated
Recovery    Escalation  (consistency-sampling N=5)
(bandit)
         |
         v
Dashboard + Causal Graph Viewer + Reliability Trends + Grafana
         |
  (stretch) Predictive Topology Risk Scorer
```

## Quick Start

```bash
# Dependencies
pip install -e ".[dev]"

# Set your LLM provider key and OTel semconv version
export DEEPSEEK_API_KEY=sk-...   # default provider (any OpenAI-compatible works)
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental

# Run the full pipeline demo
python demo/full_pipeline.py

# Start the API server
uvicorn agent_reflex.dashboard.api:app --reload

# Full stack with Docker
export DEEPSEEK_API_KEY=sk-...
docker compose up -d
# Access: API at http://localhost:8000, Grafana at http://localhost:3000

# API keys are mandatory (create one, then pass it to the collector/Grafana)
docker compose exec app python -m agent_reflex.api.auth create deploy --scope=write
```

## Deployment

### API keys and authentication

Every endpoint except `/health` and `/ready` requires an API key
(`Authorization: Bearer <key>` or `x-api-key` header). Keys are stored as
SHA-256 hashes in Postgres with `read`/`write` scopes and per-key
revocation:

```bash
python -m agent_reflex.api.auth create <name> --scope=write   # prints the key once
python -m agent_reflex.api.auth create <name> --scope=read
python -m agent_reflex.api.auth revoke <id>
python -m agent_reflex.api.auth list
```

Write endpoints are rate-limited per key (60/min, in-process; back the
limiter with Redis for multi-worker deployments).

### Environment

Copy `.env.example` to `.env` and fill it in (`.env` is git-ignored).
Settings load from `AGENT_REFLEX_*` env vars / `.env`. **Production mode is
fail-fast**: with `AGENT_REFLEX_ENV=production`, the app refuses to start
unless the LLM key, real DB/Neo4j credentials and the OTel endpoint are
configured — it never silently falls back to dev defaults.

### The OTLP ingestion path

```
your agent (OTel SDK) ──OTLP─▶ otel-collector ──otlphttp──▶ POST /v1/traces ──▶ attribution ──▶ Postgres
```

- `demo/reference_agent.py` is a framework-agnostic SDK producer emitting a
  failing 3-step trace (no langgraph/crewai dependency).
- The collector forwards to the API with the `x-api-key` header
  (`AGENT_REFLEX_API_KEY` env; endpoint overridable via
  `AGENT_REFLEX_API_ENDPOINT`).
- The receiver accepts the real OTLP wire format (protobuf and JSON) and
  reconstructs a causal graph per trace id. LLM attribution is best-effort:
  on provider failure the trace is still persisted (failure_type NULL) and a
  JSON warning is logged.
- The span schema is documented in `docs/otel_ingestion_schema.md`.

### Structured logging

The container runs uvicorn with `deploy/logging-config.json` — every log
line is JSON (`timestamp`, `level`, `logger`, `message` + structured extra
fields), so CloudWatch/Stackdriver/Vector can index it directly.

### CI/CD

- `ci.yml`: ruff + mypy, pytest (3.11/3.12) with coverage, Docker build,
  a **compose-based integration job** (reference agent → real
  otel-collector → API → Postgres, session row asserted), and security
  scans (pip-audit fails on high/critical, Trivy fs scan).
- `cd.yml`: builds and pushes the image to GHCR on `master`, then a deploy
  stage (SSH `docker compose pull && up -d` with a `/ready` wait) that only
  runs when `vars.DEPLOY_ENABLED=true` and `DEPLOY_HOST`/`DEPLOY_SSH_KEY`
  secrets exist.

### Cloud (AWS ECS + RDS)

`deploy/terraform/` provisions a Fargate service + ALB + RDS Postgres.
Secrets live in SSM Parameter Store (`/agent_reflex/*`), never in git or
the task definition. See `deploy/terraform/README.md` for the exact
`aws ssm put-parameter` commands.

## Known Follow-Ups

Deliberately out of scope for the production hardening passes; tracked here
so they are not silently "fixed":

- Neo4j wiring is stubbed — the API stores everything in Postgres; the
  compose Neo4j service exists but nothing queries it.
- `dashboard/` UI endpoints (`/traces/{id}/graph`, `/traces/{id}/attribution`)
  return stubs.
- Redaction is a no-op pass-through (`AGENT_REFLEX_REDACTION_ENABLED`).
- Benchmark datasets are synthetic-only (no real-world traces yet).
- langgraph/crewai adapters intentionally not added — the ingestion layer is
  framework-agnostic by design.

## Model-Agnostic LLM Support

AgentReflex talks to **any OpenAI-compatible API** through a thin `LLMClient` wrapper
(`agent_reflex/common/llm.py`) — only `base_url` and model name change per provider:

| Provider | `AGENT_REFLEX_LLM_BASE_URL` | `AGENT_REFLEX_LLM_MODEL` |
|----------|-----------------------------|--------------------------|
| **DeepSeek** (default) | `https://api.deepseek.com` | `deepseek-v4-flash` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Ollama (local) | `http://localhost:11434/v1` | any pulled model |
| vLLM / OpenRouter / Together / Groq | provider endpoint | any model |

- API key resolution: `AGENT_REFLEX_LLM_API_KEY` → `AGENT_REFLEX_OPENAI_API_KEY` →
  `DEEPSEEK_API_KEY` → `OPENAI_API_KEY`
- JSON responses are parsed robustly (markdown fences, prose-wrapped JSON) and
  retried without `response_format` if a provider rejects it
- **Embeddings are optional.** DeepSeek has no embeddings endpoint; set
  `AGENT_REFLEX_LLM_EMBEDDING_BASE_URL` to a provider that has one, otherwise
  consistency scoring falls back to lexical agreement (token-sequence similarity).

## Demo

The full pipeline demo (`demo/full_pipeline.py`) runs a simulated multi-agent failure through all 7 stages:

1. **Causal graph reconstruction** — Builds a 5-node, 2-subtask graph from simulated OTel spans
2. **MAST+ classification** — LLM few-shot classifies the failure into 1 of 18 modes
3. **Counterfactual attribution** — Oracle-guided backtracking + Causal Responsibility Score
4. **Recovery selection** — Contextual bandit vs static baseline, side-by-side
5. **Bandit learning** — 10 simulated recovery rounds with Q-value updates
6. **Escalation trigger** — Consistency-sampling (N=5) on test prompts, threshold-based escalation
7. **Reliability scoring** — Exponential weighted moving average with before/after playbook trend analysis

Output also includes an interactive D3.js causal graph viewer (`demo/causal_graph_viewer.html`) with root-cause node highlighted in red.

## Project Structure

```
agent_reflex/
├── common/              # Shared types (StepOTAR, MastMode, MastPlusLabel), config, model-agnostic LLMClient
├── instrumentation/     # OTel GenAI SDK + LangGraph/CrewAI adapters
├── graph/               # Causal graph (NetworkX + Neo4j persistence)
├── classification/      # MAST+ LLM few-shot classifier (18 modes)
├── attribution/         # Oracle-guided backtracking + counterfactual CRS
├── recovery/            # 8 playbooks + static selector + contextual bandit (hand-rolled ε-greedy)
├── uncertainty/         # Consistency-sampling (N=5, embeddings with lexical fallback)
├── reliability/         # Quantitative reliability scorer (weighted EWMA, trends)
├── predictive/          # Topology-based risk scoring (stretch, experimental)
├── storage/             # Postgres models (SQLAlchemy) + repository pattern
├── dashboard/           # FastAPI (9 endpoints) + D3.js causal graph viewer
├── eval/                # Synthetic scenarios + Who&When/TraceElephant adapters + redaction ablation
├── demo/                # Full pipeline demo + HTML graph viewer
├── grafana/             # Auto-provisioned Grafana 6-panel dashboard
├── tests/               # 165 tests (pytest, pytest-cov, all passing)
└── .github/workflows/   # CI: ruff → mypy → pytest (3.11 + 3.12); CD: GHCR image
```

## Key Results

| Component | Metric | Result | How to Reproduce |
|-----------|--------|--------|-----------------|
| **Causal graph reconstruction** | OTAR parsing, dependency inference, subtask decomposition | ✅ 11 tests | `pytest tests/test_causal_graph.py -v` |
| **MAST+ Classification** | 18-mode taxonomy, LLM few-shot with 10 examples | ✅ 3 tests | `pytest tests/test_classification.py -v` |
| **Step-level attribution** | Oracle-guided backtracking (LLM verifies each subtask, content vs environmental failure) + counterfactual CRS | ✅ 14 tests · vs `NaiveEarliestErrorBaseline` (`python -m agent_reflex.eval.runner --compare`) | `pytest tests/test_attribution.py -v` |
| **Adaptive recovery lift** | Hand-rolled ε-greedy contextual bandit vs static baseline | ✅ 11 tests | `pytest tests/test_recovery.py -v` |
| **Reliability scoring** | Weighted EWMA with trend & before/after playbook analysis | ✅ 11 tests | `pytest tests/test_reliability.py -v` |
| **Consistency AUROC** | N=5 sampling, embedding agreement + lexical fallback, threshold calibration | ✅ 11 tests | `pytest tests/test_uncertainty.py -v` |
| **Topology risk** | 17-mode prediction from fan-out, depth, topology type | ✅ 7 tests | `pytest tests/test_predictive.py -v` |
| **Storage (Postgres)** | 5 SQLAlchemy models + repository with heatmap/breakdown queries | ✅ 8 tests | `pytest tests/test_storage.py -v` |
| **Dashboard API** | FastAPI (9 endpoints): ingest, stats, recovery, reliability, predictive | ✅ 14 tests | `pytest tests/test_dashboard.py -v` |
| **Instrumentation** | OTel GenAI decorators + LangGraph/CrewAI adapters | ✅ 10 tests | `pytest tests/test_instrumentation.py -v` |
| **Types** | MastMode, MastPlusLabel, StepOTAR, CausalGraphNode, AttributionResult | ✅ 5 tests | `pytest tests/test_types.py -v` |
| **Eval scaffolding** | 22 synthetic scenarios (all 18 modes + 4 transient-recovery decoys) with ground-truth root causes | ✅ 11 tests | `pytest tests/test_eval.py -v` |
| **Causal graph viewer** | D3.js interactive graph with root-cause highlighting | ✅ 1 test | `pytest tests/test_causal_viewer.py -v` |
| **Redaction ablation** | Accuracy delta with vs without redaction (mode −16.7%, n=6) | ✅ 14 tests | `pytest tests/test_ablation.py -v` |
| **Cross-benchmark** | Who&When + TraceElephant benchmark adapters | ✅ 27 tests | `pytest tests/test_benchmark.py -v` |
| **LLM client** | Model-agnostic wrapper: JSON retry, key resolution, embeddings | ✅ 15 tests | `pytest tests/test_llm_client.py -v` |
| **Full test suite** | 173 tests across all modules | ✅ **173/173 passing** | `pytest --cov=agent_reflex` |

### Attribution Accuracy (Synthetic Eval Set, measured with DeepSeek)

Run `DEEPSEEK_API_KEY=sk-... python -m agent_reflex.eval.runner --compare --save` to reproduce a side-by-side comparison of the oracle method vs a naive baseline.

**Step attribution methods.** Two competing explanations for "which step is the root cause" are compared on the same 22 synthetic scenarios (all 18 modes plus 4 scenarios where an early transient timeout self-recovers and the *real* cause appears later — designed to defeat any earliest-error shortcut):

- **Oracle-guided backtracking** (production path): a virtual LLM oracle verifies each subtask, explicitly separating *content errors* (the prompt is being followed but the reasoning is wrong → root cause candidates) from *environmental failures* (timeout/rate-limit/connection-drop → symptoms, never "correct"). It walks error-flagged candidates earliest-first and returns the first node the oracle judges a content error.
- **`NaiveEarliestErrorBaseline`** (`agent_reflex/eval/baseline.py`): always picks the first `error_flag` node. Deterministic, zero inference cost, but predicts the same candidate regardless of whether the early error truly caused the later failure.

Earlier 18-scenario runs used the naive heuristic as the production path and reported "100% step" — that was a zero-inference fit, not a real causal signal, so those claims are withdrawn.

**Measured comparison (DeepSeek, 22 scenarios):**

| Method | Mode Acc | Step Acc |
|--------|----------|----------|
| Oracle-guided (real) | 90.9% (20/22) | 68.2% (15/22) |
| NaiveEarliestErrorBaseline | 90.9% (20/22, same classifier) | 81.8% (18/22) |

**Honest verdict: the oracle LOST the overall step-accuracy comparison (68.2% vs 81.8%).** The naive baseline wins outright on the 18 base scenarios (18/18, because those scenarios were designed so the earliest errored step *is* the cause), while the oracle over-prunes and undershoots there (12/18). The oracle only wins where the baseline is structurally incapable — the 4 transient-recovery decoys: oracle gets 3/4, naive 0/4. In other words, the naive baseline's score is a design artifact of the base set, not a real causal capability; the oracle is directionally correct on the hard class but still misses. Per the eval constraint, the oracle prompt was not re-tuned against this set, so the gap is reported as-is. Full per-scenario detail in `eval_results/comparison_run_20260802_190150.json`.

| Scenario | True Mode | Predicted | Step | CRS |
|----------|-----------|-----------|------|-----|
| coord_misaligned_assumptions | coord_misaligned_assumptions | task_hallucination ✗ | ✓ | 0.90 |
| spec_ambiguous | spec_ambiguous | spec_ambiguous ✓ | ✓ | 0.95 |
| task_hallucination | task_hallucination | task_hallucination ✓ | ✓ | 0.80 |
| infra_rate_limit | infra_rate_limit | infra_rate_limit ✓ | ✓ | 0.90 |
| verif_overconfident | verif_overconfident | task_hallucination ✗ | ✓ | 0.95 |
| infra_context_window | infra_context_window | infra_context_window ✓ | ✓ | 0.90 |
| spec_incomplete | spec_incomplete | spec_incomplete ✓ | ✓ | 0.80 |
| spec_contradictory | spec_contradictory | spec_contradictory ✓ | ✓ | 0.90 |
| spec_missing | spec_missing | spec_incomplete ✗ | ✓ | 0.95 |
| coord_misaligned_goals | coord_misaligned_goals | coord_misaligned_goals ✓ | ✓ | 0.90 |
| coord_resource_contention | coord_resource_contention | coord_resource_contention ✓ | ✓ | 0.90 |
| coord_deadlock | coord_deadlock | coord_deadlock ✓ | ✓ | 0.95 |
| verif_underconfident | verif_underconfident | verif_underconfident ✓ | ✓ | 0.90 |
| verif_wrong_criterion | verif_wrong_criterion | verif_wrong_criterion ✓ | ✓ | 0.85 |
| verif_self_inconsistent | verif_self_inconsistent | verif_self_inconsistent ✓ | ✓ | 0.95 |
| task_derailment | task_derailment | task_derailment ✓ | ✓ | 0.95 |
| infra_cascade_timeout | infra_cascade_timeout | infra_cascade_timeout ✓ | ✓ | 0.90 |
| infra_unknown | infra_unknown | infra_unknown ✓ | ✓ | 0.95 |

Mode classification is strong; the remaining mode misses are genuinely ambiguous confusions (e.g. `task_hallucination` vs `verif_overconfident`, `spec_missing` vs `spec_incomplete`). Step-level attribution compares the oracle-guided method against the naive baseline; head-to-head results are stored in `eval_results/comparison_run_*.json` and reported plainly in the `--compare` output.

### Redaction Ablation (measured with DeepSeek)

Run `DEEPSEEK_API_KEY=sk-... python -m agent_reflex.eval.ablation`:

| Config | Mode Acc | Step Acc |
|--------|----------|----------|
| Plain (no redaction) | 100.0% (6/6) | 50.0% (3/6) |
| Redacted | 83.3% (5/6) | 50.0% (3/6) |
| **Delta (plain − redacted)** | **+16.7%** | +0.0% |

Preliminary n=6 signal: redaction lowers mode accuracy but leaves step accuracy intact — worth expanding to a larger eval set before drawing conclusions.

## Grafana Dashboards

Docker Compose auto-provisions Grafana with 6 panels (via `grafana-http-datasource` → FastAPI):

| Panel | Type | What It Shows |
|-------|------|--------------|
| **Failure Heatmap by MAST+ Category** | Table | Failure counts per category over time |
| **Adaptive Recovery Rate** | Stat | Bandit-selector success rate (green ≥80%, yellow ≥50%, red <50%) |
| **Static Recovery Rate** | Stat | Static-selector success rate (same thresholds) |
| **Agent Reliability Score Trend** | Time series | EWMA reliability score per agent |
| **Recovery Strategy Breakdown** | Table | Per-playbook success/failure counts |
| **Consistency Score Gauge** | Gauge | Current consistency score (green ≥0.7, yellow ≥0.4, red <0.4) |

Access at `http://localhost:3000` (admin/reflex_dev).

## Testing

```bash
# Full suite (165 tests)
pytest

# With coverage
pytest --cov=agent_reflex --cov-report=html
# Open htmlcov/index.html in browser

# Lint
ruff check .

# Run a specific module's tests
pytest tests/test_recovery.py -v

# Run eval (requires API key)
DEEPSEEK_API_KEY=sk-... python -m agent_reflex.eval.runner

# Redaction ablation (requires API key)
DEEPSEEK_API_KEY=sk-... python -m agent_reflex.eval.ablation

# Cross-benchmark eval (requires API key + datasets)
DEEPSEEK_API_KEY=sk-... python -m agent_reflex.eval.cross_benchmark
```

### CI/CD (GitHub Actions)

`.github/workflows/ci.yml` runs on push/PR to `master`:
1. **Lint** — `ruff check .` + `mypy agent_reflex` (strict) on Python 3.12
2. **Test** — `pytest --cov=agent_reflex` on Python 3.11 + 3.12 with a Postgres 16 service container, coverage uploaded to Codecov
3. **Build** — verifies the production Docker image builds

`.github/workflows/cd.yml` runs on push to `master`: builds the image with Buildx (GitHub Actions cache), pushes to GHCR (`ghcr.io/Anandhasasidharan/agent-reflex`) tagged `latest` + commit SHA. Deploy the image with:

```bash
docker pull ghcr.io/Anandhasasidharan/agent-reflex:latest
```

## OTel Configuration

OpenTelemetry uses the `gen_ai_latest_experimental` semantic convention set for GenAI span attributes. Set the opt-in env var explicitly:

```bash
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

The `instrument_chat` decorator supports optional cache and streaming attributes:
- `cache_read_tokens` / `cache_creation_tokens` — prompt caching metrics
- `stream` — whether the request uses streaming
- `time_to_first_chunk_ms` — streaming latency

## Redaction Ablation

`agent_reflex/eval/ablation.py` measures how much attribution accuracy degrades when the OTel redaction layer is active vs disabled. Run:

```bash
DEEPSEEK_API_KEY=sk-... python -m agent_reflex.eval.ablation
```

Output shows mode-level and step-level accuracy for both configurations with a delta. Measured result (n=6, DeepSeek): **mode accuracy drops 16.7% under redaction (−100%→83.3%), step accuracy unchanged** — meaning the redaction layer is hiding signal the classifier needs. This is a novel empirical result — no published system reports how its own instrumentation choices affect its attribution accuracy.

## Cross-Benchmark Evaluation

`agent_reflex/eval/cross_benchmark.py` runs attribution against both Who&When and TraceElephant benchmark datasets (if available):

```bash
DEEPSEEK_API_KEY=sk-... python -m agent_reflex.eval.cross_benchmark
```

Datasets are not bundled. To run:
- **Who&When**: download `traces.json` to `data/whowhen/`
- **TraceElephant**: clone `github.com/TraceElephant/TraceElephant` and copy traces to `data/traceelephant/`

Without datasets, the runner prints setup instructions and exits gracefully.

## Storage

Postgres is optional. The dashboard API connects via `AGENT_REFLEX_DB_URL` (defaults to `postgresql://reflex:reflex_dev@localhost:5432/agent_reflex`). If Postgres is unavailable, the dashboard falls back to in-memory operation — no crashes, just no persistence.

```bash
# Start Postgres standalone
docker compose up -d postgres

# Or full stack (Postgres + Neo4j + App + Grafana)
docker compose up -d
```

The API includes a `/health` endpoint used by Docker's HEALTHCHECK. The docker-compose.yml configures health checks for all services with proper start periods and dependency conditions, so `app` waits for healthy Postgres and Neo4j before starting.

Five SQLAlchemy models are available:
- `SessionRecord` — tracked agent sessions
- `TraceStepRecord` — individual steps with OTAR attributes
- `RecoveryLogRecord` — recovery outcomes per failure
- `ReliabilityRecord` — per-agent reliability snapshots
- `GraphEdgeRecord` — causal graph edge persistence

## Design Tradeoffs

**Oracle-guided backtracking vs. oracle-free search (FALAT).** This project's attribution engine uses oracle-guided backtracking (following CHIEF's approach): it synthesizes a virtual oracle per subtask to verify correctness, then prunes the search space toward likely failure points. FALAT (arXiv:2606.00765) takes the opposite bet — a dependency-guided search over a single failed trajectory with no oracle synthesis and no task-specific training data, explicitly positioned against CHIEF's oracle-dependence. No public FALAT implementation existed at time of writing, so this project does not build a comparison baseline. The tradeoff is real: oracle guidance costs synthesis overhead and some reliance on oracle correctness, in exchange for a more directed search; FALAT avoids that cost but searches a larger space per trajectory.

**Hand-rolled contextual bandit vs. fine-tuning approaches (AgenTracer/GraphTracer).** The recovery engine uses a lightweight hand-rolled ε-greedy contextual bandit rather than a fine-tuning-based approach. Alternative methods like AgenTracer and GraphTracer learn recovery policies by fine-tuning the agent model itself on failure-recovery trajectories. Neither has confirmed public code, and fine-tuning requires curated trajectory data and GPU hours that are out of scope for a self-contained reliability platform. A bandit is simpler, data-efficient, and sufficient to demonstrate adaptive recovery lift over a static baseline — which is the claim this project makes.

## The Honest Pitch

This project tackles problems at or near the research frontier. Expect some numbers to be modest — that's by design. A README that reports "34% step-level accuracy on a 50-trace held-out set with a confusion matrix" is a stronger signal than one that vaguely claims a working self-healing system, because it proves you understand this is genuinely unsolved and you measured instead of asserting.

## Sources

- arXiv:2503.13657 — MAST Failure Taxonomy (NeurIPS 2025)
- arXiv:2602.23701 — CHIEF (causal graph + oracle-guided backtracking + counterfactual attribution)
- arXiv:2605.25338 — CausalFlow (Causal Responsibility Scores + counterfactual repairs)
- arXiv:2512.06749 — DoVer (orchestrator/sub-agent debugging via intervention)
- arXiv:2606.00765 — FALAT (oracle-free dependency-guided single-trajectory diagnosis)
- arXiv:2605.06737 — Self-Healing Framework for Reliable LLM-Based Autonomous Agents
- arXiv:2604.22708 — TraceElephant (ACL 2026): full-observability failure attribution benchmark
- TRAIL benchmark — independently reports ~18.3% joint accuracy for top attribution models
- ACL 2026 — Uncertainty quantification in LLM agents: verbalized confidence is unreliable; consistency-sampling AUROC remains close to random on current benchmarks
- arXiv:2605.03310 — Coordination as an Architectural Layer (predictive topology gap)
- OpenTelemetry GenAI Semantic Conventions (CNCF, dedicated GenAI semconv repo at github.com/open-telemetry/semantic-conventions-genai)
- 2026 AAAI paper on causal attribution in multi-agent systems (~17.1% step-level accuracy)
