# AgentReflex — Build Plan v2 (July 2026)

> **v2 changelog:** merges the original "Full Scope" plan with the
> July 2026 landscape check. Two real additions (TraceElephant eval,
> a redaction-vs-accuracy ablation), one hygiene pass (OTel semconv
> location + attributes), and two new "why we didn't" README paragraphs.
> Everything marked **[NEW]** below is an addition; everything else is
> unchanged from v1. Total added effort: ~1 week, mostly absorbed inside
> existing phases rather than appended — this is still a 9–10 week plan,
> not a 10–11 week one.

**Build time:** 9–10 weeks · **Difficulty:** Hard (research-adjacent) · **GPU required:** No

## One-liner

A multi-agent reliability platform that doesn't just *detect* failures
against the MAST taxonomy — it reconstructs each execution as a
hierarchical causal graph, attributes failure to the exact responsible
step and agent using counterfactual intervention (not pattern matching),
learns which recovery strategy actually works per failure signature via a
bandit-based adaptive controller, calibrates human escalation on
consistency-based uncertainty instead of the model's own (unreliable)
verbalized confidence — **[NEW]** and validates all of this against two
independent benchmarks with different design philosophies, plus one
measured result about its own instrumentation choices that nobody else
in this space has published.

## What Changed From the MVP Version, and Why

The MVP plan scoped down to: pattern + embedding classification against
5–8 MAST modes, fixed recovery playbooks, a dashboard. That's a solid,
completable six-week project — and it's also now roughly what Arize,
Galileo, Confident AI, and Braintrust already ship. Detection-against-a-
taxonomy is commoditized. Here's what genuinely isn't, as of mid-2026:

- **Flat classification vs. causal attribution.** Tagging a trace with
  "task derailment" tells you *what* happened. It doesn't tell you *which*
  upstream step *caused* it. A cluster of 2026 papers — CHIEF
  (arXiv:2602.23701), CausalFlow (arXiv:2605.25338), DoVer
  (arXiv:2512.06749) — treat this as a distinct, harder problem: converting
  flat logs into a causal graph and using counterfactual intervention to
  separate true root causes from downstream symptoms. This is close to a
  live research frontier, not a solved feature.
- **Static playbooks vs. adaptive, learned recovery.** The MVP's playbook
  engine is a fixed lookup table. A May 2026 self-healing framework paper
  (arXiv:2605.06737) pairs failure detection with a *quantitative
  reliability model* and *adaptive replanning*, not just a static response.
  Separately, hierarchical reinforcement-learning approaches to self-
  healing (applied so far mostly in cyber-defense contexts) show 15–25%
  faster recovery when a master policy learns which sub-policy to invoke,
  instead of a fixed if/else mapping.
- **Verbalized confidence vs. calibrated uncertainty for escalation.** The
  MVP didn't specify how the self-healing controller decides *when* to
  trust an agent's own claim that it succeeded. An ACL 2026 paper found
  that asking a model how confident it is degrades badly with the noisy
  long-context memory agents accumulate, and that consistency-sampling
  (asking the same question multiple times, measuring agreement) is a more
  principled — though still imperfect — signal.
- **Reactive-only vs. an attempt at prediction.** A separate 2026 paper on
  coordination as an architectural layer makes an explicit observation: no
  existing method connects an agent system's *architecture* (its topology,
  before it ever runs) to a predicted distribution over which of the 14
  MAST failure modes it's likely to exhibit. That's an open gap, not a
  shipped feature anywhere. Attempting a scaled-down version of it — and
  reporting the result honestly, even if mixed — is a genuinely
  differentiated thing to put in a portfolio.
- **[NEW] Single-benchmark attribution claims vs. cross-benchmark
  validation.** By mid-2026, Who&When is no longer the only failure-
  attribution eval in the field. TraceElephant (ACL 2026) specifically
  argues that output-only, partial-observability benchmarks like Who&When
  understate what's achievable, because a developer-facing debugging tool
  should get full execution observability. Reporting accuracy on only one
  benchmark now reads as less rigorous than it did when this plan was
  first written — a second benchmark with a different design philosophy
  is a stronger, harder-to-dismiss claim.

## Read This Before You Build: Honesty Is the Feature

Several of the upgrades below are attempts at problems the field has not
solved. The Who&When benchmark — the standard eval set for failure
attribution — reports that even strong methods historically struggled to
exceed the high teens in step-level accuracy from logs alone, and newer
methods like CHIEF improve on prior baselines without claiming the problem
solved. **[NEW]** This is no longer a claim resting on one benchmark: TRAIL
independently reports ~18.3% joint accuracy for top models on a similarly
shaped task, and a 2026 AAAI causal-attribution paper independently reports
~17.1% on a Who&When-style setup. Three independent sources converging on
"under 20%" is a much stronger basis for "this is genuinely hard" than one
paper's number was. Consistency-based uncertainty scoring for agent
failure prediction currently performs only modestly better than a coin
flip on published benchmarks. **Build the harder pieces, run them against
real eval sets, and report your actual numbers — including if they're
mediocre.** A README that says "my attribution engine hits 34% step-level
accuracy on Who&When and 29% on TraceElephant, here's why those numbers
differ and here's the confusion matrix for both" is a far stronger signal
than a single-benchmark claim, because it proves you understand this is
genuinely unsolved and you measured rather than asserted, from more than
one angle.

## Core Features

### 1. OTel GenAI-Native Instrumentation SDK **[UPDATED]**
Build on the real, CNCF-backed OpenTelemetry GenAI semantic conventions
instead of a custom schema — `gen_ai.request.model`, `gen_ai.usage.*`
tokens, an `invoke_agent` span with child `chat` and `execute_tool` spans.
This buys you free interoperability with Datadog, Honeycomb, New Relic,
MLflow, and Uptrace, all of which natively understand these attributes
now. Layer the emerging Agentic Systems conventions (still an open GitHub
proposal, but worth adopting early) on top for multi-agent-specific
concepts: Task, Action, Agent, Team, Artifact, Memory. Store prompt/
completion content as span *events*, not attributes (the documented
anti-pattern is storing large text in indexed attributes), with a
redaction layer in the collector for anything sensitive.

**[NEW]** Two concrete, current-spec changes to make from day one:

- **The `gen_ai.*` conventions have moved to a dedicated OpenTelemetry
  GenAI semantic conventions repository**, separate from the main
  `open-telemetry/semantic-conventions` repo. Point your Sources section
  and onboarding docs at the new location.
- **Explicitly set `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`**
  and say so in your README setup instructions. The conventions are still
  formally in "Development" status (attribute names can still shift), and
  instrumentation libraries default to the *old* attribute set unless you
  opt in — a reviewer who knows OTel will check this specifically.
- Adopt `gen_ai.usage.cache_read.input_tokens` /
  `cache_creation.input_tokens` if your target framework uses prompt
  caching, and `gen_ai.request.stream` /
  `gen_ai.response.time_to_first_chunk` for streaming tool calls — both
  are new since the original plan and worth including if relevant to your
  chosen framework (LangGraph/CrewAI).

### 2. Hierarchical Causal Graph Reconstruction
Instead of a flat trace, parse each step into Observation-Thought-Action-
Result (OTAR) structure (following CHIEF's approach) and explicitly model
data dependencies between steps as a graph, decomposing the overall task
into subtasks. This is what makes attribution tractable — you're
searching a structured graph, not scanning a linear log.

### 3. Failure Attribution Engine (the centerpiece)
Given a failed execution's causal graph:
- **Oracle-guided backtracking:** a top-down, divide-and-conquer search
  that verifies subtasks against synthesized "virtual oracles" rather than
  inspecting every granular action, pruning the search space toward the
  likely failure point (CHIEF's approach).
- **Counterfactual screening:** for candidate failure steps, compute a
  Causal Responsibility Score by intervening on that step (would the
  outcome have changed?) to distinguish true root causes from steps that
  merely propagated someone else's failure (CausalFlow's approach).
- Because your target is a multi-agent orchestrator/sub-agent topology
  (LangGraph/CrewAI), lean toward DoVer's framing over CausalFlow's —
  DoVer is built specifically for orchestrator/sub-agent systems, using
  re-plan steps as natural checkpoints, which maps directly onto how
  LangGraph structures execution.
- Output: not just "task derailment" but "task derailment, caused by
  step 7 in Agent B, with Causal Responsibility Score 0.82."

**[NEW — design-rationale note, add to README, do not implement]:** FALAT
(arXiv:2606.00765) takes the opposite design bet from this plan — a
dependency-guided search that diagnoses a single failed trajectory with
*no* oracle synthesis and no task-specific training data, explicitly
positioned against CHIEF's oracle-dependence. No public reference
implementation was found for FALAT at time of writing, so this plan does
not build it as a comparison baseline. But the tradeoff is real and worth
naming explicitly: oracle-guided backtracking (this plan's choice) costs
you oracle-synthesis overhead and some reliance on the oracle being
correct, in exchange for a more directed search; FALAT's no-oracle
approach avoids that cost but searches a larger space per trajectory.
State this tradeoff plainly rather than presenting oracle-guidance as the
obviously correct choice.

### 4. MAST+ Classifier
The original 14 MAST modes across Specification/Coordination/Verification,
*plus* a fourth category several 2026 production analyses call out as
missing from the academic taxonomy: **Infrastructure/Operational**
failures — rate limits, context-window overflow, cascading timeouts. These
are less frequent than spec/coordination issues but produce the most
visible production disruptions, and MAST alone doesn't cover them.

### 5. Quantitative Reliability Scoring
Move from a binary pass/fail per session to a continuous reliability
score per agent and per session, following the reliability-aware framing
of the May 2026 self-healing paper — this is what lets you show a trend
line ("this agent's reliability score improved 18% after playbook X was
introduced") instead of just a failure count.

### 6. Adaptive Recovery Engine
Keep the MVP's playbook library, but stop treating it as a fixed lookup.
Add a contextual bandit layer (a hand-rolled epsilon-greedy or Thompson
sampling implementation is plenty — you don't need a heavy RL framework)
that selects among available playbooks per failure signature and updates
based on observed recovery success. Report the actual recovery-rate lift
of the adaptive selector over a static-mapping baseline — this comparison
*is* your evidence the adaptive layer is worth anything. Where
CausalFlow's counterfactual repairs are available (a validated "wrong
step → corrected step" pair), use the corrected step directly as a
targeted, minimal repair instead of a generic re-prompt.

### 7. Uncertainty-Calibrated Escalation
Replace "ask the agent how confident it is" with consistency-sampling:
run the same step N times (start with N=5), measure agreement, and use
that as your escalation signal instead of verbalized confidence. Calibrate
a threshold against your own held-out failure set and report your AUROC
honestly against the ~0.5 random baseline the field is currently stuck
near — this is exactly the kind of result an ML-literate interviewer will
respect more than an unsubstantiated claim of a "smart" escalation policy.

### 8. Predictive Topology Risk Scoring (stretch, explicitly experimental)
Score an agent architecture's likely failure-mode distribution *before*
running it, from static topology features: fan-out per node, hierarchy
depth, and whether the structure looks like an unconstrained "bag of
agents" versus a formal planner-worker topology (unstructured topologies
have been linked to dramatically higher error amplification in recent
scaling research). Treat this as a genuinely open research attempt. If it
doesn't work well, that's a fine, honest result to report.

### 9. Dashboard + Causal Graph Viewer
Failure heatmap by MAST+ category, per-strategy recovery rate (adaptive
vs. static baseline, shown side by side), agent-level reliability trend,
and a causal graph viewer highlighting the attributed root-cause node in
red. If you ever build AgentMeshSec, this viewer can reuse the same
Neo4j/graph-visualization stack.

### 10. [NEW] Cross-Benchmark Attribution Evaluation
Run the Failure Attribution Engine against **both** Who&When (as
originally planned) and **TraceElephant** (`github.com/TraceElephant/
TraceElephant` — public repo, dataset hosted on HuggingFace, one-click
execution runner already implemented). Report step-level and agent-level
accuracy on both, side by side, with a short explanation of why the
numbers might differ: Who&When is output-only/partial-observability,
TraceElephant provides fully observable, reproducible execution
environments and is explicitly designed to enable dynamic attribution
methods (including counterfactual execution analysis) — which is a closer
match to what this plan's attribution engine is actually doing than
Who&When's original framing assumed.

### 11. [NEW] Redaction-vs-Attribution-Accuracy Ablation
This is the one genuinely novel empirical result in this plan — a
measurement about *your own system*, not a citation of someone else's.
Run the Failure Attribution Engine twice over the same held-out failed-
execution set: once with your OTel collector's redaction layer (Feature
1) disabled (full span content visible), once with it enabled (sensitive
content redacted per your collector policy). Report whether attribution
accuracy degrades when the engine only has redacted spans to reason
over, and by how much. This directly connects two of this plan's own
features — the instrumentation layer and the attribution engine — into
one finding nobody else in this space has published for their own system.

## Architecture

```
                 OTel GenAI-Native Instrumentation SDK
        (gen_ai.* + Task/Action/Agent/Team/Artifact/Memory)
     [opted into gen_ai_latest_experimental; cache + streaming attrs]
                              |
                              v
              Collector + Causal Graph Reconstructor
             (OTAR parsing, dependency graph, redaction layer)
                              |
                +-------------+--------------+
                |                             |
                v                             v
     Failure Attribution Engine       MAST+ Classifier
   (oracle-guided backtracking +      (14 MAST modes +
    counterfactual screening ->        Infrastructure/
    Causal Responsibility Score)       Operational category)
                |                             |
                |   [NEW] evaluated against both
                |   Who&When AND TraceElephant,
                |   plus redaction on/off ablation
                |
                +-------------+--------------+
                              |
                              v
                 Quantitative Reliability Scorer
                              |
                +-------------+--------------+
                |                             |
                v                             v
      Adaptive Recovery Engine        Uncertainty-Calibrated
   (playbooks + contextual bandit      Escalation Controller
    + counterfactual repair)          (consistency-sampling)
                |                             |
                +-------------+--------------+
                              |
                              v
        Dashboard + Causal Graph Viewer + Reliability Trends
                              |
                (stretch) Predictive Topology Risk Scorer
                  (pre-deployment architecture analysis)
```

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| Tracing | OpenTelemetry SDK, GenAI semantic conventions, `gen_ai_latest_experimental` opt-in | Real standard, free interop with Datadog/Honeycomb/MLflow/Uptrace; opt-in avoids silently running on stale attribute names |
| Causal graph | Neo4j or NetworkX | Structured attribution needs a graph, not a flat table |
| Attribution logic | Custom oracle-guided backtracking + counterfactual scoring (Python) | No off-the-shelf library does this yet — this is the differentiator |
| Adaptive recovery | Hand-rolled contextual bandit (epsilon-greedy or Thompson sampling), or `mabwiser` | Lightweight; no need for a full RL framework |
| Uncertainty | Multi-sample consistency checking against your own LLM calls | Avoids the documented unreliability of verbalized confidence |
| Storage | Postgres (traces + graph edges) | Relational + graph-adjacent queries |
| Dashboard | Grafana + a custom causal-graph viewer (D3/vis.js) | Standard metrics view + a purpose-built graph view |
| Eval | **[NEW]** Who&When + TraceElephant (both) | Two benchmarks, two observability philosophies — harder to dismiss as one-eval overfit |

## Implementation Plan

**Phase 1 (Weeks 1–2) — OTel-native SDK + causal graph reconstruction:**
decorators emitting real `gen_ai.*`-conventioned spans (opted into
`gen_ai_latest_experimental`, including cache/streaming attributes where
relevant); OTAR parsing of each step; dependency-graph construction from
the trace.

**Phase 2 (Weeks 3–4) — Failure Attribution Engine + MAST+ classifier
[UPDATED]:** implement oracle-guided backtracking and counterfactual
screening; validate against **both Who&When and TraceElephant** (clone
`TraceElephant/TraceElephant`, pull the HuggingFace dataset, run its
existing one-click runner); report honest step-level and agent-level
accuracy on both, plus a confusion matrix for each and a short note on why
the numbers may differ. Budget +2–3 days here versus v1 for the second
benchmark integration.

**Phase 2b (end of Phase 1 / start of Phase 2, ~2–3 days) — [NEW]
Redaction ablation:** run the attribution engine with your collector's
redaction layer on and off over the same held-out set; report the
accuracy delta as its own small table.

**Phase 3 (Weeks 5–6) — Adaptive Recovery Engine:** playbook library;
contextual bandit selector; counterfactual-repair generation where
attribution data supports it; run enough trials to report a real
recovery-rate comparison between adaptive selection and a static baseline.

**Phase 4 (Week 7) — Uncertainty-calibrated escalation:** consistency-
sampling implementation; threshold calibration against your held-out
failure set; report AUROC honestly.

**Phase 5 (Week 8, stretch) — Predictive topology risk scoring:** attempt
it, report whatever you find, including negative or mixed results.

**Phase 6 (Weeks 9–10) — Dashboard, causal graph viewer, and writeup
[UPDATED]:** polish; a scripted live demo (trigger a failure, watch it get
attributed, watch recovery selection happen, watch escalation trigger); a
results section in your README with all the numbers from Phases 2–5 in
one place, **plus**: the OTel semconv repo/attribute updates in Sources;
the "why oracle-guided over FALAT's oracle-free approach" paragraph; the
"why a hand-rolled bandit pipeline over AgenTracer/GraphTracer's
fine-tuning route" paragraph; TRAIL added to Sources alongside Who&When.

## Interview Narrative

"Detecting agent failures against a taxonomy is table stakes now — every
major observability vendor does it. I built the harder layer underneath:
a causal graph reconstruction of each execution, counterfactual attribution
to find the actual root-cause step instead of just tagging symptoms, an
adaptive recovery controller that learns which playbook works per failure
signature instead of using a fixed mapping, and an escalation trigger
calibrated on consistency-sampled uncertainty instead of the model's own
unreliable confidence. I validated attribution against two independent
benchmarks with different observability assumptions — Who&When and
TraceElephant — rather than just one, and I measured something nobody else
has published: how much attribution accuracy costs you when your traces
are redacted for privacy versus when they're not. Some of this — especially
the predictive piece — is close to an open research problem, and I report
my actual numbers rather than claiming it's solved."

## Recruiter Signal

| What it proves | Why it matters |
|---|---|
| Reads and implements bleeding-edge research | You're citing papers from the same month you're building, not last year |
| Distributed-systems reliability thinking | Causal attribution and adaptive recovery are the same intellectual move as root-cause analysis in traditional distributed systems |
| Honest empirical reporting | Reporting real, possibly mediocre numbers on hard problems reads as more credible than an unsubstantiated "it works" |
| Depth over breadth | One system built deeply beats five systems built shallowly, in an interview |
| **[NEW]** Cross-benchmark rigor | Validating against two benchmarks with different design philosophies, not just the one your architecture happens to fit best |
| **[NEW]** Original empirical contribution | The redaction ablation is a result about your own system — not a citation, a finding |

## Explicitly Not Doing (and why)

- **Reimplementing FALAT or AgenTracer/GraphTracer as comparison
  baselines.** Neither has confirmed public code as of this writing.
  Rebuilding either from a paper description for one comparison data point
  would cost 1–2 weeks against a plan that's already tight. Cited as
  design-rationale paragraphs instead (see Feature 3 and Sources).
- **A third benchmark.** Two (Who&When, TraceElephant) is enough to make
  the cross-validation point; a third adds diminishing signal for real
  time cost.

## Sources

- arXiv:2503.13657 — MAST Failure Taxonomy (NeurIPS 2025)
- arXiv:2602.23701 — CHIEF: hierarchical causal graph + oracle-guided
  backtracking + counterfactual attribution
- arXiv:2605.25338 — CausalFlow: Causal Responsibility Scores + minimal
  counterfactual repairs
- arXiv:2512.06749 — DoVer: intervention-driven debugging for
  orchestrator/sub-agent topologies
- Who&When benchmark — standard eval set for failure attribution accuracy
- **[NEW]** arXiv:2604.22708 — TraceElephant (ACL 2026): "Seeing the Whole
  Elephant" — full-observability failure attribution benchmark;
  `github.com/TraceElephant/TraceElephant`, dataset on HuggingFace
- **[NEW]** TRAIL benchmark — independently reports ~18.3% joint accuracy,
  reinforcing the attribution-gap framing alongside Who&When
- **[NEW]** arXiv:2606.00765 — FALAT: oracle-free dependency-guided search
  for single-trajectory failure diagnosis (cited for design-tradeoff
  discussion, not implemented)
- arXiv:2605.06737 — A Self-Healing Framework for Reliable LLM-Based
  Autonomous Agents (May 2026): reliability-aware detection + adaptive
  replanning
- 2026 AAAI paper reframing multi-agent reliability via Byzantine fault
  tolerance; also independently reports ~17.1% attribution accuracy on a
  Who&When-style setup
- ACL 2026 paper on uncertainty quantification in LLM agents: verbalized
  confidence is unreliable; consistency-sampling AUROC remains close to
  random on current benchmarks
- arXiv:2605.03310 — "Coordination as an Architectural Layer": explicitly
  identifies the gap between architecture/topology and predicted failure
  modes
- DeepMind "Science of Scaling" / bag-of-agents error-amplification
  findings — motivates topology-risk scoring
- OpenTelemetry GenAI Semantic Conventions (CNCF, now in a dedicated
  GenAI semantic conventions repository) and the open Agentic Systems
  conventions proposal (GitHub, semantic-conventions-genai #35)
