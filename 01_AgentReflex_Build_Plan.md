# AgentReflex — Upgraded Build Plan (Full Scope)

> **This replaces the earlier scoped-down MVP version.** You said you don't
> want to cut these down — you want the real thing. This version is built
> from research published as recently as this month (May–June 2026), and it
> is honestly harder and longer than the original 5–6 week plan. That's the
> point: MVP scope proves you can build a toy; this scope proves you can
> build the thing the field itself hasn't fully solved yet.

**Build time:** 9–10 weeks · **Difficulty:** Hard (research-adjacent) · **GPU required:** No (embedding/consistency-sampling calls only, no training)

## One-liner

A multi-agent reliability platform that doesn't just *detect* failures
against the MAST taxonomy — it reconstructs each execution as a
hierarchical causal graph, attributes failure to the exact responsible
step and agent using counterfactual intervention (not pattern matching),
learns which recovery strategy actually works per failure signature via a
bandit-based adaptive controller, and calibrates human escalation on
consistency-based uncertainty instead of the model's own (unreliable)
verbalized confidence.

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

## Read This Before You Build: Honesty Is the Feature

Several of the upgrades below are attempts at problems the field has not
solved. The Who&When benchmark — the standard eval set for failure
attribution — reports that even strong methods historically struggled to
exceed the high teens in step-level accuracy from logs alone, and newer
methods like CHIEF improve on prior baselines without claiming the problem
solved. Consistency-based uncertainty scoring for agent failure prediction
currently performs only modestly better than a coin flip on published
benchmarks. **Build the harder pieces, run them against a real eval set,
and report your actual numbers — including if they're mediocre.** A
README that says "my attribution engine hits 34% step-level accuracy on a
50-trace held-out set, here's the confusion matrix" is a far stronger
signal than one that vaguely claims a working self-healing system, because
it proves you understand this is genuinely unsolved and you measured
instead of asserting.

## Core Features

### 1. OTel GenAI-Native Instrumentation SDK
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
*is* your evidence the adaptive layer is worth anything. Where CausalFlow's
counterfactual repairs are available (a validated "wrong step → corrected
step" pair), use the corrected step directly as a targeted, minimal
repair instead of a generic re-prompt.

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

## Architecture

```
                 OTel GenAI-Native Instrumentation SDK
        (gen_ai.* + Task/Action/Agent/Team/Artifact/Memory)
                              |
                              v
              Collector + Causal Graph Reconstructor
                  (OTAR parsing, dependency graph)
                              |
                +-------------+--------------+
                |                             |
                v                             v
     Failure Attribution Engine       MAST+ Classifier
   (oracle-guided backtracking +      (14 MAST modes +
    counterfactual screening ->        Infrastructure/
    Causal Responsibility Score)       Operational category)
                |                             |
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
| Tracing | OpenTelemetry SDK, GenAI semantic conventions | Real standard, free interop with Datadog/Honeycomb/MLflow/Uptrace |
| Causal graph | Neo4j or NetworkX | Structured attribution needs a graph, not a flat table |
| Attribution logic | Custom oracle-guided backtracking + counterfactual scoring (Python) | No off-the-shelf library does this yet — this is the differentiator |
| Adaptive recovery | Hand-rolled contextual bandit (epsilon-greedy or Thompson sampling), or `mabwiser` | Lightweight; no need for a full RL framework |
| Uncertainty | Multi-sample consistency checking against your own LLM calls | Avoids the documented unreliability of verbalized confidence |
| Storage | Postgres (traces + graph edges) | Relational + graph-adjacent queries |
| Dashboard | Grafana + a custom causal-graph viewer (D3/vis.js) | Standard metrics view + a purpose-built graph view |
| Eval | Who&When benchmark (or a self-constructed equivalent of similar shape) | Standard, citable way to report attribution accuracy |

## Implementation Plan

**Phase 1 (Weeks 1–2) — OTel-native SDK + causal graph reconstruction:**
decorators emitting real `gen_ai.*`-conventioned spans; OTAR parsing of
each step; dependency-graph construction from the trace.

**Phase 2 (Weeks 3–4) — Failure Attribution Engine + MAST+ classifier:**
implement oracle-guided backtracking and counterfactual screening;
validate against Who&When or your own constructed labeled test set; report
honest step-level and agent-level accuracy numbers, including a confusion
matrix.

**Phase 3 (Weeks 5–6) — Adaptive Recovery Engine:** playbook library;
contextual bandit selector; counterfactual-repair generation where
attribution data supports it; run enough trials to report a real
recovery-rate comparison between adaptive selection and a static baseline.

**Phase 4 (Week 7) — Uncertainty-calibrated escalation:** consistency-
sampling implementation; threshold calibration against your held-out
failure set; report AUROC honestly.

**Phase 5 (Week 8, stretch) — Predictive topology risk scoring:** attempt
it, report whatever you find, including negative or mixed results.

**Phase 6 (Weeks 9–10) — Dashboard, causal graph viewer, and writeup:**
polish; a scripted live demo (trigger a failure, watch it get attributed,
watch recovery selection happen, watch escalation trigger); a results
section in your README with all the numbers from Phases 2–5 in one place.

## Interview Narrative

"Detecting agent failures against a taxonomy is table stakes now — every
major observability vendor does it. I built the harder layer underneath:
a causal graph reconstruction of each execution, counterfactual attribution
to find the actual root-cause step instead of just tagging symptoms, an
adaptive recovery controller that learns which playbook works per failure
signature instead of using a fixed mapping, and an escalation trigger
calibrated on consistency-sampled uncertainty instead of the model's own
unreliable confidence. Some of this — especially the predictive piece — is
close to an open research problem, and I report my actual numbers rather
than claiming it's solved."

## Recruiter Signal

| What it proves | Why it matters |
|---|---|
| Reads and implements bleeding-edge research | You're citing papers from the same month you're building, not last year |
| Distributed-systems reliability thinking | Causal attribution and adaptive recovery are the same intellectual move as root-cause analysis in traditional distributed systems |
| Honest empirical reporting | Reporting real, possibly mediocre numbers on hard problems reads as more credible than an unsubstantiated "it works" |
| Depth over breadth | One system built deeply beats five systems built shallowly, in an interview |

## Sources

- arXiv:2503.13657 — MAST Failure Taxonomy (NeurIPS 2025)
- arXiv:2602.23701 — CHIEF: hierarchical causal graph + oracle-guided
  backtracking + counterfactual attribution
- arXiv:2605.25338 — CausalFlow: Causal Responsibility Scores + minimal
  counterfactual repairs
- arXiv:2512.06749 — DoVer: intervention-driven debugging for
  orchestrator/sub-agent topologies
- Who&When benchmark — standard eval set for failure attribution accuracy
- arXiv:2605.06737 — A Self-Healing Framework for Reliable LLM-Based
  Autonomous Agents (May 2026): reliability-aware detection + adaptive
  replanning
- 2026 AAAI paper reframing multi-agent reliability via Byzantine fault
  tolerance
- ACL 2026 paper on uncertainty quantification in LLM agents: verbalized
  confidence is unreliable; consistency-sampling AUROC remains close to
  random on current benchmarks
- arXiv:2605.03310 — "Coordination as an Architectural Layer": explicitly
  identifies the gap between architecture/topology and predicted failure
  modes
- DeepMind "Science of Scaling" / bag-of-agents error-amplification
  findings — motivates topology-risk scoring
- OpenTelemetry GenAI Semantic Conventions (CNCF) and the open Agentic
  Systems conventions proposal (GitHub, semantic-conventions-genai #35)
