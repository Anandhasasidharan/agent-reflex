import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "react-router";
import {
  Background,
  Controls,
  ReactFlow,
  type Edge,
  type Node,
  type NodeMouseHandler,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { api, ApiError, AuthError } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useAuthRedirect } from "../App";
import { EmptyState, ErrorState, Loading, Panel } from "../components/Async";
import type { Attribution, GraphNode, GraphResponse } from "../lib/types";

export const PLAYBOOKS = [
  "re_prompt",
  "backtrack_to_checkpoint",
  "swap_agent",
  "escalate_to_human",
  "circuit_breaker",
  "rate_limit_backoff",
  "context_window_summarize",
  "tool_fallback",
];

/** Node colouring follows the console's signalling language:
 * root cause = signal, error-flagged = degraded, normal = evidence. */
const NODE_FILL: Record<string, string> = {
  root: "#ff6b4a",
  error: "#f2b84b",
  normal: "#5eead4",
};

const fieldClass =
  "w-full rounded border border-line bg-bg px-2 py-1.5 text-ink focus:border-evidence focus:outline-none focus:ring-1 focus:ring-evidence/50";

export function SessionDetailPage() {
  const { sessionId = "" } = useParams();
  const { readKey, writeKey, setWriteKey } = useAuth();
  const redirectOnAuthError = useAuthRedirect();

  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [attribution, setAttribution] = useState<Attribution | null>(null);
  const [attributionNull, setAttributionNull] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notFound, setNotFound] = useState(false);

  const [feedback, setFeedback] = useState({
    playbook: PLAYBOOKS[0],
    success: true,
    partial: false,
  });
  const [writeKeyPrompt, setWriteKeyPrompt] = useState(false);
  const [writeInput, setWriteInput] = useState("");
  const [feedbackDone, setFeedbackDone] = useState(false);
  const [feedbackError, setFeedbackError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const load = useCallback(async () => {
    if (!readKey) return;
    setError(null);
    setNotFound(false);
    try {
      const [g, a] = await Promise.all([
        api.getGraph(sessionId, readKey),
        api.getAttribution(sessionId, readKey),
      ]);
      setGraph(g);
      setAttribution(a.attribution);
      setAttributionNull(a.attribution === null);
    } catch (err) {
      if (redirectOnAuthError(err)) return;
      if (err instanceof ApiError && err.status === 404) {
        setNotFound(true);
        return;
      }
      setError(err instanceof Error ? err.message : "Failed to load session.");
    }
  }, [readKey, sessionId, redirectOnAuthError]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- data fetch on mount; setState happens post-await
    void load();
  }, [load]);

  const nodes = useMemo(() => (graph ? layoutGraph(graph) : []), [graph]);
  const edges = useMemo(
    () =>
      (graph?.edges ?? []).map((e, i) => ({
        id: `e-${i}`,
        source: e.source,
        target: e.target,
        style: { stroke: e.type === "data_dependency" ? "#5eead4" : "#232a3d", strokeDasharray: e.type === "data_dependency" ? "5 5" : undefined },
      })) satisfies Edge[],
    [graph],
  );

  const onNodeClick: NodeMouseHandler = useCallback((_, node) => {
    setSelectedNodeId(String(node.id));
  }, []);

  const selectedNode = useMemo(
    () => graph?.nodes.find((n) => n.id === selectedNodeId) ?? null,
    [graph, selectedNodeId],
  );

  async function submitFeedback(e: React.FormEvent) {
    e.preventDefault();
    const keyForWrite = writeKey ?? (writeKeyPrompt ? writeInput.trim() : null);
    if (!keyForWrite) {
      setWriteKeyPrompt(true);
      return;
    }
    setSubmitting(true);
    setFeedbackError(null);
    setFeedbackDone(false);
    try {
      await api.recoveryFeedback(
        {
          session_id: sessionId,
          playbook_name: feedback.playbook,
          success: feedback.success,
          partial: feedback.partial,
        },
        keyForWrite,
      );
      if (writeKeyPrompt && writeInput.trim()) {
        setWriteKey(writeInput.trim());
        setWriteKeyPrompt(false);
      }
      setFeedbackDone(true);
    } catch (err) {
      if (err instanceof AuthError) {
        setFeedbackError("That write key was rejected — check it and try again.");
        return;
      }
      if (err instanceof ApiError && err.status === 403) {
        setWriteKeyPrompt(true);
        setFeedbackError("Your current key has read scope only. Paste a write-scoped key to submit feedback.");
        return;
      }
      setFeedbackError(err instanceof Error ? err.message : "Feedback submission failed.");
    } finally {
      setSubmitting(false);
    }
  }

  if (notFound) {
    return <EmptyState title={`Session ${sessionId} not found`} hint="It may have been pruned, or the ID is wrong." />;
  }
  if (error) return <ErrorState message={error} retry={load} />;
  if (!graph) return <Loading label="Loading session…" />;

  return (
    <div className="space-y-4">
      <header>
        <h1 className="break-all font-mono text-lg font-semibold tracking-tight">{sessionId}</h1>
        <p className="text-sm text-ink-dim">
          {graph.nodes.length} step(s), {graph.edges.length} edge(s). Click a node for its full OTAR detail.
        </p>
      </header>

      <div className="grid gap-4 xl:grid-cols-[1fr_340px]">
        <Panel title="Causal graph" actions={<GraphLegend />}>
          <div className="h-[480px] rounded-md border border-line bg-surface">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodeClick={onNodeClick}
              fitView
              proOptions={{ hideAttribution: true }}
              nodesConnectable={false}
              elementsSelectable
            >
              <Background color="#232a3d" gap={20} />
              <Controls />
            </ReactFlow>
          </div>
        </Panel>

        <div className="space-y-4">
          <Panel title="Attribution">
            {attributionNull || !attribution ? (
              <EmptyState
                title="Attribution unavailable for this session"
                hint="The LLM attribution failed or was skipped at ingest time. The graph is still persisted (best-effort ingestion) — the trace exists, but no failure type or root cause was determined."
              />
            ) : (
              <dl className="space-y-3 text-sm">
                <div>
                  <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-faint">Failure type</dt>
                  <dd className="mt-0.5 font-mono text-signal">{attribution.failure_type}</dd>
                </div>
                <div>
                  <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-faint">Root cause</dt>
                  <dd className="mt-0.5 font-mono text-ink">{attribution.cause_node_id ?? "—"}</dd>
                </div>
                <div>
                  <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-faint">Causal responsibility (CRS)</dt>
                  <dd className="mt-0.5 font-mono text-evidence">{attribution.crs.toFixed(2)}</dd>
                </div>
                <div>
                  <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-faint">Evidence</dt>
                  <dd className="mt-0.5 space-y-1.5">
                    {attribution.evidence.length === 0 ? (
                      <span className="text-ink-faint">no evidence recorded</span>
                    ) : (
                      attribution.evidence.map((e, i) => (
                        <p key={i} className="rounded-md border border-line/60 bg-bg/60 p-2 font-mono text-xs text-ink-dim">
                          {e}
                        </p>
                      ))
                    )}
                  </dd>
                </div>
              </dl>
            )}
          </Panel>

          <Panel title="Step detail">
            {selectedNode ? (
              <NodeDetail node={selectedNode} />
            ) : (
              <p className="text-sm text-ink-dim">Click a node in the graph to see its observation / thought / action / result.</p>
            )}
          </Panel>

          <Panel title="Recovery feedback">
            <p className="mb-3 text-xs text-ink-dim">
              Tell the bandit how this session's recovery went. Requires a write-scoped key.
            </p>
            <form onSubmit={submitFeedback} className="space-y-3 text-sm">
              <label className="block">
                <span className="mb-1 block text-xs text-ink-faint">Playbook</span>
                <select
                  value={feedback.playbook}
                  onChange={(e) => setFeedback({ ...feedback, playbook: e.target.value })}
                  className={`${fieldClass} w-full font-mono text-xs`}
                >
                  {PLAYBOOKS.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={feedback.success}
                  onChange={(e) => setFeedback({ ...feedback, success: e.target.checked })}
                  className="size-4 accent-evidence"
                />
                <span className="text-ink-dim">Recovery succeeded</span>
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={feedback.partial}
                  onChange={(e) => setFeedback({ ...feedback, partial: e.target.checked })}
                  className="size-4 accent-evidence"
                />
                <span className="text-ink-dim">Partially recovered</span>
              </label>

              {writeKeyPrompt && (
                <label className="block">
                  <span className="mb-1 block text-xs text-ink-faint">Write-scoped API key</span>
                  <input
                    type="password"
                    value={writeInput}
                    onChange={(e) => setWriteInput(e.target.value)}
                    placeholder="python -m agent_reflex.api.auth create feedback --scope=write"
                    className={`${fieldClass} w-full font-mono text-xs`}
                  />
                </label>
              )}

              {feedbackError && (
                <p role="alert" className="text-xs text-signal">
                  {feedbackError}
                </p>
              )}
              {feedbackDone && (
                <p role="status" className="text-xs text-evidence">
                  Feedback recorded. It appears in the overview's recovery stats on next load.
                </p>
              )}

              <button
                type="submit"
                disabled={submitting}
                className="rounded-md border border-evidence/40 bg-evidence-dim px-3 py-1.5 text-sm font-medium text-evidence hover:bg-evidence hover:text-bg disabled:opacity-50"
              >
                {submitting ? "Submitting…" : writeKey ? "Submit feedback" : writeKeyPrompt ? "Save key & submit" : "Submit feedback"}
              </button>
            </form>
          </Panel>
        </div>
      </div>
    </div>
  );
}

function NodeDetail({ node }: { node: GraphNode }) {
  const rows: Array<[string, string]> = [
    ["Observation", node.otar.observation],
    ["Thought", node.otar.thought],
    ["Action", node.otar.action],
    ["Result", node.otar.result],
  ];
  return (
    <div className="space-y-2 text-sm">
      <p className="font-mono text-xs text-ink-dim">
        {node.label}
        {node.error && <span className="ml-2 text-degraded">● error-flagged</span>}
        {node.is_root_cause && <span className="ml-2 text-signal">● root cause</span>}
      </p>
      {rows.map(([label, value]) => (
        <div key={label}>
          <dt className="text-[11px] uppercase tracking-[0.12em] text-ink-faint">{label}</dt>
          <dd className="mt-0.5 whitespace-pre-wrap rounded-md border border-line/60 bg-bg/60 p-2 font-mono text-xs text-ink-dim">
            {value || "—"}
          </dd>
        </div>
      ))}
    </div>
  );
}

function GraphLegend() {
  return (
    <div className="flex gap-3 text-[11px] text-ink-faint">
      <span className="flex items-center gap-1">
        <span className="inline-block size-2.5 rounded-full" style={{ background: NODE_FILL.root }} /> root cause
      </span>
      <span className="flex items-center gap-1">
        <span className="inline-block size-2.5 rounded-full" style={{ background: NODE_FILL.error }} /> error
      </span>
      <span className="flex items-center gap-1">
        <span className="inline-block size-2.5 rounded-full" style={{ background: NODE_FILL.normal }} /> normal
      </span>
    </div>
  );
}

/** Simple layered layout: columns by step_index, agents stacked within a layer. */
function layoutGraph(graph: GraphResponse): Node[] {
  const layers = new Map<number, string[]>();
  for (const n of graph.nodes) {
    const list = layers.get(n.step_index) ?? [];
    list.push(n.id);
    layers.set(n.step_index, list);
  }
  const positions = new Map<string, { x: number; y: number }>();
  for (const [step, ids] of layers) {
    ids.forEach((id, i) => positions.set(id, { x: step * 170, y: i * 120 }));
  }
  return graph.nodes.map((n) => {
    const pos = positions.get(n.id) ?? { x: 0, y: 0 };
    const fill = n.is_root_cause ? NODE_FILL.root : n.error ? NODE_FILL.error : NODE_FILL.normal;
    return {
      id: n.id,
      position: pos,
      data: { label: n.label },
      className: n.is_root_cause ? "is-root" : undefined,
      style: {
        background: fill,
        color: "#0b0e14",
        border: n.is_root_cause ? "2px solid #ff6b4a" : "1px solid #232a3d",
        borderRadius: 6,
        fontWeight: 600,
        fontSize: 12,
        padding: "6px 10px",
      },
      ariaLabel: `step ${n.label}${n.is_root_cause ? ", root cause" : ""}`,
    } satisfies Node;
  });
}