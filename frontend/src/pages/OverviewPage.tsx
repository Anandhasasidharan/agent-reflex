import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useAuthRedirect } from "../App";
import { EmptyState, ErrorState, HealthBadge, Loading, Panel } from "../components/Async";
import { Sparkline } from "../components/Sparkline";
import type { AgentStatus, EvalResultsResponse, HeatmapCell, RecoveryBreakdownRow, RecoveryStats, SessionSummary } from "../lib/types";

export function OverviewPage() {
  const { readKey } = useAuth();
  const redirectOnAuthError = useAuthRedirect();

  const [agents, setAgents] = useState<AgentStatus[] | null>(null);
  const [heatmap, setHeatmap] = useState<HeatmapCell[] | null>(null);
  const [recoveryStats, setRecoveryStats] = useState<RecoveryStats | null>(null);
  const [breakdown, setBreakdown] = useState<RecoveryBreakdownRow[] | null>(null);
  const [evalData, setEvalData] = useState<EvalResultsResponse | null>(null);
  const [failures, setFailures] = useState<SessionSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  const load = useCallback(async () => {
    if (!readKey) return;
    setError(null);
    try {
      const [a, h, rs, b, ev, f] = await Promise.all([
        api.agentsStatus(readKey),
        api.heatmap(readKey),
        api.recoveryStats(readKey),
        api.recoveryBreakdown(readKey),
        api.evalResults(readKey),
        api.listSessions({ limit: 50 }, readKey),
      ]);
      setAgents(a);
      setHeatmap(h);
      setRecoveryStats(rs);
      setBreakdown(b);
      setEvalData(ev);
      setFailures(f.items.filter((s) => s.failure_type !== null));
    } catch (err) {
      if (redirectOnAuthError(err)) return;
      setError(err instanceof Error ? err.message : "Failed to load overview data.");
    }
  }, [readKey, redirectOnAuthError]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- data fetch on mount; setState happens post-await
    void load();
  }, [load, reloadKey]);

  // Poll every 30s so the overview tracks newly-ingested sessions.
  useEffect(() => {
    const timer = setInterval(() => setReloadKey((k) => k + 1), 30_000);
    return () => clearInterval(timer);
  }, []);

  if (error) return <ErrorState message={error} retry={load} />;
  if (!agents || !heatmap || !recoveryStats || !breakdown || !failures) return <Loading label="Loading overview…" />;

  const sortedAgents = [...agents].sort(
    (x, y) => healthRank(x.health) - healthRank(y.health),
  );

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-xl font-semibold tracking-tight">Fleet overview</h1>
        <p className="text-sm text-ink-dim">
          Real data from the backend API — no placeholders. Polled every 30 seconds.
        </p>
      </header>

      <Panel title="Fleet health" actions={<span className="text-xs text-ink-faint">{sortedAgents.length} agents</span>}>
        {sortedAgents.length === 0 ? (
          <EmptyState title="No agents yet" hint="Ingest a trace (e.g. demo/reference_agent.py) to populate this table." />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-line text-xs uppercase tracking-wide text-ink-faint">
                  <th className="py-2 pr-4 font-medium">Agent</th>
                  <th className="py-2 pr-4 font-medium">Health</th>
                  <th className="py-2 pr-4 font-medium">Reliability</th>
                  <th className="py-2 pr-4 font-medium">Trend</th>
                  <th className="py-2 font-medium">Sessions</th>
                </tr>
              </thead>
              <tbody>
                {sortedAgents.map((agent) => (
                  <tr key={agent.agent_id} className="border-b border-line">
                    <td className="py-2 pr-4 font-mono text-ink">{agent.agent_id}</td>
                    <td className="py-2 pr-4">
                      <HealthBadge health={agent.health} />
                    </td>
                    <td className="py-2 pr-4">
                      {agent.score === null ? (
                        <span className="text-ink-faint">no data</span>
                      ) : (
                        <span className={scoreColor(agent.score)}>{agent.score.toFixed(3)}</span>
                      )}
                    </td>
                    <td className="py-2 pr-4">
                      <div className="flex items-center gap-2">
                        <Sparkline values={agent.history} />
                        <span className="text-xs text-ink-faint">{trendArrow(agent.trend)}</span>
                      </div>
                    </td>
                    <td className="py-2 text-ink-faint">{agent.n_sessions}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="Failure heatmap">
          <Heatmap cells={heatmap} />
        </Panel>

        <Panel title="Recovery effectiveness">
          {recoveryStats.total_trials === 0 && breakdown.length === 0 ? (
            <EmptyState
              title="No recovery trials yet"
              hint="Submit recovery feedback from a session detail page (requires a write key) to see adaptive vs static effectiveness."
            />
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <StatCard label="Adaptive selector" value={pct(recoveryStats.adaptive_success_rate)} sub={`${recoveryStats.adaptive_trials} trials`} />
                <StatCard label="Static selector" value={pct(recoveryStats.static_success_rate)} sub={`${recoveryStats.static_trials} trials`} />
              </div>
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-line text-ink-faint">
                    <th className="py-1 pr-3 font-medium">Playbook</th>
                    <th className="py-1 pr-3 font-medium">Selector</th>
                    <th className="py-1 pr-3 text-right font-medium">Total</th>
                    <th className="py-1 text-right font-medium">Success rate</th>
                  </tr>
                </thead>
                <tbody>
                  {breakdown.map((row, i) => (
                    <tr key={`${row.playbook}-${row.selector}-${i}`} className="border-b border-line">
                      <td className="py-1 pr-3 font-mono text-ink-dim">{row.playbook}</td>
                      <td className="py-1 pr-3 text-ink-dim">{row.selector}</td>
                      <td className="py-1 pr-3 text-right text-ink-dim">{row.total}</td>
                      <td className="py-1 text-right text-ink-dim">{pct(row.success_rate)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      </div>

      <Panel title="Attribution accuracy (measured, not asserted)">
        {!evalData || !evalData.available ? (
          <EmptyState
            title="No comparison eval results on the server"
            hint="Run python -m agent_reflex.eval.runner --compare --save on the backend to populate eval_results/."
          />
        ) : (
          <EvalComparison data={evalData} />
        )}
      </Panel>

      <Panel title="Recent failures" actions={<span className="text-xs text-ink-faint">{failures.length} with attributed failure</span>}>
        {failures.length === 0 ? (
          <EmptyState title="No failures recorded" hint="When a session is attributed with a failure type it shows up here, linking into its causal graph." />
        ) : (
          <ul className="divide-y divide-line">
            {failures.slice(0, 10).map((s) => (
              <li key={s.session_id}>
                <Link to={`/sessions/${encodeURIComponent(s.session_id)}`} className="flex flex-wrap items-center gap-3 py-2 hover:bg-surface/60">
                  <span className="rounded bg-signal-dim px-1.5 py-0.5 font-mono text-[11px] text-signal">{s.failure_type}</span>
                  <span className="min-w-0 flex-1 truncate text-sm text-ink">
                    {s.task_description || "(no task description)"}
                  </span>
                  <span className="font-mono text-xs text-ink-faint">{shortId(s.session_id)}</span>
                  <span className="text-xs text-ink-faint">{formatDate(s.created_at)}</span>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </Panel>
    </div>
  );
}

function healthRank(h: string): number {
  return h === "critical" ? 0 : h === "degraded" ? 1 : h === "healthy" ? 2 : 3;
}

function scoreColor(score: number): string {
  if (score >= 0.7) return "text-evidence";
  if (score >= 0.4) return "text-degraded";
  return "text-signal";
}

function trendArrow(trend: string): string {
  return trend === "up" ? "▲" : trend === "down" ? "▼" : "—";
}

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function shortId(id: string): string {
  return id.length > 12 ? `${id.slice(0, 8)}…` : id;
}

function formatDate(raw: string | null): string {
  if (!raw) return "—";
  const d = new Date(raw);
  if (Number.isNaN(d.getTime())) return raw;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function StatCard({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="rounded border border-line bg-surface p-3">
      <p className="text-xs text-ink-faint">{label}</p>
      <p className="mt-1 text-xl font-semibold tracking-tight text-ink">{value}</p>
      <p className="text-xs text-ink-faint">{sub}</p>
    </div>
  );
}

/** Failure heatmap as a matrix: rows = failure type, columns = day. */
function Heatmap({ cells }: { cells: HeatmapCell[] }) {
  if (cells.length === 0) {
    return <EmptyState title="No attributed failures yet" hint="Failures with a MAST+ type appear here by day." />;
  }
  const byKey = new Map(cells.map((c) => [`${c.failure_type}|${c.date}`, c.count]));
  const types = [...new Set(cells.map((c) => c.failure_type))].sort();
  const dates = [...new Set(cells.map((c) => c.date))].sort().slice(-14);
  const maxCount = Math.max(...cells.map((c) => c.count), 1);

  return (
    <div className="overflow-x-auto">
      <table className="text-xs">
        <thead>
          <tr>
            <th className="pr-2 text-left font-medium text-ink-faint">failure type</th>
            {dates.map((d) => (
              <th key={d} className="px-1 pb-1 text-right font-medium text-ink-faint" title={d}>
                {d.slice(8, 10)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {types.map((t) => (
            <tr key={t}>
              <td className="max-w-40 truncate pr-2 font-mono text-[11px] text-ink-dim" title={t}>
                {t}
              </td>
              {dates.map((d) => {
                const count = byKey.get(`${t}|${d}`) ?? 0;
                return (
                  <td key={d} className="px-1 py-0.5 text-center">
                    <span
                      title={`${t} on ${d}: ${count}`}
                      className={`inline-block h-4 w-4 rounded-sm ${count === 0 ? "bg-line" : "bg-signal"} `}
                      style={count === 0 ? undefined : { opacity: 0.3 + 0.7 * (count / maxCount) }}
                    />
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** The project's honest framing: oracle AND naive baseline, both shown. */
function EvalComparison({ data }: { data: EvalResultsResponse }) {
  const oracle = data.oracle;
  const naive = data.naive_baseline;
  if (!oracle || !naive) return <EmptyState title="Incomplete eval data" />;
  return (
    <div>
      <p className="mb-3 text-xs text-ink-dim">
        Side-by-side step attribution on {data.total_scenarios ?? "?"} synthetic scenarios (file:{" "}
        <code className="rounded border border-line bg-surface px-1 font-mono text-[11px]">{data.file}</code>). The oracle
        method wins only on the transient-recovery decoys; the naive baseline's overall score is partly a design artifact
        of the base set — both numbers are shown, not one.
      </p>
      <div className="grid max-w-md grid-cols-2 gap-3">
        <StatCard label="Oracle-guided backtracking" value={`${oracle.step_accuracy_pct}% step`} sub={`${oracle.mode_accuracy_pct}% mode`} />
        <StatCard label="Naive earliest-error baseline" value={`${naive.step_accuracy_pct}% step`} sub={`${naive.mode_accuracy_pct}% mode`} />
      </div>
    </div>
  );
}