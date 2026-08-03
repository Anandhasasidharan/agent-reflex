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
        <h1 className="text-xl font-semibold">Fleet overview</h1>
        <p className="text-sm text-slate-400">
          Real data from the backend API — no placeholders. Polled every 30 seconds.
        </p>
      </header>

      <Panel title="Fleet health" actions={<span className="text-xs text-slate-500">{sortedAgents.length} agents</span>}>
        {sortedAgents.length === 0 ? (
          <EmptyState title="No agents yet" hint="Ingest a trace (e.g. demo/reference_agent.py) to populate this table." />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-slate-800 text-xs uppercase tracking-wide text-slate-500">
                  <th className="py-2 pr-4 font-medium">Agent</th>
                  <th className="py-2 pr-4 font-medium">Health</th>
                  <th className="py-2 pr-4 font-medium">Reliability</th>
                  <th className="py-2 pr-4 font-medium">Trend</th>
                  <th className="py-2 font-medium">Sessions</th>
                </tr>
              </thead>
              <tbody>
                {sortedAgents.map((agent) => (
                  <tr key={agent.agent_id} className="border-b border-slate-800/60">
                    <td className="py-2 pr-4 font-mono text-slate-200">{agent.agent_id}</td>
                    <td className="py-2 pr-4">
                      <HealthBadge health={agent.health} />
                    </td>
                    <td className="py-2 pr-4">
                      {agent.score === null ? (
                        <span className="text-slate-500">no data</span>
                      ) : (
                        <span className={scoreColor(agent.score)}>{agent.score.toFixed(3)}</span>
                      )}
                    </td>
                    <td className="py-2 pr-4">
                      <div className="flex items-center gap-2">
                        <Sparkline values={agent.history} />
                        <span className="text-xs text-slate-500">{trendArrow(agent.trend)}</span>
                      </div>
                    </td>
                    <td className="py-2 text-slate-400">{agent.n_sessions}</td>
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
                  <tr className="border-b border-slate-800 text-slate-500">
                    <th className="py-1 pr-3 font-medium">Playbook</th>
                    <th className="py-1 pr-3 font-medium">Selector</th>
                    <th className="py-1 pr-3 text-right font-medium">Total</th>
                    <th className="py-1 text-right font-medium">Success rate</th>
                  </tr>
                </thead>
                <tbody>
                  {breakdown.map((row, i) => (
                    <tr key={`${row.playbook}-${row.selector}-${i}`} className="border-b border-slate-800/60">
                      <td className="py-1 pr-3 font-mono text-slate-300">{row.playbook}</td>
                      <td className="py-1 pr-3 text-slate-400">{row.selector}</td>
                      <td className="py-1 pr-3 text-right text-slate-400">{row.total}</td>
                      <td className="py-1 text-right text-slate-300">{pct(row.success_rate)}</td>
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

      <Panel title="Recent failures" actions={<span className="text-xs text-slate-500">{failures.length} with attributed failure</span>}>
        {failures.length === 0 ? (
          <EmptyState title="No failures recorded" hint="When a session is attributed with a failure type it shows up here, linking into its causal graph." />
        ) : (
          <ul className="divide-y divide-slate-800/60">
            {failures.slice(0, 10).map((s) => (
              <li key={s.session_id}>
                <Link to={`/sessions/${encodeURIComponent(s.session_id)}`} className="flex flex-wrap items-center gap-3 py-2 hover:bg-slate-900/60">
                  <span className="rounded bg-red-950/60 px-1.5 py-0.5 font-mono text-[11px] text-red-300">{s.failure_type}</span>
                  <span className="min-w-0 flex-1 truncate text-sm text-slate-200">
                    {s.task_description || "(no task description)"}
                  </span>
                  <span className="font-mono text-xs text-slate-500">{shortId(s.session_id)}</span>
                  <span className="text-xs text-slate-500">{formatDate(s.created_at)}</span>
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
  if (score >= 0.7) return "text-emerald-400";
  if (score >= 0.4) return "text-amber-400";
  return "text-red-400";
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
    <div className="rounded border border-slate-800 bg-slate-950/50 p-3">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-1 text-xl font-semibold text-slate-100">{value}</p>
      <p className="text-xs text-slate-500">{sub}</p>
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
            <th className="pr-2 text-left font-medium text-slate-500">failure type</th>
            {dates.map((d) => (
              <th key={d} className="px-1 pb-1 text-right font-medium text-slate-600" title={d}>
                {d.slice(8, 10)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {types.map((t) => (
            <tr key={t}>
              <td className="max-w-40 truncate pr-2 font-mono text-[11px] text-slate-400" title={t}>
                {t}
              </td>
              {dates.map((d) => {
                const count = byKey.get(`${t}|${d}`) ?? 0;
                return (
                  <td key={d} className="px-1 py-0.5 text-center">
                    <span
                      title={`${t} on ${d}: ${count}`}
                      className={`inline-block h-4 w-4 rounded-sm ${count === 0 ? "bg-slate-900" : "bg-red-600"} `}
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
      <p className="mb-3 text-xs text-slate-400">
        Side-by-side step attribution on {data.total_scenarios ?? "?"} synthetic scenarios (file:{" "}
        <code className="rounded bg-slate-800 px-1">{data.file}</code>). The oracle method wins only on the
        transient-recovery decoys; the naive baseline's overall score is partly a design artifact of the base set —
        both numbers are shown, not one.
      </p>
      <div className="grid max-w-md grid-cols-2 gap-3">
        <StatCard label="Oracle-guided backtracking" value={`${oracle.step_accuracy_pct}% step`} sub={`${oracle.mode_accuracy_pct}% mode`} />
        <StatCard label="Naive earliest-error baseline" value={`${naive.step_accuracy_pct}% step`} sub={`${naive.mode_accuracy_pct}% mode`} />
      </div>
    </div>
  );
}
