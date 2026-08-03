import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useAuthRedirect } from "../App";
import { EmptyState, ErrorState, Loading } from "../components/Async";
import type { SessionSummary } from "../lib/types";

const PAGE_SIZE = 20;

export function SessionsPage() {
  const { readKey } = useAuth();
  const redirectOnAuthError = useAuthRedirect();

  const [items, setItems] = useState<SessionSummary[] | null>(null);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [agentId, setAgentId] = useState("");
  const [failureType, setFailureType] = useState("");
  const [since, setSince] = useState("");
  const [until, setUntil] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(
    async (nextOffset: number, filters: { agentId: string; failureType: string; since: string; until: string }) => {
      if (!readKey) return;
      setLoading(true);
      setError(null);
      try {
        const data = await api.listSessions(
          {
            limit: PAGE_SIZE,
            offset: nextOffset,
            agent_id: filters.agentId || undefined,
            failure_type: filters.failureType || undefined,
            since: filters.since || undefined,
            until: filters.until || undefined,
          },
          readKey,
        );
        setItems(data.items);
        setTotal(data.total);
        setOffset(nextOffset);
      } catch (err) {
        if (redirectOnAuthError(err)) return;
        setError(err instanceof Error ? err.message : "Failed to load sessions.");
      } finally {
        setLoading(false);
      }
    },
    [readKey, redirectOnAuthError],
  );

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- data fetch on mount; setState happens post-await
    void load(0, { agentId, failureType, since, until });
  }, [load, agentId, failureType, since, until]);

  const pageCount = Math.ceil(total / PAGE_SIZE);
  const page = Math.floor(offset / PAGE_SIZE);

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-xl font-semibold">Sessions</h1>
        <p className="text-sm text-slate-400">{total} session(s) — filtered by the controls below.</p>
      </header>

      <form
        className="flex flex-wrap items-end gap-3 rounded-lg border border-slate-800 bg-slate-900/60 p-3 text-sm"
        onSubmit={(e) => {
          e.preventDefault();
          void load(0, { agentId, failureType, since, until });
        }}
      >
        <label className="block">
          <span className="mb-1 block text-xs text-slate-500">Agent ID</span>
          <input
            value={agentId}
            onChange={(e) => setAgentId(e.target.value)}
            placeholder="e.g. otlp_receiver"
            className="rounded border border-slate-700 bg-slate-950 px-2 py-1.5 text-slate-100 focus:border-emerald-500 focus:outline-none"
          />
        </label>
        <label className="block">
          <span className="mb-1 block text-xs text-slate-500">Failure type</span>
          <input
            value={failureType}
            onChange={(e) => setFailureType(e.target.value)}
            placeholder="e.g. infra_unknown"
            className="rounded border border-slate-700 bg-slate-950 px-2 py-1.5 text-slate-100 focus:border-emerald-500 focus:outline-none"
          />
        </label>
        <label className="block">
          <span className="mb-1 block text-xs text-slate-500">From</span>
          <input
            type="datetime-local"
            value={since}
            onChange={(e) => setSince(e.target.value)}
            className="rounded border border-slate-700 bg-slate-950 px-2 py-1.5 text-slate-100 focus:border-emerald-500 focus:outline-none"
          />
        </label>
        <label className="block">
          <span className="mb-1 block text-xs text-slate-500">Until</span>
          <input
            type="datetime-local"
            value={until}
            onChange={(e) => setUntil(e.target.value)}
            className="rounded border border-slate-700 bg-slate-950 px-2 py-1.5 text-slate-100 focus:border-emerald-500 focus:outline-none"
          />
        </label>
        <button
          type="submit"
          className="rounded bg-slate-700 px-3 py-1.5 text-slate-100 hover:bg-slate-600"
        >
          Apply
        </button>
      </form>

      {error && <ErrorState message={error} retry={() => void load(offset, { agentId, failureType, since, until })} />}

      {loading ? (
        <Loading label="Loading sessions…" />
      ) : items && items.length === 0 ? (
        <EmptyState
          title="No sessions match"
          hint="Ingest a trace via demo/reference_agent.py (through the OTLP path) to see it here."
        />
      ) : (
        items && (
          <div className="overflow-x-auto rounded-lg border border-slate-800">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-900/80 text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="px-3 py-2 font-medium">Session</th>
                  <th className="px-3 py-2 font-medium">Agent</th>
                  <th className="px-3 py-2 font-medium">Failure type</th>
                  <th className="px-3 py-2 font-medium">Cause</th>
                  <th className="px-3 py-2 text-right font-medium">CRS</th>
                  <th className="px-3 py-2 font-medium">Created</th>
                </tr>
              </thead>
              <tbody>
                {items.map((s) => (
                  <tr key={s.session_id} className="border-t border-slate-800/60 hover:bg-slate-900/50">
                    <td className="px-3 py-2">
                      <Link
                        to={`/sessions/${encodeURIComponent(s.session_id)}`}
                        className="font-mono text-emerald-400 hover:underline"
                      >
                        {shortId(s.session_id)}
                      </Link>
                    </td>
                    <td className="px-3 py-2 font-mono text-xs text-slate-400">{s.agent_id}</td>
                    <td className="px-3 py-2">
                      {s.failure_type ? (
                        <span className="rounded bg-red-950/60 px-1.5 py-0.5 font-mono text-[11px] text-red-300">
                          {s.failure_type}
                        </span>
                      ) : (
                        <span className="text-slate-600">—</span>
                      )}
                    </td>
                    <td className="max-w-48 truncate px-3 py-2 font-mono text-xs text-slate-500">{s.cause_node_id}</td>
                    <td className="px-3 py-2 text-right text-slate-300">
                      {s.causal_responsibility_score > 0 ? s.causal_responsibility_score.toFixed(2) : "—"}
                    </td>
                    <td className="px-3 py-2 text-xs text-slate-500">{formatDate(s.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      )}

      {total > PAGE_SIZE && (
        <nav aria-label="Pagination" className="flex items-center gap-2 text-sm">
          <button
            type="button"
            disabled={page === 0}
            onClick={() => void load((page - 1) * PAGE_SIZE, { agentId, failureType, since, until })}
            className="rounded border border-slate-700 px-3 py-1 text-slate-300 hover:bg-slate-900 disabled:opacity-40"
          >
            Previous
          </button>
          <span className="text-slate-500">
            page {page + 1} / {pageCount}
          </span>
          <button
            type="button"
            disabled={page + 1 >= pageCount}
            onClick={() => void load((page + 1) * PAGE_SIZE, { agentId, failureType, since, until })}
            className="rounded border border-slate-700 px-3 py-1 text-slate-300 hover:bg-slate-900 disabled:opacity-40"
          >
            Next
          </button>
        </nav>
      )}
    </div>
  );
}

function shortId(id: string): string {
  return id.length > 16 ? `${id.slice(0, 12)}…` : id;
}

function formatDate(raw: string | null): string {
  if (!raw) return "—";
  const d = new Date(raw);
  if (Number.isNaN(d.getTime())) return raw;
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}
