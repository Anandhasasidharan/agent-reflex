import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useAuthRedirect } from "../App";
import { EmptyState, ErrorState, Loading } from "../components/Async";
import type { SessionSummary } from "../lib/types";

const PAGE_SIZE = 20;

const inputClass =
  "rounded border border-line bg-bg px-2 py-1.5 text-ink focus:border-evidence focus:outline-none focus:ring-1 focus:ring-evidence/50";

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
        <h1 className="text-xl font-semibold tracking-tight">Sessions</h1>
        <p className="text-sm text-ink-dim">{total} session(s) — filtered by the controls below.</p>
      </header>

      <form
        className="flex flex-wrap items-end gap-3 rounded-lg border border-line bg-surface/70 p-3 text-sm"
        onSubmit={(e) => {
          e.preventDefault();
          void load(0, { agentId, failureType, since, until });
        }}
      >
        <label className="block">
          <span className="mb-1 block text-xs text-ink-faint">Agent ID</span>
          <input
            value={agentId}
            onChange={(e) => setAgentId(e.target.value)}
            placeholder="e.g. otlp_receiver"
            className={inputClass}
          />
        </label>
        <label className="block">
          <span className="mb-1 block text-xs text-ink-faint">Failure type</span>
          <input
            value={failureType}
            onChange={(e) => setFailureType(e.target.value)}
            placeholder="e.g. infra_unknown"
            className={inputClass}
          />
        </label>
        <label className="block">
          <span className="mb-1 block text-xs text-ink-faint">From</span>
          <input
            type="datetime-local"
            value={since}
            onChange={(e) => setSince(e.target.value)}
            className={inputClass}
          />
        </label>
        <label className="block">
          <span className="mb-1 block text-xs text-ink-faint">Until</span>
          <input
            type="datetime-local"
            value={until}
            onChange={(e) => setUntil(e.target.value)}
            className={inputClass}
          />
        </label>
        <button
          type="submit"
          className="rounded-md border border-evidence/40 bg-evidence-dim px-3 py-1.5 text-sm text-evidence hover:bg-evidence hover:text-bg"
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
          <div className="overflow-x-auto rounded-lg border border-line">
            <table className="w-full text-left text-sm">
              <thead className="bg-surface text-xs uppercase tracking-wide text-ink-faint">
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
                  <tr key={s.session_id} className="border-t border-line hover:bg-surface/60">
                    <td className="px-3 py-2">
                      <Link
                        to={`/sessions/${encodeURIComponent(s.session_id)}`}
                        className="font-mono text-evidence hover:underline"
                      >
                        {shortId(s.session_id)}
                      </Link>
                    </td>
                    <td className="px-3 py-2 font-mono text-xs text-ink-dim">{s.agent_id}</td>
                    <td className="px-3 py-2">
                      {s.failure_type ? (
                        <span className="rounded bg-signal-dim px-1.5 py-0.5 font-mono text-[11px] text-signal">
                          {s.failure_type}
                        </span>
                      ) : (
                        <span className="text-ink-faint">—</span>
                      )}
                    </td>
                    <td className="max-w-48 truncate px-3 py-2 font-mono text-xs text-ink-faint">{s.cause_node_id}</td>
                    <td className="px-3 py-2 text-right font-mono text-ink-dim">
                      {s.causal_responsibility_score > 0 ? s.causal_responsibility_score.toFixed(2) : "—"}
                    </td>
                    <td className="px-3 py-2 text-xs text-ink-faint">{formatDate(s.created_at)}</td>
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
            className="rounded-md border border-line px-3 py-1 text-ink-dim hover:border-ink-dim hover:text-ink disabled:opacity-40"
          >
            Previous
          </button>
          <span className="text-ink-faint">
            page {page + 1} / {pageCount}
          </span>
          <button
            type="button"
            disabled={page + 1 >= pageCount}
            onClick={() => void load((page + 1) * PAGE_SIZE, { agentId, failureType, since, until })}
            className="rounded-md border border-line px-3 py-1 text-ink-dim hover:border-ink-dim hover:text-ink disabled:opacity-40"
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