import { useState } from "react";
import type { FormEvent } from "react";
import { Navigate, useNavigate } from "react-router";
import { useAuth } from "../lib/auth";
import { api } from "../lib/api";

const fieldClass =
  "w-full rounded border border-line bg-bg px-3 py-2 font-mono text-xs text-ink placeholder:text-ink-faint focus:border-evidence focus:outline-none focus:ring-1 focus:ring-evidence/50";

export function SettingsPage() {
  const { readKey, setReadKey, setWriteKey, authError, clearAuthError } = useAuth();
  const navigate = useNavigate();
  const [readInput, setReadInput] = useState("");
  const [writeInput, setWriteInput] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [verifying, setVerifying] = useState(false);

  if (readKey) {
    return <Navigate to="/overview" replace />;
  }

  async function verifyAndSave(e: FormEvent) {
    e.preventDefault();
    if (!readInput.trim()) {
      setMessage("Paste a read-scoped API key first.");
      return;
    }
    setVerifying(true);
    setMessage(null);
    try {
      await api.agentsStatus(readInput.trim());
      setReadKey(readInput.trim());
      if (writeInput.trim()) setWriteKey(writeInput.trim());
      navigate("/overview", { replace: true });
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "That key did not work.");
    } finally {
      setVerifying(false);
    }
  }

  return (
    <div className="mx-auto max-w-xl">
      <h1 className="mb-1 text-xl font-semibold tracking-tight">API key</h1>
      <p className="mb-6 text-sm text-ink-dim">
        This dashboard reads session, causation, and reliability data from your AgentReflex backend. It needs a{" "}
        <code className="rounded border border-line bg-surface px-1 font-mono text-xs">read</code>-scoped API key.
      </p>

      {authError && (
        <div role="alert" className="mb-4 rounded-md border border-signal/40 bg-signal-dim p-3 text-sm text-signal">
          {authError} — your key was invalid or missing, so you were sent here.
          <button type="button" className="ml-2 underline" onClick={clearAuthError}>
            dismiss
          </button>
        </div>
      )}

      <form onSubmit={verifyAndSave}
        className="space-y-4 rounded-lg border border-line bg-surface/70 p-4">
        <label className="block text-sm">
          <span className="mb-1 block font-medium text-ink-dim">Read key</span>
          <input
            type="password"
            value={readInput}
            onChange={(e) => setReadInput(e.target.value)}
            placeholder="paste read-scoped key"
            className={fieldClass}
          />
        </label>

        <details className="text-sm text-ink-faint">
          <summary className="cursor-pointer text-ink-dim hover:text-ink">No key? Create one on your server</summary>
          <pre className="mt-3 overflow-x-auto rounded-md border border-line bg-bg p-3 font-mono text-xs leading-relaxed text-ink-dim">{`# on the machine running AgentReflex
python -m agent_reflex.api.auth create frontend --scope=read

# in Docker:
docker compose exec app python -m agent_reflex.api.auth create frontend --scope=read

# it prints the key exactly once — paste it here.`}</pre>
        </details>

        <label className="block text-sm">
          <span className="mb-1 block font-medium text-ink-dim">
            Write key <span className="font-normal text-ink-faint">(optional — recovery feedback only)</span>
          </span>
          <input
            type="password"
            value={writeInput}
            onChange={(e) => setWriteInput(e.target.value)}
            placeholder="e.g. write-scoped key (only needed to submit feedback)"
            className={fieldClass}
          />
        </label>

        <button
          type="submit"
          disabled={verifying}
          className="rounded-md border border-evidence/40 bg-evidence-dim px-4 py-2 text-sm font-medium text-evidence hover:bg-evidence hover:text-bg disabled:opacity-50"
        >
          {verifying ? "Verifying…" : "Save & continue"}
        </button>

        {message && (
          <p role="alert" className="text-sm text-signal">
            {message}
          </p>
        )}
      </form>
    </div>
  );
}