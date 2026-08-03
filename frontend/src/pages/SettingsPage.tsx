import { useState } from "react";
import type { FormEvent } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { useAuth } from "../lib/auth";
import { api } from "../lib/api";

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
      <h1 className="mb-1 text-xl font-semibold">API key</h1>
      <p className="mb-6 text-sm text-slate-400">
        This dashboard reads session, causation, and reliability data from your AgentReflex backend. It needs a{" "}
        <code className="rounded bg-slate-800 px-1">read</code>-scoped API key.
      </p>

      {authError && (
        <div role="alert" className="mb-4 rounded-md border border-amber-800 bg-amber-950/40 p-3 text-sm text-amber-200">
          {authError} — your key was invalid or missing, so you were sent here.
          <button type="button" className="ml-2 underline" onClick={clearAuthError}>
            dismiss
          </button>
        </div>
      )}

      <form onSubmit={verifyAndSave}
        className="space-y-4 rounded-lg border border-slate-800 bg-slate-900/60 p-4">
        <label className="block text-sm">
          <span className="mb-1 block font-medium text-slate-300">Read key</span>
          <input
            type="password"
            value={readInput}
            onChange={(e) => setReadInput(e.target.value)}
            placeholder="paste read-scoped key"
            className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 focus:border-emerald-500 focus:outline-none"
          />
        </label>

        <details className="text-sm text-slate-400">
          <summary className="cursor-pointer text-slate-300">No key? Create one on your server</summary>
          <pre className="mt-3 overflow-x-auto rounded bg-slate-950 p-3 text-xs text-slate-300">{`# on the machine running AgentReflex
python -m agent_reflex.api.auth create frontend --scope=read

# in Docker:
docker compose exec app python -m agent_reflex.api.auth create frontend --scope=read

# it prints the key exactly once — paste it here.`}</pre>
        </details>

        <label className="block text-sm">
          <span className="mb-1 block font-medium text-slate-300">
            Write key <span className="font-normal text-slate-500">(optional — recovery feedback only)</span>
          </span>
          <input
            type="password"
            value={writeInput}
            onChange={(e) => setWriteInput(e.target.value)}
            placeholder="e.g. write-scoped key (only needed to submit feedback)"
            className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 focus:border-emerald-500 focus:outline-none"
          />
        </label>

        <button
          type="submit"
          disabled={verifying}
          className="rounded bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
        >
          {verifying ? "Verifying…" : "Save & continue"}
        </button>

        {message && (
          <p role="alert" className="text-sm text-red-300">
            {message}
          </p>
        )}
      </form>
    </div>
  );
}