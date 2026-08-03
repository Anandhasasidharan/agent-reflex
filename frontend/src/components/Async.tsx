import type { ReactNode } from "react";

/** Shared loading / error / empty states — every view uses these so no
 * fetch failure or empty result ever renders as a blank screen. */

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div role="status" className="flex items-center gap-2 py-8 text-sm text-slate-400">
      <span className="inline-block size-3 animate-spin rounded-full border-2 border-slate-500 border-t-transparent" />
      {label}
    </div>
  );
}

export function ErrorState({ message, retry }: { message: string; retry?: () => void }) {
  return (
    <div role="alert" className="rounded-md border border-red-800/60 bg-red-950/40 p-4 text-sm text-red-200">
      <p className="font-medium">Something went wrong</p>
      <p className="mt-1">{message}</p>
      {retry && (
        <button
          type="button"
          onClick={retry}
          className="mt-3 rounded border border-red-700 px-3 py-1 text-xs text-red-100 hover:bg-red-900/60"
        >
          Retry
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="rounded-md border border-slate-800 bg-slate-900/50 p-6 text-center text-sm text-slate-400">
      <p className="font-medium text-slate-300">{title}</p>
      {hint && <p className="mt-1">{hint}</p>}
    </div>
  );
}

export function Panel({ title, children, actions }: { title: string; children: ReactNode; actions?: ReactNode }) {
  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
      <header className="mb-3 flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-300">{title}</h2>
        {actions}
      </header>
      {children}
    </section>
  );
}

export function HealthBadge({ health }: { health: string }) {
  const styles: Record<string, string> = {
    healthy: "bg-emerald-900/60 text-emerald-300 border-emerald-700",
    degraded: "bg-amber-900/60 text-amber-300 border-amber-700",
    critical: "bg-red-900/60 text-red-300 border-red-700",
    unknown: "bg-slate-800 text-slate-400 border-slate-700",
  };
  return (
    <span className={`inline-block rounded-full border px-2 py-0.5 text-xs ${styles[health] ?? styles.unknown}`}>
      {health}
    </span>
  );
}
