import type { ReactNode } from "react";

/** Shared loading / error / empty states — every view uses these so no
 * fetch failure or empty result ever renders as a blank screen. */

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div role="status" className="flex items-center gap-2 py-8 text-sm text-ink-dim">
      <span className="inline-block size-3 animate-spin rounded-full border-2 border-line border-t-evidence" />
      {label}
    </div>
  );
}

export function ErrorState({ message, retry }: { message: string; retry?: () => void }) {
  return (
    <div role="alert" className="rounded-lg border border-signal/30 bg-signal-dim p-4 text-sm">
      <p className="font-medium text-signal">Something went wrong</p>
      <p className="mt-1 text-ink-dim">{message}</p>
      {retry && (
        <button
          type="button"
          onClick={retry}
          className="mt-3 rounded border border-signal/40 bg-signal-dim px-3 py-1 text-xs text-signal hover:bg-signal hover:text-bg"
        >
          Retry
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-line bg-surface/50 p-6 text-center text-sm text-ink-dim">
      <p className="font-medium text-ink">{title}</p>
      {hint && <p className="mt-1 text-ink-faint">{hint}</p>}
    </div>
  );
}

export function Panel({ title, children, actions }: { title: string; children: ReactNode; actions?: ReactNode }) {
  return (
    <section className="rounded-lg border border-line bg-surface/70 p-4">
      <header className="mb-3 flex items-center justify-between gap-2">
        <h2 className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-ink-dim">
          <span aria-hidden className="inline-block h-3 w-px bg-evidence/70" />
          {title}
        </h2>
        {actions}
      </header>
      {children}
    </section>
  );
}

export function HealthBadge({ health }: { health: string }) {
  const styles: Record<string, string> = {
    healthy: "border-evidence/40 bg-evidence-dim text-evidence",
    degraded: "border-degraded/40 bg-degraded-dim text-degraded",
    critical: "border-signal/40 bg-signal-dim text-signal",
    unknown: "border-line bg-surface text-ink-faint",
  };
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs ${
        styles[health] ?? styles.unknown
      }`}
    >
      <span aria-hidden className="size-1.5 rounded-full bg-current opacity-80" />
      {health}
    </span>
  );
}
