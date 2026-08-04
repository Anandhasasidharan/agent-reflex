import { NavLink, Outlet } from "react-router";
import { useAuth } from "../lib/auth";

const navItems = [
  { to: "/overview", label: "Overview" },
  { to: "/sessions", label: "Sessions" },
];

/** Navigation reads like a trace timeline: a vertical baseline with tick
 * marks at each stop. The active stop is lit in evidence, everything else
 * stays at line strength. Collapses to a top bar on narrow screens. */
function NavList() {
  return (
    <nav aria-label="Main" className="relative pl-5">
      <span aria-hidden className="absolute bottom-1 left-[5px] top-1 w-px bg-line" />
      <ul className="space-y-1">
        {[...navItems, { to: "/settings", label: "Settings" }].map((item) => (
          <li key={item.to}>
            <NavLink
              to={item.to}
              className={({ isActive }) =>
                `group relative block rounded px-3 py-1.5 text-sm ${
                  isActive ? "bg-surface text-ink" : "text-ink-dim hover:text-ink"
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <span
                    aria-hidden
                    className={`absolute -left-5 top-1/2 h-3 w-[3px] -translate-y-1/2 rounded-full transition-colors ${
                      isActive ? "bg-evidence shadow-[0_0_8px_rgba(94,234,212,0.6)]" : "bg-line group-hover:bg-ink-dim/60"
                    }`}
                  />
                  {item.label}
                </>
              )}
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}

export function Layout() {
  const { readKey, writeKey, clearKeys } = useAuth();

  return (
    <div className="flex min-h-screen">
      <aside className="hidden w-52 shrink-0 flex-col border-r border-line bg-surface/40 lg:flex">
        <div className="flex items-center gap-2.5 border-b border-line px-4 py-4">
          <span aria-hidden className="grid size-7 place-items-center rounded-sm border border-signal/50 text-[13px] font-bold text-signal">
            A
          </span>
          <div className="leading-tight">
            <p className="text-sm font-semibold tracking-tight text-ink">
              Agent<span className="text-evidence">Reflex</span>
            </p>
            <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-faint">causal console</p>
          </div>
        </div>

        <div className="flex-1 px-4 py-4">
          <NavList />
        </div>

        <div className="border-t border-line px-4 py-3 text-xs">
          {readKey ? (
            <div className="flex items-center justify-between gap-2">
              <span className="flex items-center gap-1.5 text-ink-faint">
                <span aria-hidden className="size-1.5 rounded-full bg-evidence" />
                {writeKey ? "read + write" : "read only"}
              </span>
              <button
                type="button"
                onClick={clearKeys}
                className="rounded border border-line px-2 py-1 text-ink-dim hover:border-ink-dim hover:text-ink"
              >
                Sign out
              </button>
            </div>
          ) : (
            <span className="flex items-center gap-1.5 text-degraded">
              <span aria-hidden className="size-1.5 rounded-full bg-degraded" />
              not authenticated
            </span>
          )}
        </div>
      </aside>

      <div className="min-w-0 flex-1">
        <header className="border-b border-line bg-surface/40 lg:hidden">
          <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-4 px-4 py-3">
            <span className="text-sm font-bold tracking-tight text-ink">
              Agent<span className="text-evidence">Reflex</span>
            </span>
            <NavList />
            {readKey ? (
              <button
                type="button"
                onClick={clearKeys}
                className="ml-auto rounded border border-line px-2 py-1 text-xs text-ink-dim hover:border-ink-dim hover:text-ink"
              >
                Sign out
              </button>
            ) : (
              <span className="ml-auto text-xs text-degraded">not authenticated</span>
            )}
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-4 py-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
