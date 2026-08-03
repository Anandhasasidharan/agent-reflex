import { NavLink, Outlet } from "react-router-dom";
import { useAuth } from "../lib/auth";

const navItems = [
  { to: "/overview", label: "Overview" },
  { to: "/sessions", label: "Sessions" },
];

export function Layout() {
  const { readKey, writeKey, clearKeys } = useAuth();

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-800 bg-slate-950/90">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-4 px-4 py-3">
          <span className="text-sm font-bold tracking-tight text-slate-100">
            Agent<span className="text-emerald-400">Reflex</span>
          </span>
          <nav aria-label="Main" className="flex gap-1">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `rounded px-3 py-1.5 text-sm ${
                    isActive ? "bg-slate-800 text-white" : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
            <NavLink
              to="/settings"
              className={({ isActive }) =>
                `rounded px-3 py-1.5 text-sm ${
                  isActive ? "bg-slate-800 text-white" : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                }`
              }
            >
              Settings
            </NavLink>
          </nav>
          <div className="ml-auto flex items-center gap-3 text-xs text-slate-500">
            {readKey ? (
              <>
                <span className="hidden sm:inline">read key set</span>
                {writeKey && <span className="hidden sm:inline text-amber-500/80">write key set</span>}
                <button
                  type="button"
                  onClick={clearKeys}
                  className="rounded border border-slate-700 px-2 py-1 text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                >
                  Sign out
                </button>
              </>
            ) : (
              <span className="text-amber-500/80">not authenticated</span>
            )}
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-6">
        <Outlet />
      </main>
    </div>
  );
}
