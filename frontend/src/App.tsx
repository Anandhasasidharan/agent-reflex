import { BrowserRouter, Navigate, Route, Routes, useNavigate } from "react-router";
import type { ReactNode } from "react";
import { AuthProvider, useAuth } from "./lib/auth";
import { Layout } from "./components/Layout";
import { SettingsPage } from "./pages/SettingsPage";
import { OverviewPage } from "./pages/OverviewPage";
import { SessionsPage } from "./pages/SessionsPage";
import { SessionDetailPage } from "./pages/SessionDetailPage";
import { AuthError } from "./lib/api";

function RequireAuth({ children }: { children: ReactNode }) {
  const { readKey } = useAuth();
  if (!readKey) {
    return <Navigate to="/settings" replace />;
  }
  return <>{children}</>;
}

/** Call this when an API call raises AuthError to bounce to settings. */
export function useAuthRedirect() {
  const { notifyAuthError } = useAuth();
  const navigate = useNavigate();
  return (err: unknown) => {
    if (err instanceof AuthError) {
      notifyAuthError(err.message);
      navigate("/settings");
      return true;
    }
    return false;
  };
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/settings" element={<SettingsPage />} />
            <Route
              path="/overview"
              element={
                <RequireAuth>
                  <OverviewPage />
                </RequireAuth>
              }
            />
            <Route
              path="/sessions"
              element={
                <RequireAuth>
                  <SessionsPage />
                </RequireAuth>
              }
            />
            <Route
              path="/sessions/:sessionId"
              element={
                <RequireAuth>
                  <SessionDetailPage />
                </RequireAuth>
              }
            />
            <Route path="*" element={<Navigate to="/overview" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
