import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";

/**
 * API key handling.
 *
 * Keys are treated as credentials: kept in memory and mirrored to
 * sessionStorage (gone when the tab closes — deliberately NOT localStorage,
 * so a new browser session starts at the settings screen again). A separate
 * optional write key is only requested where the UI needs it (recovery
 * feedback) — a read-only session never asks for write access.
 */

const READ_KEY_STORAGE = "agent_reflex_read_key";
const WRITE_KEY_STORAGE = "agent_reflex_write_key";

interface AuthState {
  readKey: string | null;
  writeKey: string | null;
  authError: string | null;
  setReadKey: (key: string) => void;
  setWriteKey: (key: string) => void;
  clearKeys: () => void;
  notifyAuthError: (message: string) => void;
  clearAuthError: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [readKey, setReadKeyState] = useState<string | null>(() => sessionStorage.getItem(READ_KEY_STORAGE));
  const [writeKey, setWriteKeyState] = useState<string | null>(() => sessionStorage.getItem(WRITE_KEY_STORAGE));
  const [authError, setAuthError] = useState<string | null>(null);

  const setReadKey = useCallback((key: string) => {
    sessionStorage.setItem(READ_KEY_STORAGE, key);
    setReadKeyState(key);
  }, []);

  const setWriteKey = useCallback((key: string) => {
    sessionStorage.setItem(WRITE_KEY_STORAGE, key);
    setWriteKeyState(key);
  }, []);

  const clearKeys = useCallback(() => {
    sessionStorage.removeItem(READ_KEY_STORAGE);
    sessionStorage.removeItem(WRITE_KEY_STORAGE);
    setReadKeyState(null);
    setWriteKeyState(null);
  }, []);

  const notifyAuthError = useCallback((message: string) => setAuthError(message), []);
  const clearAuthError = useCallback(() => setAuthError(null), []);

  const value = useMemo(
    () => ({
      readKey,
      writeKey,
      authError,
      setReadKey,
      setWriteKey,
      clearKeys,
      notifyAuthError,
      clearAuthError,
    }),
    [readKey, writeKey, authError, setReadKey, setWriteKey, clearKeys, notifyAuthError, clearAuthError],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
  return ctx;
}
