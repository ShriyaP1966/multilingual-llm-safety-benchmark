/**
 * history.ts
 *
 * Browser-local evaluation history. Stored ONLY in the current
 * browser via localStorage — never sent to the backend, never
 * synced, no account, no database. Every read and write is wrapped
 * so a private window or blocked storage degrades gracefully to an
 * in-memory list for the session.
 */

import { useCallback, useEffect, useState } from "react";
import type { EvaluateResult } from "../api";

const KEY = "clvpi.history.v1";
const LIMIT = 40; // keep the most recent 40 evaluations

export interface HistoryEntry {
  id: string;
  at: number; // epoch ms
  text: string; // the evaluated response (kept local only)
  result: EvaluateResult;
}

function read(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as HistoryEntry[]) : [];
  } catch {
    return [];
  }
}

function write(entries: HistoryEntry[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(entries.slice(0, LIMIT)));
  } catch {
    /* storage unavailable — session-only, nothing to persist */
  }
}

/**
 * React hook exposing the local history plus add / clear actions.
 * Falls back to session-only memory when storage throws.
 */
export function useHistory() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);

  useEffect(() => {
    setEntries(read());
  }, []);

  const add = useCallback((text: string, result: EvaluateResult) => {
    setEntries((current) => {
      const entry: HistoryEntry = {
        id:
          (globalThis.crypto?.randomUUID?.() as string) ??
          `${Date.now()}-${Math.random().toString(36).slice(2)}`,
        at: Date.now(),
        text,
        result,
      };
      const next = [entry, ...current].slice(0, LIMIT);
      write(next);
      return next;
    });
  }, []);

  const clear = useCallback(() => {
    setEntries([]);
    try {
      localStorage.removeItem(KEY);
    } catch {
      /* ignore */
    }
  }, []);

  return { entries, add, clear, limit: LIMIT };
}

/** Human-friendly relative time, e.g. "3 min ago". */
export function relativeTime(at: number): string {
  const secs = Math.round((Date.now() - at) / 1000);
  if (secs < 45) return "just now";
  if (secs < 90) return "1 min ago";
  const mins = Math.round(secs / 60);
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs} hr${hrs === 1 ? "" : "s"} ago`;
  const days = Math.round(hrs / 24);
  if (days < 7) return `${days} day${days === 1 ? "" : "s"} ago`;
  return new Date(at).toLocaleDateString();
}
