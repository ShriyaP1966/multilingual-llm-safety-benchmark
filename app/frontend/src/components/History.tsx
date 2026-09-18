import { useMemo, useState } from "react";
import type { HistoryEntry } from "../lib/history";
import { classLabel } from "../lib/labels";
import ResultView from "./ResultView";

/**
 * History — browser-local evaluation log with search and filters.
 * Reads from localStorage only (lib/history.ts); nothing is sent to
 * the backend. Entries are grouped by day; long response text is
 * clamped so it can never determine the page width.
 */

function dayLabel(at: number): string {
  const d = new Date(at);
  const today = new Date();
  const y = new Date();
  y.setDate(today.getDate() - 1);
  const same = (a: Date, b: Date) =>
    a.getFullYear() === b.getFullYear() &&
    a.getMonth() === b.getMonth() &&
    a.getDate() === b.getDate();
  if (same(d, today)) return "Today";
  if (same(d, y)) return "Yesterday";
  return d.toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
  });
}

const clockTime = (at: number) =>
  new Date(at).toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });

interface Props {
  entries: HistoryEntry[];
  onClear: () => void;
  limit: number;
  onGoToEvaluator: () => void;
}

export default function History({
  entries,
  onClear,
  limit,
  onGoToEvaluator,
}: Props) {
  const [openId, setOpenId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [cls, setCls] = useState("all");
  const [lang, setLang] = useState("all");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return entries.filter((e) => {
      if (cls !== "all" && e.result.predicted_label !== cls) return false;
      if (lang !== "all" && e.result.language !== lang) return false;
      if (q && !e.text.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [entries, query, cls, lang]);

  const groups = useMemo(() => {
    const map = new Map<string, HistoryEntry[]>();
    for (const e of filtered) {
      const k = dayLabel(e.at);
      (map.get(k) ?? map.set(k, []).get(k)!).push(e);
    }
    return [...map.entries()];
  }, [filtered]);

  const open = entries.find((e) => e.id === openId) ?? null;

  // ---------- empty state ----------
  if (entries.length === 0) {
    return (
      <div className="stack">
        <div>
          <h2 className="section">Evaluation history</h2>
          <p className="section-note" style={{ marginBottom: 0 }}>
            Stored locally in this browser.
          </p>
        </div>
        <div className="card">
          <div className="empty">
            <div className="ei">◷</div>
            <p style={{ margin: 0, fontWeight: 600, color: "var(--ink-2)", fontSize: 16 }}>
              No evaluations yet
            </p>
            <p style={{ margin: "8px 0 16px", fontSize: 14 }}>
              Your evaluated responses will appear here.
            </p>
            <button className="btn btn-primary btn-sm" onClick={onGoToEvaluator}>
              Go to Evaluator
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="stack">
      <div className="card">
        <div className="head-row">
          <div>
            <h2 className="section">Evaluation history</h2>
            <p className="section-note" style={{ marginBottom: 0 }}>
              Your most recent {limit} evaluations. Click any entry to reopen
              its result.
            </p>
          </div>
          <button className="btn btn-ghost btn-sm" onClick={onClear}>
            Clear history
          </button>
        </div>

        <div className="privacy">
          <span className="privacy-tag">Stored locally</span>
          <span>
            Your history is stored only in this browser and is not sent to the
            server or shared across devices.
          </span>
        </div>

        <div className="filters">
          <input
            className="search"
            type="search"
            aria-label="Search response text"
            placeholder="Search response text…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <select
            aria-label="Filter by class"
            value={cls}
            onChange={(e) => setCls(e.target.value)}
          >
            <option value="all">All classes</option>
            <option value="COMPLIANCE">Compliance</option>
            <option value="NON_COMPLIANCE">Non-compliance</option>
            <option value="REFUSAL">Refusal</option>
          </select>
          <select
            aria-label="Filter by language"
            value={lang}
            onChange={(e) => setLang(e.target.value)}
          >
            <option value="all">All languages</option>
            <option value="en">English</option>
            <option value="hi">Hindi</option>
            <option value="mr">Marathi</option>
          </select>
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="card">
          <div className="empty" style={{ padding: "32px 20px" }}>
            <p style={{ margin: 0, fontSize: 14 }}>
              No entries match the current filters.
            </p>
          </div>
        </div>
      ) : (
        groups.map(([day, items]) => (
          <div key={day}>
            <div className="day-head">{day}</div>
            <div className="hist-list">
              {items.map((e) => (
                <button
                  key={e.id}
                  className="hist-card"
                  onClick={() => setOpenId((c) => (c === e.id ? null : e.id))}
                  aria-expanded={openId === e.id}
                >
                  <div className="hist-top">
                    <span className="hist-time">{clockTime(e.at)}</span>
                    <span className={`pill ${e.result.predicted_label}`}>
                      {classLabel(e.result.predicted_label)}
                    </span>
                    <span className="hist-sub">
                      {e.result.language_name} · {e.result.model} ·{" "}
                      {e.result.response_characters.toLocaleString()} chars
                    </span>
                  </div>
                  <div className="hist-preview">{e.text}</div>
                  <div className="hist-foot">
                    <span className="view">
                      {openId === e.id ? "Hide result" : "View result →"}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        ))
      )}

      {open && (
        <div className="card">
          <div className="head-row" style={{ marginBottom: 16 }}>
            <h3 style={{ margin: 0 }}>
              Reopened evaluation ·{" "}
              <span
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: 12.5,
                  color: "var(--ink-3)",
                  fontWeight: 500,
                }}
              >
                {new Date(open.at).toLocaleString()}
              </span>
            </h3>
            <button
              className="btn btn-ghost btn-sm"
              onClick={() => setOpenId(null)}
            >
              Close
            </button>
          </div>
          <div className="reopened-text">{open.text}</div>
          <div style={{ marginTop: 16 }}>
            <ResultView result={open.result} defaultOpenDetails />
          </div>
        </div>
      )}
    </div>
  );
}
