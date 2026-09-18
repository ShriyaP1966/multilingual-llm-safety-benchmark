import { useEffect, useState } from "react";
import { api, type Health, type Meta } from "./api";
import { useHistory } from "./lib/history";
import Evaluator from "./components/Evaluator";
import History from "./components/History";
import Benchmark from "./components/Benchmark";
import About from "./components/About";
import Methodology from "./components/Methodology";

const PROJECT_TITLE =
  "Cross-Lingual Vulnerability and Prompt Injection in Low-Resource Languages";

type Tab = "evaluate" | "results" | "history" | "about" | "methodology";

const NAV: { key: Tab; label: string; icon: string; page: string }[] = [
  { key: "evaluate", label: "Evaluator", icon: "◎", page: "Safety Evaluator" },
  { key: "results", label: "Results", icon: "▤", page: "Benchmark / Results" },
  { key: "history", label: "History", icon: "◷", page: "Evaluation History" },
  { key: "about", label: "Research", icon: "❋", page: "About the Research" },
  { key: "methodology", label: "Methodology", icon: "⚗", page: "Methodology" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("evaluate");
  const [health, setHealth] = useState<Health | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);
  const history = useHistory();

  useEffect(() => {
    Promise.all([api.health(), api.meta()])
      .then(([h, m]) => {
        setHealth(h);
        setMeta(m);
      })
      .catch((e: Error) => setBootError(e.message));
  }, []);

  const apiUp = health?.status === "ok";
  const current = NAV.find((n) => n.key === tab)!;

  return (
    <div className="app">
      {/* ---------- persistent sidebar (desktop) ---------- */}
      <aside className="sidebar">
        <div className="brand">
          <div className="mark">CLVPI</div>
          <div className="sub">Safety Benchmark</div>
        </div>
        <nav aria-label="Sections">
          {NAV.map((n) => (
            <button
              key={n.key}
              onClick={() => setTab(n.key)}
              aria-current={tab === n.key ? "page" : undefined}
            >
              <span className="ic" aria-hidden="true">
                {n.icon}
              </span>
              {n.label}
              {n.key === "history" && history.entries.length > 0 && (
                <span className="badge">{history.entries.length}</span>
              )}
            </button>
          ))}
        </nav>
        <div className="foot">
          <div className="st">
            <span className={`dot ${apiUp ? "up" : "down"}`} />
            Local backend · {apiUp ? "connected" : "offline"}
          </div>
          {health && <div>device · {health.device}</div>}
          {health && (
            <div>
              models ·{" "}
              {Object.entries(health.models)
                .map(([k, v]) => `${k} ${v.available ? "✓" : "✗"}`)
                .join("  ")}
            </div>
          )}
        </div>
      </aside>

      {/* ---------- mobile top bar ---------- */}
      <div className="mobilebar">
        <div className="mrow">
          <span className="mark">CLVPI</span>
          <span style={{ fontSize: 11, opacity: 0.7 }}>
            <span className={`dot ${apiUp ? "up" : "down"}`} /> API{" "}
            {apiUp ? "on" : "off"}
          </span>
        </div>
        <div className="mtabs">
          {NAV.map((n) => (
            <button
              key={n.key}
              onClick={() => setTab(n.key)}
              aria-current={tab === n.key ? "page" : undefined}
            >
              {n.label}
              {n.key === "history" && history.entries.length > 0
                ? ` (${history.entries.length})`
                : ""}
            </button>
          ))}
        </div>
      </div>

      {/* ---------- main ---------- */}
      <div className="main">
        <div className="topbar">
          <div className="ctx">
            <div className="pg">{current.page}</div>
            <div className="pj">{PROJECT_TITLE}</div>
          </div>
          <div className="badge-r">Research Demonstration</div>
        </div>

        <div className="content">
          {bootError && (
            <div className="note caution" style={{ marginBottom: 20 }}>
              <strong>Cannot reach the backend API</strong>
              <p>{bootError}</p>
              <p>
                Start it from the repository root with{" "}
                <code>uvicorn app.backend.main:app --port 8077</code>, then
                reload this page.
              </p>
            </div>
          )}

          {tab === "evaluate" && (
            <Evaluator meta={meta} onEvaluated={history.add} />
          )}
          {tab === "results" && <Benchmark />}
          {tab === "history" && (
            <History
              entries={history.entries}
              onClear={history.clear}
              limit={history.limit}
              onGoToEvaluator={() => setTab("evaluate")}
            />
          )}
          {tab === "about" && <About />}
          {tab === "methodology" && <Methodology />}

          <footer className="page-foot">
            {PROJECT_TITLE} · Multilingual LLM Safety Evaluation Framework.
            Results are read from the frozen research outputs. Safety labels are
            evaluator-produced silver labels, not ground truth; statistical
            results are associations, not causal claims; the classifier is a
            research artifact and is not production-ready.
          </footer>
        </div>
      </div>
    </div>
  );
}
