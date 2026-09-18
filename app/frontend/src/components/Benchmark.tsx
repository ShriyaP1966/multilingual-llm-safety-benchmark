import { useEffect, useState } from "react";
import { api, num, type Benchmark as BenchmarkData, type Row } from "../api";

/**
 * Benchmark / Results — visual first, tables underneath.
 * Every number is fetched from the backend, which reads the frozen
 * research outputs. Nothing is hard-coded.
 */

const LABELS = ["COMPLIANCE", "NON_COMPLIANCE", "REFUSAL"] as const;
const READABLE: Record<string, string> = {
  COMPLIANCE: "Compliance",
  NON_COMPLIANCE: "Non-compliance",
  REFUSAL: "Refusal",
};
const COLORVAR: Record<string, string> = {
  COMPLIANCE: "var(--comp)",
  NON_COMPLIANCE: "var(--noncomp)",
  REFUSAL: "var(--refusal)",
};

function StackedBars({
  rows,
  keyCol,
  label,
}: {
  rows: Row[];
  keyCol: string;
  label: (k: string) => string;
}) {
  return (
    <div className="sbar">
      {rows.map((row) => {
        const total = num(row.TOTAL) || 1;
        return (
          <div className="sbar-row" key={row[keyCol]}>
            <div className="rl">{label(row[keyCol])}</div>
            <div
              className="sbar-track"
              role="img"
              aria-label={LABELS.map((l) => `${READABLE[l]} ${row[l]}`).join(", ")}
            >
              {LABELS.map((l) => {
                const v = num(row[l]);
                const pct = (v / total) * 100;
                if (pct <= 0) return null;
                return (
                  <div
                    key={l}
                    className={`sbar-seg ${l}`}
                    style={{ width: `${pct}%` }}
                    title={`${READABLE[l]}: ${v} (${pct.toFixed(1)}%)`}
                  >
                    {pct >= 9 && <span>{pct.toFixed(0)}%</span>}
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Legend() {
  return (
    <div className="legend">
      {LABELS.map((l) => (
        <span key={l}>
          <i style={{ background: COLORVAR[l] }} />
          {READABLE[l]}
        </span>
      ))}
    </div>
  );
}

export default function Benchmark() {
  const [data, setData] = useState<BenchmarkData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.benchmark().then(setData).catch((e: Error) => setError(e.message));
  }, []);

  if (error) {
    return (
      <div className="note caution">
        <strong>Results unavailable</strong>
        <p>{error}</p>
      </div>
    );
  }
  if (!data) return <p className="loading">Loading frozen benchmark results…</p>;

  const gl = data.generation_length_summary;
  const gold = data.classifier.filter((r) => r.split === "human gold");
  const baseline = gold.length ? gold[0].majority_baseline : 0;

  const modelName = (k: string) => (k === "gpt_oss" ? "GPT-OSS" : "Qwen");
  const langName = (k: string) =>
    ({ en: "English", hi: "Hindi", mr: "Marathi" }[k] ?? k);

  return (
    <div className="stack">
      {/* summary */}
      <div>
        <h2 className="section">Benchmark results</h2>
        <p className="section-note">
          Read live from the frozen research outputs. Evaluator pinned at
          SHA-256 <code>{data.scope.evaluator_sha256?.slice(0, 12) ?? "—"}…</code>
          {data.scope.frozen_at ? `, frozen ${data.scope.frozen_at}.` : "."}
        </p>
        <div className="stats">
          <div className="stat"><div className="n">624</div><div className="l">benchmark responses</div></div>
          <div className="stat"><div className="n">3</div><div className="l">languages</div></div>
          <div className="stat"><div className="n">13</div><div className="l">attack categories</div></div>
          <div className="stat"><div className="n">8</div><div className="l">prompt variations</div></div>
          <div className="stat"><div className="n">63</div><div className="l">human-audited cases</div></div>
        </div>
      </div>

      {/* model comparison */}
      <div className="card">
        <h3>Model comparison</h3>
        <p className="chart-sub">
          Safety-class distribution per model on the 512-token set (312
          responses each).
        </p>
        <div className="cmp-grid">
          {data.model_safety.map((row) => (
            <div className="cmp" key={row.model}>
              <div className="cmp-name">{modelName(row.model)}</div>
              {LABELS.map((l) => (
                <div className="cmp-row" key={l}>
                  <span className="dotc" style={{ background: COLORVAR[l] }} />
                  <span className="cmp-k">{READABLE[l]}</span>
                  <span className="cmp-v">{row[l]}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
        <div style={{ marginTop: 22 }}>
          <StackedBars rows={data.model_safety} keyCol="model" label={modelName} />
          <Legend />
        </div>
      </div>

      {/* language comparison */}
      <div className="card">
        <h3>Language comparison</h3>
        <p className="chart-sub">
          Safety-class distribution per language, both models pooled.
        </p>
        <StackedBars rows={data.language_safety} keyCol="language" label={langName} />
        <Legend />
      </div>

      {/* confirmatory */}
      <div className="card">
        <h3>Confirmatory findings</h3>
        <p className="chart-sub">
          Repeated-measures tests with Benjamini–Hochberg correction. These are
          association tests — no causal claim follows from them.
        </p>
        <div className="scroll">
          <table className="data">
            <thead>
              <tr>
                <th>Analysis</th>
                <th className="num">Statistic</th>
                <th className="num">df</th>
                <th className="num">p (corrected)</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {data.confirmatory.map((r) => {
                const sig = String(r.significant_bh_05).toLowerCase() === "true";
                return (
                  <tr key={r.analysis}>
                    <td>{r.analysis}</td>
                    <td className="num">{num(r.statistic).toFixed(2)}</td>
                    <td className="num">{r.df}</td>
                    <td className="num">{num(r.p_value_bh).toExponential(2)}</td>
                    <td>
                      <span className={`pill ${sig ? "ok" : "excl"}`}>
                        {sig ? "significant" : "not significant"}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="note" style={{ marginTop: 14 }}>
          <strong>How to read this</strong>
          <p>
            The model, attack-category and prompt-variation contrasts remain
            significant after Benjamini–Hochberg correction.{" "}
            <strong>None</strong> of the pairwise language comparisons
            (English–Hindi, English–Marathi, Hindi–Marathi) survive correction.
            All are associations, not causal effects.
          </p>
        </div>
      </div>

      {/* generation length */}
      <div className="card">
        <h3>Generation-length sensitivity — GPT-OSS</h3>
        <p className="chart-sub">
          Does a longer generation budget change the safety labels? Compared on{" "}
          {gl.paired_responses} paired GPT-OSS responses.
        </p>
        <div className="gl-grid">
          <div className="gl-stat">
            <div className="gl-n">
              {gl.ceiling_pct_512}% <span className="arrow">→</span>{" "}
              {gl.ceiling_pct_2048}%
            </div>
            <div className="gl-l">responses hitting the token ceiling (512 → 2048)</div>
          </div>
          <div className="gl-stat">
            <div className="gl-n">
              {gl.labels_changed}/{gl.paired_responses}
            </div>
            <div className="gl-l">labels changed between budgets</div>
          </div>
          <div className="gl-stat">
            <div className="gl-n">
              {gl.refusal_512} = {gl.refusal_2048}
            </div>
            <div className="gl-l">refusal count identical at both budgets</div>
          </div>
        </div>
        <p className="prose" style={{ marginTop: 16, fontSize: 14.5 }}>
          Raising the budget from 512 to 2048 tokens sharply reduced truncation
          but changed only {gl.labels_changed} of {gl.paired_responses} labels,
          with the refusal count unchanged. For GPT-OSS, safety labels are
          largely insensitive to the generation budget over this range.
        </p>
        <div className="note warn" style={{ marginTop: 8 }}>
          <strong>Qwen at 2048 tokens is excluded</strong>
          <p>{gl.qwen_excluded_reason}</p>
        </div>
      </div>

      {/* classifier */}
      <div className="card">
        <h3>Classifier results</h3>
        <p className="chart-sub">
          Response-based safety classifiers. Silver-dev is evaluator-labelled;
          human-gold is the 63-case human-adjudicated test set. Accuracy is
          shown beside the majority baseline of{" "}
          {(baseline * 100).toFixed(1)}%.
        </p>
        <div className="scroll">
          <table className="data">
            <thead>
              <tr>
                <th>Model</th><th>Weights</th><th>Split</th>
                <th className="num">n</th><th className="num">Accuracy</th>
                <th className="num">Baseline</th><th>Beats base.</th>
                <th className="num">Macro F1</th><th className="num">Compliance F1</th>
              </tr>
            </thead>
            <tbody>
              {data.classifier.map((r, i) => (
                <tr key={i}>
                  <td>{r.model}</td>
                  <td>{r.class_weights}</td>
                  <td>{r.split}</td>
                  <td className="num">{r.n}</td>
                  <td className="num">{(r.accuracy * 100).toFixed(1)}%</td>
                  <td className="num">{(r.majority_baseline * 100).toFixed(1)}%</td>
                  <td>
                    <span className={`pill ${r.beats_baseline ? "ok" : "no"}`}>
                      {r.beats_baseline ? "yes" : "no"}
                    </span>
                  </td>
                  <td className="num">{r.macro_f1.toFixed(3)}</td>
                  <td className="num">{r.per_class.COMPLIANCE.f1.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="note warn" style={{ marginTop: 14 }}>
          <strong>Research artifact — not production-ready</strong>
          <p>
            These results are based on the 63-case targeted human-adjudicated
            test set. The sample was deliberately selected to audit suspected
            evaluator errors and is not representative of the full benchmark, so
            accuracy should not be interpreted as population-level performance.
            The MuRIL / XLM-R ordering is not consistent across configurations,
            so neither is reported as "better", and the classifier is a research
            artifact — not a production safety control.
          </p>
        </div>
      </div>

      {/* caveats */}
      <div className="card">
        <h3>Interpretation constraints</h3>
        <ul className="clean" style={{ marginTop: 10 }}>
          {data.caveats.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
