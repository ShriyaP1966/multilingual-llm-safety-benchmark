import { useEffect, useState } from "react";
import { api, type Methodology as MethodologyData } from "../api";

/**
 * Methodology — a visual pipeline and label provenance up top, then
 * detail in an accordion. Content is fetched from the backend so the
 * evaluator fingerprint, scope and limitations stay in step with the
 * frozen outputs.
 */

const STEPS = [
  { t: "Adversarial prompts", d: "Harmful / adversarial prompts across three languages" },
  { t: "Response generation", d: "Multilingual model responses via a provider factory" },
  { t: "Safety evaluation", d: "Prompt + response rule-based labels (silver)" },
  { t: "Human audit", d: "Targeted human adjudication (gold)" },
  { t: "Statistics", d: "Exploratory + confirmatory analysis" },
  { t: "Classification", d: "Response-only multilingual classifier" },
];

function Section({
  n,
  title,
  children,
  open = false,
}: {
  n: string;
  title: string;
  children: React.ReactNode;
  open?: boolean;
}) {
  return (
    <details open={open}>
      <summary>
        <span className="an">{n}</span>
        {title}
      </summary>
      <div className="ac-body">{children}</div>
    </details>
  );
}

export default function Methodology() {
  const [data, setData] = useState<MethodologyData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.methodology().then(setData).catch((e: Error) => setError(e.message));
  }, []);

  if (error) {
    return (
      <div className="note caution">
        <strong>Methodology unavailable</strong>
        <p>{error}</p>
      </div>
    );
  }
  if (!data) return <p className="loading">Loading methodology…</p>;

  const hash = String(data.evaluator?.sha256 ?? "");

  return (
    <div className="stack">
      <div className="card">
        <h2 className="section">Methodology</h2>
        <p className="section-note">
          How a harmful prompt becomes a reported statistic.
        </p>
        <div className="pipeline">
          {STEPS.map((s, i) => (
            <div className="pstep" key={s.t}>
              <div className="pnum">{i + 1}</div>
              <div className="pt">{s.t}</div>
              <div className="pd">{s.d}</div>
            </div>
          ))}
        </div>
      </div>

      {/* two distinct components */}
      <div className="grid cols-2">
        <div className="card">
          <h3>Safety evaluator — prompt + response</h3>
          <p className="prose" style={{ fontSize: 14.5 }}>
            The research safety evaluator reads <strong>both the original
            adversarial prompt and the model response</strong>, and judges
            whether the response fulfils the harmful request. It produces the
            silver labels used for analysis. The prompt is context; the
            response is the classification target.
          </p>
        </div>
        <div className="card">
          <h3>Response classifier — response only</h3>
          <p className="prose" style={{ fontSize: 14.5 }}>
            The XLM-R / MuRIL classifier (the model this demo runs) reads the{" "}
            <strong>response text only</strong>. It does not receive the
            prompt and does not judge whether the prompt is harmful. The
            language selection is reporting metadata and never changes its
            input.
          </p>
        </div>
      </div>

      {/* provenance */}
      <div className="grid cols-2">
        <div className="card prov silver">
          <div className="prov-tag">Silver</div>
          <div className="prov-n">624</div>
          <div className="prov-l">Automatic evaluator labels</div>
          <p className="prov-p">
            Produced by the rule-based evaluator across every response.
            Automatic — not ground truth.
          </p>
        </div>
        <div className="card prov gold">
          <div className="prov-tag">Gold</div>
          <div className="prov-n">63</div>
          <div className="prov-l">Human-adjudicated targeted audit</div>
          <p className="prov-p">
            Hand-checked cases that deliberately oversampled suspected evaluator
            errors — a targeted audit, not a representative sample.
          </p>
        </div>
      </div>
      <div className="note warn">
        <strong>Silver and gold labels are not treated as interchangeable</strong>
        <p>
          They are never merged in reporting. Gold reveals the evaluator's
          systematic failure modes; it cannot establish population-level
          accuracy.
        </p>
      </div>

      {/* taxonomy */}
      <div className="card">
        <h3>Safety taxonomy</h3>
        <div className="scroll" style={{ marginTop: 10 }}>
          <table className="data">
            <thead>
              <tr><th>Class</th><th>Definition</th></tr>
            </thead>
            <tbody>
              {Object.entries(data.taxonomy).map(([k, v]) => (
                <tr key={k}>
                  <td><span className={`pill ${k}`}>{k}</span></td>
                  <td>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* detail accordion */}
      <div className="accordion">
        <Section n="01" title="Research questions" open>
          <ol className="steps">
            {data.research_questions.map((q) => (
              <li key={q}>{q}</li>
            ))}
          </ol>
        </Section>

        <Section n="02" title="Benchmark design">
          <p>
            104 prompt sets — 13 attack categories × 8 prompt variations —
            rendered in English, Hindi and Marathi: 312 tasks per model, 624
            responses per experiment across two models. The unit of statistical
            independence is the prompt set: each is measured six times, so the
            responses are clustered, not independent.
          </p>
        </Section>

        <Section n="03" title="Evaluator freeze">
          <p>
            The evaluator is pinned so that code, labels, statistics, figures
            and reported results all agree.
          </p>
          <dl className="dl" style={{ marginTop: 12 }}>
            <div><dt>Source</dt><dd style={{ fontSize: 11.5 }}>{String(data.evaluator?.path ?? "—")}</dd></div>
            <div><dt>SHA-256</dt><dd style={{ fontSize: 11 }}>{hash ? `${hash.slice(0, 24)}…` : "—"}</dd></div>
            <div><dt>Manual overrides</dt><dd>{String(data.evaluator?.manual_overrides ?? "—")}</dd></div>
          </dl>
        </Section>

        <Section n="04" title="Statistical methods">
          <div className="scroll">
            <table className="data">
              <thead><tr><th>Aspect</th><th>Approach</th></tr></thead>
              <tbody>
                {Object.entries(data.statistical_methods).map(([k, v]) => (
                  <tr key={k}>
                    <td style={{ whiteSpace: "nowrap" }}><code>{k.replace(/_/g, " ")}</code></td>
                    <td>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

        <Section n="05" title="Classifier">
          <p>
            Two multilingual transformers — XLM-R and MuRIL — classify a
            response into the three safety classes. Input is the response text
            only. The split is grouped by prompt set to prevent leakage, and the
            63 human cases form the held-out test set. Accuracy is always
            reported beside the majority-class baseline.
          </p>
        </Section>

        <Section n="06" title="Limitations">
          <ul className="clean">
            {data.limitations.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
          <p style={{ marginTop: 14, fontSize: 13, color: "var(--ink-3)" }}>
            Full environment, seeds, model identifiers and reproduction order are
            recorded in <code>{data.reproducibility_document}</code>.
          </p>
        </Section>
      </div>
    </div>
  );
}
