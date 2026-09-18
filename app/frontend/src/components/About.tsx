/**
 * Research — narrative framing. Static: it explains the study rather
 * than reporting numbers (every figure lives on Results, read from the
 * frozen outputs). Key findings are stated qualitatively and point to
 * Results for the exact frozen values.
 */
export default function About() {
  return (
    <div className="stack">
      <div className="card">
        <h2 className="section">Research objective</h2>
        <div className="prose">
          <p>
            <strong>
              Cross-Lingual Vulnerability and Prompt Injection in Low-Resource
              Languages
            </strong>{" "}
            asks whether a language model's safety behaviour holds up equally
            across languages, and whether adversarial prompt framing shifts
            that behaviour differently depending on the language used.
          </p>
          <p>
            Most published safety evaluation is done in English. If a model
            refuses a harmful request in English but complies with the same
            request in Hindi or Marathi, its safety training has not
            transferred across languages — and speakers of those languages
            carry the risk. That gap is the subject of this study.
          </p>
          <p>
            The benchmark software built to carry out the work is the{" "}
            <strong>Multilingual LLM Safety Evaluation Framework</strong>. The
            framework is the instrument; the study is the research.
          </p>
        </div>
      </div>

      <div className="card">
        <h3>Research questions</h3>
        <ol className="steps" style={{ marginTop: 12 }}>
          <li>Do safety outcomes differ between the evaluated language models?</li>
          <li>Do safety outcomes differ across English, Hindi and Marathi?</li>
          <li>Do different attack categories produce different safety outcomes?</li>
          <li>Do prompt variations affect safety behaviour?</li>
          <li>How reliable is automatic safety evaluation compared with human judgement?</li>
          <li>Can multilingual transformers classify the safety of model responses?</li>
          <li>Does classifier performance on automatically labelled data generalise to human-validated data?</li>
        </ol>
      </div>

      <div className="card">
        <h3>Benchmark design</h3>
        <div className="stats" style={{ marginTop: 14 }}>
          <div className="stat"><div className="n">104</div><div className="l">prompt sets (13 categories × 8 variations)</div></div>
          <div className="stat"><div className="n">3</div><div className="l">languages — English, Hindi, Marathi</div></div>
          <div className="stat"><div className="n">312</div><div className="l">tasks per model (104 × 3 languages)</div></div>
          <div className="stat"><div className="n">624</div><div className="l">responses per experiment, two models</div></div>
        </div>
        <p className="prose" style={{ marginTop: 16 }}>
          The eight prompt variations change framing, not content — baseline,
          urgency, trusted relationship, roleplay, hypothetical, obfuscation,
          multilingual code-switch, and emotional appeal — so the study can
          separate <em>what</em> is asked from <em>how</em> it is asked. Each
          prompt set is measured six times (three languages × two models), so
          the responses are clustered rather than independent.
        </p>
      </div>

      <div className="card">
        <h3>Key findings</h3>
        <p className="chart-sub">
          Stated qualitatively here; exact frozen values are on the Results
          page.
        </p>
        <div className="findings">
          <div className="finding">
            <div className="fk">Models differ</div>
            <div className="ft">Refusal rates vary by model</div>
            <p>
              The two models refuse at significantly different rates — a
              statistically significant association after correction.
            </p>
          </div>
          <div className="finding">
            <div className="fk">Language matters</div>
            <div className="ft">Outcomes vary across languages</div>
            <p>
              Safety-class distributions differ by language, but under the
              final evaluator no pairwise language comparison survives
              multiple-testing correction. The differences are descriptive, not
              confirmed at the corrected threshold.
            </p>
          </div>
          <div className="finding">
            <div className="fk">Evaluator is imperfect</div>
            <div className="ft">Automatic labels are silver, not gold</div>
            <p>
              A targeted human audit surfaced systematic disagreements with the
              automatic evaluator, which is why silver and gold labels are kept
              separate.
            </p>
          </div>
          <div className="finding">
            <div className="fk">Classifier is limited</div>
            <div className="ft">Performance varies by configuration</div>
            <p>
              Classifier performance varies by configuration on the targeted
              human-adjudicated test set. That set contains 63 targeted cases
              and is not a representative sample, so these results should not be
              read as population-level accuracy. It is a research artifact, not
              a production tool.
            </p>
          </div>
        </div>
      </div>

      <div className="card">
        <h3>Ethical considerations</h3>
        <ul className="clean" style={{ marginTop: 10 }}>
          <li>Harmful prompts are used in a controlled research setting to measure defensive behaviour, not to produce usable harmful content.</li>
          <li>This demonstration classifies responses. It does not generate content and does not run prompts against a live model.</li>
          <li>The sample responses in the evaluator are refusals and safe alternatives, chosen so the demo never displays actionable harmful material.</li>
          <li>The classifier is a research artifact. It is not production-ready and must not be used as a safety control.</li>
        </ul>
      </div>
    </div>
  );
}
