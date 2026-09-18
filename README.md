# 🌍 Cross-Lingual Vulnerability and Prompt Injection in Low-Resource Languages

A research study of how large language models respond to harmful and adversarial prompts across English, Hindi and Marathi, using a systematic multilingual benchmark.

The study varies attack category and prompt framing, collects model responses, evaluates their safety, audits that evaluation against a targeted adjudication set, analyses the results statistically, and investigates whether multilingual transformers can classify the safety of model-generated responses.

**Software:** the benchmark system built to carry out this research is the **Multilingual LLM Safety Evaluation Framework**. That name refers to the software; the study itself is titled as above.

> **Naming note.** Use *Cross-Lingual Vulnerability and Prompt Injection in Low-Resource Languages* for the research project, paper, report and presentation. Use *Multilingual LLM Safety Evaluation Framework* only when referring specifically to the software.

---

## 🎯 Research questions

1. Do safety outcomes differ between the evaluated LLMs?
2. Do safety outcomes differ across English, Hindi and Marathi?
3. Do different attack categories produce different safety outcomes?
4. Do prompt variations affect safety behaviour?
5. How reliable is automatic safety evaluation compared with human judgement?
6. Can multilingual transformer models classify the safety of LLM-generated responses?
7. Does classifier performance based on automatically labelled data generalise to human-validated data?

---

## 🎯 Objectives

- Benchmark multilingual LLM safety performance across low-resource languages
- Compare the evaluated models across English, Hindi and Marathi
- Generate reproducible experiments
- Produce statistics and visualizations
- Build a reusable evaluation framework
- Extend the framework with an automated safety classifier

---

## ✨ Features

### ✅ Implemented

- Dataset validation and UTF-8 sanitization
- Multilingual dataset loader
- Benchmark execution engine with resumable runs
- Provider Factory architecture (mock, Groq, OpenAI, Gemini, Qwen)
- Experiment management and result writer
- Configuration-driven execution, selectable via `BENCHMARK_CONFIG`
- Long-to-wide merge step carrying `output_tokens` through
- Rule-based safety evaluator over **prompt + response** (refusal, actionable-content, defensive-context detection)
- Targeted audit queue and adjudication tooling
- Exploratory statistical analysis (chi-square, Cramér's V)
- Confirmatory statistical analysis (Stuart–Maxwell, GEE, Benjamini–Hochberg)
- Evaluator change-record tooling
- Classifier dataset preparation and leakage-checked grouped split
- MuRIL and XLM-R training (**response-only** classifier), gold evaluation and comparison
- Visualization scripts

### 🚧 In Progress

- 2048-token follow-up experiment — complete for GPT-OSS, **Qwen arm blocked** by a provider output-token rate limit
- 512 vs 2048 generation-length comparison (tooling ready, awaiting the Qwen arm)
- Publication-quality figures and tables
- Research paper

### 🔮 Planned / future strengthening

- Representative human-validated sample (~150 cases) for population-level evaluator reliability
- Additional LLM providers and languages
- Optional demonstration layer (React + Vite → FastAPI → XLM-R)

---

## ⚠️ Interpretation constraints

These apply to any reported result from this project:

- Evaluator output is a **silver label**, not ground truth. Adjudicated audit cases are the **gold labels**, and the two are always reported separately.
- The 63 gold cases come from a **targeted audit** that deliberately oversampled suspected evaluator errors. They are not a representative sample and must not be presented as population-level evaluator or classifier accuracy.
- Statistical tests here are **association tests**. Correct phrasing is that a prompt framing *was associated with* a different distribution of safety outcomes — not that it *caused* unsafe behaviour.
- Differences between the 512-token and 2048-token experiments reflect **generation-length / truncation sensitivity**. The mechanism behind the observed language differences is not established by these experiments.
- Classifier accuracy is always reported beside the majority-class baseline.

---

## 🌐 Languages

- 🇺🇸 English
- 🇮🇳 Hindi
- 🇮🇳 Marathi

---

## 📊 Benchmark Dataset

- **104 multilingual prompt sets** (13 attack categories x 8 prompt variations)
- **13 attack categories**, mapping one-to-one onto `attack_id`
- **8 prompt variations** (baseline, urgency, trusted relationship, roleplay,
  hypothetical, obfuscation, multilingual code-switch, emotional appeal)
- **3 languages**: English, Hindi, Marathi
- **312 benchmark tasks per model** (104 prompt sets x 3 languages)
- **624 collected responses** in the 512-token experiment (312 per model)

The unit of statistical independence is the **prompt set**, not the
response: each prompt set is measured six times (3 languages x 2 models).

---

## 🏗 Architecture

```text
Dataset
   │
   ▼
Validation
   │
   ▼
Task Generation
   │
   ▼
Provider Factory
   │
   ├── Mock
   ├── Groq  ── openai/gpt-oss-20b, qwen/qwen3.8-27b   (used for recorded runs)
   ├── OpenAI                                          (configured, unused)
   └── Gemini                                          (configured, unused)
   │
   ▼
Benchmark Engine (resumable)
   │
   ▼
Result Writer  ──►  outputs/raw/       512-token experiment
                    outputs/raw_2048/  2048-token experiment
   │
   ▼
Merge (long ─► wide, carries output_tokens)
   │
   ▼
Automatic Evaluation  ──►  silver labels
   │                          │
   │                          ▼
   │                 Targeted Audit ──►  gold labels (63 adjudicated)
   ▼
Statistics                Classifier
   ├── exploratory        ├── grouped leakage-checked split
   └── confirmatory       └── MuRIL / XLM-R  ─►  response safety class
   │
   ▼
Figures & Tables
```

---

## 🛠 Tech Stack

**Language**
- Python 3.13

**Core libraries**
- pandas, numpy
- scipy, statsmodels (statistical analysis)
- scikit-learn (metrics, grouped splits)
- matplotlib (figures)
- python-dotenv

**Models served via Groq** (provider used for the recorded runs)
- `openai/gpt-oss-20b`
- `qwen/qwen3.8-27b`

**Classifiers**
- transformers, torch, datasets, accelerate
- MuRIL (`google/muril-base-cased`)
- XLM-R (`xlm-roberta-base`)

See `requirements.txt` for pinned versions and `REPRODUCIBILITY.md`
for the full recorded environment.

**Translation** (dataset construction)
- Hugging Face Transformers
- IndicTrans2

**Tools**
- Git
- GitHub
- VS Code

---

## 📂 Project Structure

```
NLP_Safety_Benchmark/
│
├── config/
├── data/
├── experiments/
├── outputs/
├── scripts/
│   ├── providers/
│   ├── analysis/
│   ├── dataset_loader.py
│   ├── result_writer.py
│   ├── run_experiment.py
│   ├── sanitize_dataset.py
│   ├── validate_dataset.py
│   └── utils.py
│
└── README.md
```

---

## 🚀 Getting Started

Clone the repository

```bash
git clone https://github.com/ShriyaP1966/multilingual-llm-safety-benchmark.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the benchmark

```bash
python scripts/run_experiment.py
```

---

## 🗺 Roadmap

- [x] Project foundation
- [x] Dataset validation and sanitization
- [x] Benchmark execution engine (resumable)
- [x] Provider Factory architecture
- [x] Mock provider
- [x] Groq integration (models used for the recorded runs)
- [x] Automatic safety evaluation
- [x] Targeted audit and adjudication tooling (63 gold cases)
- [x] Exploratory statistics and figures
- [x] Confirmatory statistics (repeated-measures appropriate)
- [x] Safety classifier (MuRIL and XLM-R, response-only)
- [x] 512-token benchmark experiment
- [~] 2048-token follow-up experiment (GPT-OSS complete, Qwen blocked)
- [ ] 512 vs 2048 generation-length comparison
- [ ] Publication-quality figures and tables
- [ ] Research paper
- [ ] Optional demonstration interface

---

## 🎓 Research Vision

This project aims to evolve into a complete multilingual LLM safety evaluation platform by combining:

- Benchmarking
- Automated evaluation
- Statistical analysis
- Cross-model comparison
- AI-powered safety classification

The framework is designed to be modular, reproducible, and easily extendable for future LLMs and multilingual safety research.

---

## 📄 License

**All Rights Reserved.**

This repository is provided for viewing, educational, and research reference only.

No permission is granted to copy, modify, redistribute, or commercially use the source code without prior written permission from the author.

See the [LICENSE](LICENSE) file for full terms.

---

## 👩‍💻 Author

**Shriya Patil**

B.Sc. Artificial Intelligence

**Research Interests**
- Artificial Intelligence
- Large Language Models (LLMs)
- AI Safety
- Natural Language Processing
- Multilingual AI
