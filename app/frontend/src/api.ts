/**
 * api.ts
 *
 * Typed client for the research demonstration API.
 *
 * The browser never runs the classifier. Every prediction and every
 * benchmark number comes from the FastAPI backend, which reads the
 * frozen research outputs.
 */

const BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export type SafetyLabel = "COMPLIANCE" | "NON_COMPLIANCE" | "REFUSAL";
export type LanguageCode = "en" | "hi" | "mr";
/**
 * Reporting language actually sent to / returned by the API. "und" =
 * undetermined: auto-detect saw Devanagari, which is shared by Hindi
 * and Marathi, so the language cannot be identified from script alone.
 * It is NEVER silently resolved to "hi".
 */
export type ReportLanguage = LanguageCode | "und";
export type ModelKey = "xlmr" | "muril";

export interface HealthModel {
  available: boolean;
  error: string | null;
  directory: string;
}

export interface Health {
  status: string;
  project: string;
  framework: string;
  device: string;
  models: Record<string, HealthModel>;
  frozen_outputs_present: boolean;
}

export interface Meta {
  project: string;
  framework: string;
  languages: Record<string, string>;
  labels: SafetyLabel[];
  label_definitions: Record<string, string>;
  models: {
    key: string;
    base_model: string | null;
    available: boolean;
    class_weights_applied: boolean | null;
  }[];
  available_models: string[];
  max_length: number;
  task: string;
}

export interface ProbabilityEntry {
  label: SafetyLabel;
  probability: number;
}

export interface EvaluateResult {
  predicted_label: SafetyLabel;
  confidence: number;
  probabilities: ProbabilityEntry[];
  language: string;
  language_name: string;
  model: string;
  model_base: string | null;
  response_characters: number;
  response_words: number;
  input_tokens: number;
  truncated_at_max_length: boolean;
  max_length: number;
  latency_ms: number;
  label_definition: string;
  confidence_note: string;
}

export type Row = Record<string, string>;

export interface ScopeEntry {
  run: string;
  model: string;
  input?: string;
  rows?: number;
  reason?: string;
}

export interface ClassifierRow {
  model: string;
  class_weights: string;
  split: string;
  n: number;
  accuracy: number;
  majority_baseline: number;
  beats_baseline: boolean;
  macro_f1: number;
  per_class: Record<
    string,
    { precision: number; recall: number; f1: number; support: number }
  >;
}

export interface Benchmark {
  scope: {
    included: ScopeEntry[];
    excluded: ScopeEntry[];
    frozen_at: string | null;
    evaluator_sha256: string | null;
    primary_analysis_set: string | null;
    primary_analysis_rows: number | null;
  };
  model_safety: Row[];
  language_safety: Row[];
  model_language_safety: Row[];
  attack_category_safety: Row[];
  variation_safety: Row[];
  exploratory: Row[];
  confirmatory: Row[];
  evaluator_vs_human: Row[];
  evaluator_human_confusion: Row[];
  classifier: ClassifierRow[];
  generation_length: Row[];
  generation_length_summary: GenerationLengthSummary;
  caveats: string[];
}

export interface GenerationLengthSummary {
  model: string;
  paired_responses: number;
  labels_changed: number;
  ceiling_pct_512: number | null;
  ceiling_pct_2048: number | null;
  refusal_512: number | null;
  refusal_2048: number | null;
  qwen_excluded_reason: string;
}

export interface Methodology {
  project: string;
  framework: string;
  research_questions: string[];
  benchmark: Record<string, unknown>;
  taxonomy: Record<string, string>;
  pipeline: string[];
  evaluator: Record<string, unknown>;
  scope: { included: ScopeEntry[]; excluded: ScopeEntry[] };
  statistical_methods: Record<string, string>;
  limitations: string[];
  reproducibility_document: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* keep the status text */
    }
    throw new Error(detail);
  }

  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/api/health"),
  meta: () => request<Meta>("/api/meta"),
  benchmark: () => request<Benchmark>("/api/benchmark"),
  methodology: () => request<Methodology>("/api/methodology"),
  evaluate: (payload: {
    text: string;
    language: ReportLanguage;
    model: ModelKey;
  }) =>
    request<EvaluateResult>("/api/evaluate", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};

/** Percentage formatter used across the results tables. */
export const pct = (value: number, digits = 1) =>
  `${(value * 100).toFixed(digits)}%`;

/** Parse a numeric CSV cell that arrives as a string. */
export const num = (value: string | undefined) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
};
