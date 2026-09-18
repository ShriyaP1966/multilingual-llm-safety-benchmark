/**
 * lang.ts
 *
 * Client-side SCRIPT detection, used for reporting metadata and a
 * consistency check only. It never changes the classifier input: the
 * multilingual classifier always receives the raw response text, and
 * the language selection is metadata.
 *
 * Detection is script-based and deliberately honest about its limits:
 *   - Latin letters      -> English
 *   - Devanagari letters -> Hindi OR Marathi (the script cannot tell
 *                           them apart, so the user selects manually)
 */

import type { LanguageCode, ReportLanguage } from "../api";

export type DetectedScript = "latin" | "devanagari" | "mixed" | "unknown";

/** Language selector value: a concrete language, or auto (report by script). */
export type LangChoice = LanguageCode | "auto";

export interface LangInfo {
  script: DetectedScript;
  /**
   * Reporting language actually sent/stored. May be "und" (ambiguous)
   * when auto-detect sees Devanagari, because script cannot tell Hindi
   * from Marathi apart. NEVER silently "hi".
   */
  resolved: ReportLanguage;
  /** True only for a genuine Latin/Devanagari conflict with an explicit choice. */
  mismatch: boolean;
  /** Short status line under the selector (reporting-only framing). */
  status: string | null;
  /** Longer warning shown when there is a real inconsistency. */
  warning: string | null;
}

const DEVANAGARI = /[ऀ-ॿ]/g;
const LATIN = /[A-Za-z]/g;

const NAME: Record<LanguageCode, string> = {
  en: "English",
  hi: "Hindi",
  mr: "Marathi",
};

function detectScript(text: string): DetectedScript {
  const trimmed = text.trim();
  if (trimmed.length < 4) return "unknown";
  const dev = (trimmed.match(DEVANAGARI) || []).length;
  const lat = (trimmed.match(LATIN) || []).length;
  const total = dev + lat;
  if (total === 0) return "unknown";
  const devShare = dev / total;
  if (devShare >= 0.4) return "devanagari";
  if (devShare <= 0.1) return "latin";
  return "mixed";
}

export function analyzeLanguage(text: string, choice: LangChoice): LangInfo {
  const script = detectScript(text);

  // Resolve the reporting language actually sent to the API.
  // SCRIPT DETECTION is not LANGUAGE IDENTIFICATION: Devanagari is
  // shared by Hindi and Marathi, so auto-detect must NOT produce "hi".
  // It resolves Latin -> English, Devanagari/mixed/unknown -> "und"
  // (ambiguous). An explicit user selection is always respected.
  let resolved: ReportLanguage;
  if (choice === "auto") {
    resolved = script === "latin" ? "en" : "und";
  } else {
    resolved = choice;
  }

  let status: string | null = null;
  let warning: string | null = null;
  let mismatch = false;

  if (choice === "auto") {
    if (script === "devanagari") {
      status =
        "Script detected: Devanagari · Language: Ambiguous " +
        "(Hindi/Marathi). Hindi and Marathi both use Devanagari, so the " +
        "language cannot be determined reliably from script alone. " +
        "Select Hindi or Marathi for reporting.";
    } else if (script === "latin") {
      status = "Script detected: Latin · Language: English (en).";
    } else if (script === "mixed") {
      status =
        "Mixed script detected (possible code-switching) · Language: " +
        "Ambiguous. Select a language for reporting.";
    } else {
      status = null;
    }
  } else {
    // explicit choice — frame as reporting metadata
    status = `${NAME[choice]} selected · used for reporting only.`;

    if (script === "devanagari" && choice === "en") {
      mismatch = true;
      warning =
        "The text is in Devanagari script, but English is selected. " +
        "This is a consistency check only — the classifier reads the " +
        "text either way. Devanagari is used by both Hindi and Marathi.";
    } else if (script === "latin" && (choice === "hi" || choice === "mr")) {
      mismatch = true;
      warning =
        `The text is in Latin script, but ${NAME[choice]} is selected. ` +
        "This is a consistency check only — the classifier reads the " +
        "text either way.";
    } else if (script === "devanagari" && (choice === "hi" || choice === "mr")) {
      // consistent; add the honest note that script can't confirm which
      status =
        `${NAME[choice]} selected · used for reporting only. Script is ` +
        "Devanagari, shared by Hindi and Marathi, so this choice is not " +
        "verified automatically.";
    }
  }

  return { script, resolved, mismatch, status, warning };
}

export const languageName = (code: LanguageCode) => NAME[code];
