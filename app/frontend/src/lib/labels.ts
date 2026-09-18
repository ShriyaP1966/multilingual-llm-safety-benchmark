/**
 * labels.ts
 *
 * One canonical display form for the three safety classes, so the UI
 * never mixes "Non-compliance" and "NON_COMPLIANCE". The underlying
 * taxonomy values from the API are unchanged; this is display only.
 */

import type { SafetyLabel } from "../api";

const DISPLAY: Record<SafetyLabel, string> = {
  COMPLIANCE: "COMPLIANCE",
  NON_COMPLIANCE: "NON-COMPLIANCE",
  REFUSAL: "REFUSAL",
};

/** Uppercase, hyphenated class label used everywhere a class is named. */
export function classLabel(label: SafetyLabel | string): string {
  return DISPLAY[label as SafetyLabel] ?? String(label);
}
