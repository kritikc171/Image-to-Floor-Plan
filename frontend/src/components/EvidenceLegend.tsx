/** Evidence-status legend shared by results and documentation. */

import type { EvidenceStatus } from "../api/client";

const STATUS_LABELS: Record<EvidenceStatus, string> = {
  OBSERVED: "Observed",
  INFERRED: "Inferred",
  GENERATED: "Generated",
  UNCERTAIN: "Uncertain",
};

const STATUS_DESCRIPTIONS: Record<EvidenceStatus, string> = {
  OBSERVED: "directly visible in your photo",
  INFERRED: "estimated from visible evidence",
  GENERATED: "derived from an AI-generated view",
  UNCERTAIN: "not enough evidence to decide",
};

export function EvidenceLegend() {
  return (
    <ul className="legend" aria-label="Evidence status legend">
      {(Object.keys(STATUS_LABELS) as EvidenceStatus[]).map((status) => (
        <li key={status} className="legend__item">
          <span
            className={`legend__dot legend__dot--${status.toLowerCase()}`}
            aria-hidden="true"
          />
          <span>
            <strong>{STATUS_LABELS[status]}:</strong>{" "}
            {STATUS_DESCRIPTIONS[status]}
          </span>
        </li>
      ))}
    </ul>
  );
}
