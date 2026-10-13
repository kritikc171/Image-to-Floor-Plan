/** Successful result: floor plan, objects, assumptions, actions. */

import { useCallback, useState } from "react";
import { floorplanUrl, type GenerationResult } from "../api/client";
import { EvidenceLegend } from "./EvidenceLegend";

function StatusText(props: { status: string }) {
  const labels: Record<string, string> = {
    OBSERVED: "Observed",
    INFERRED: "Inferred",
    GENERATED: "Generated",
    UNCERTAIN: "Uncertain",
  };
  return (
    <span className={`chip chip--${props.status.toLowerCase()}`}>
      {labels[props.status] ?? props.status}
    </span>
  );
}

export function ResultPanel(props: {
  jobId: string;
  result: GenerationResult;
  onReset: () => void;
}) {
  const { result } = props;
  const { hypothesis } = result;
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);

  const download = useCallback(async () => {
    setDownloading(true);
    setDownloadError(null);
    try {
      const response = await fetch(floorplanUrl(props.jobId));
      if (!response.ok) throw new Error("Download failed.");
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `floor-plan-${props.jobId}.png`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch {
      setDownloadError("The floor plan could not be downloaded.");
    } finally {
      setDownloading(false);
    }
  }, [props.jobId]);

  const objectRows = hypothesis.objects.slice(0, 8);

  return (
    <section className="result" aria-label="Floor plan result">
      <h2 className="result__title">Your approximate floor plan</h2>
      <p className="result__subtitle">
        Schematic approximation from a single photo — not to scale.
      </p>

      <div className="result__grid">
        <figure className="result__figure">
          <img
            src={floorplanUrl(props.jobId)}
            alt="Approximate top-down floor plan"
          />
          <figcaption>
            <button
              type="button"
              className="button button--primary"
              onClick={download}
              disabled={downloading}
            >
              {downloading ? "Preparing…" : "Download PNG"}
            </button>
            {downloadError && (
              <p className="result__error" role="alert">
                {downloadError}
              </p>
            )}
          </figcaption>
        </figure>

        <aside className="result__summary">
          <h3>Detected objects</h3>
          {objectRows.length === 0 ? (
            <p className="result__empty">
              No objects could be reliably identified from this photo.
            </p>
          ) : (
            <ul className="object-list">
              {objectRows.map((obj) => (
                <li key={obj.id} className="object-list__item">
                  <span className="object-list__label">
                    {obj.label ?? obj.type}
                  </span>
                  <StatusText status={obj.evidence_status} />
                </li>
              ))}
            </ul>
          )}

          <h3>Legend</h3>
          <EvidenceLegend />

          <h3>Assumptions &amp; limitations</h3>
          <ul className="assumption-list">
            {hypothesis.uncertainty.assumptions.map((a, i) => (
              <li key={i}>{a}</li>
            ))}
          </ul>

          {hypothesis.uncertainty.open_questions.length > 0 && (
            <>
              <h3>Open questions</h3>
              <ul className="assumption-list">
                {hypothesis.uncertainty.open_questions.map((q, i) => (
                  <li key={i}>{q}</li>
                ))}
              </ul>
            </>
          )}

          <p className="result__provenance">
            Reasoning:{" "}
            {hypothesis.provenance.reasoning_provider === "groq"
              ? "AI-assisted (Groq)"
              : "deterministic fallback"}
            {" · "}
            Depth: {hypothesis.provenance.depth_method}
          </p>

          <button
            type="button"
            className="button button--secondary"
            onClick={props.onReset}
          >
            Start over
          </button>
        </aside>
      </div>
    </section>
  );
}
