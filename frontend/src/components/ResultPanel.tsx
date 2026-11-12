/** Successful result: floor plan (left) + information cards (right). */

import { useCallback, useState } from "react";
import { floorplanUrl, type GenerationResult } from "../api/client";
import { API_BASE } from "../api/client";
import { EvidenceLegend } from "./EvidenceLegend";

function formatConfidence(confidence: number | null): string {
  if (confidence === null || confidence === undefined) return "—";
  return `${Math.round(confidence * 100)}%`;
}

const STATUS_LABELS: Record<string, string> = {
  OBSERVED: "Observed",
  INFERRED: "Inferred",
  GENERATED: "Generated",
  UNCERTAIN: "Uncertain",
};

const ICONS: Record<string, string> = {
  plan: "🏛️",
  objects: "📦",
  qa: "💬",
  legend: "🔎",
  limit: "⚠️",
};

function Chip(props: { status: string }) {
  const status = props.status.toLowerCase();
  return (
    <span className={`status-chip status-chip--${status}`}>
      {STATUS_LABELS[props.status] ?? props.status}
    </span>
  );
}

function FloorPlanCard(props: {
  jobId: string;
  onDownloadPng: () => void;
  onDownloadJson: () => void;
  downloading: boolean;
  downloadError: string | null;
}) {
  return (
    <section className="card result__figure" aria-label="Generated floor plan">
      <header className="card__head">
        <span className="icon-chip icon-chip--plan">{ICONS.plan}</span>
        <div>
          <h3>Your approximate floor plan</h3>
          <p
            className="result__subtitle"
            style={{ margin: 0, fontSize: "0.8rem" }}
          >
            Estimated from one photo — verify dimensions by hand.
          </p>
        </div>
      </header>
      <div className="result__plan-wrap">
        <img
          src={floorplanUrl(props.jobId)}
          alt="Approximate top-down floor plan"
        />
        <p className="result__plan-caption">
          Walls, furniture footprints and numbered markers are estimates —
          verify by hand.
        </p>
      </div>
      <div className="result__actions">
        <button
          type="button"
          className="button button--primary"
          onClick={props.onDownloadPng}
          disabled={props.downloading}
        >
          {props.downloading ? "Preparing…" : "Download PNG"}
        </button>
        <button
          type="button"
          className="button button--secondary"
          onClick={props.onDownloadJson}
        >
          Download JSON Layout
        </button>
        {props.downloadError && (
          <p className="result__error" role="alert">
            {props.downloadError}
          </p>
        )}
      </div>
    </section>
  );
}

function DetectedObjectsCard(props: {
  objects: GenerationResult["hypothesis"]["objects"];
}) {
  const { objects } = props;
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const toggle = (id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const isExpanded = (id: string) => expanded.has(id);

  if (objects.length === 0) {
    return (
      <section className="card">
        <header className="card__head">
          <span className="icon-chip icon-chip--objects">{ICONS.objects}</span>
          <h3>Detected objects</h3>
        </header>
        <div className="card__body">
          <p className="empty-state">
            No objects could be reliably identified from this photo.
          </p>
      </div>
    </section>
  );
}


  return (
    <section className="card">
      <header className="card__head">
        <span className="icon-chip icon-chip--objects">{ICONS.objects}</span>
        <div>
          <h3>
            Detected objects{" "}
            <span className="count-badge">{objects.length}</span>
          </h3>
        </div>
      </header>
      <div className="card__body">
        <ul className="object-list">
          {objects.map((obj) => {
            const id = obj.id;
            const open = isExpanded(id);
            const label = obj.label ?? obj.type;
            const conf =
              obj.confidence !== null ? formatConfidence(obj.confidence) : "—";
            return (
              <li
                key={id}
                className={`object-list__item ${
                  open ? "object-list__item--expanded" : ""
                }`}
                onClick={() => toggle(id)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    toggle(id);
                  }
                }}
              >
                <span
                  className="object__number"
                  style={{
                    background:
                      STATUS_COLORS[obj.evidence_status as keyof typeof STATUS_COLORS] ??
                      "#94a3b8",
                  }}
                >
                  {String(id).split("-").pop() ?? id}
                </span>
                <div className="object__body">
                  <span className="object__label">{label}</span>
                  <span className="object__meta">
                    <Chip status={obj.evidence_status} />{" "}
                    <span className="object__confidence">{conf}</span>
                  </span>
                </div>
                <span className="object__chevron">
                  {open ? "▲" : "▼"}
                </span>
                {open && (
                  <div className="object__details">
                    {obj.uncertainty_reason && (
                      <p>
                        <b>Why this status:</b>{" "}
                        {obj.uncertainty_reason}
                      </p>
                    )}
                    {obj.layout_issues?.length && (
                      <p>
                        <b>Layout notes:</b> {" "}
                        {obj.layout_issues.join("; ")}.
                      </p>
                    )}
                    {obj.position && (
                      <p>
                        <b>Approximate plan position:</b> ({obj.position.x.toFixed(
                          2
                        )}, {obj.position.y.toFixed(2)})
                      </p>
                    )}
                    {!obj.position && (
                      <p>
                        <b>Position:</b> not estimated — shown as unplaced.
                      </p>
                    )}
                    {obj.image_bbox && (
                      <p>
                        <b>Image region:</b> [{obj.image_bbox.x_min.toFixed(2)},
                        {obj.image_bbox.y_min.toFixed(2)} →{" "}
                        {obj.image_bbox.x_max.toFixed(2)},
                        {obj.image_bbox.y_max.toFixed(2)}]
                      </p>
                    )}
                    {obj.footprint && (
                      <p>
                        <b>Approx. footprint:</b> {obj.footprint.width} ×{" "}
                        {obj.footprint.depth} (normalised plan units, not
                        measured)
                      </p>
                    )}
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      </div>
    </section>
  );
}

const STATUS_COLORS: Record<string, string> = {
  OBSERVED: "#16a34a",
  INFERRED: "#d97706",
  GENERATED: "#a855f7",
  UNCERTAIN: "#94a3b8",
};

function QACard(props: {
  jobId: string;
  onDownloadError?: (msg: string) => void;
}) {
  const [question, setQuestion] = useState("");
  const [qaLoading, setQaLoading] = useState(false);
  const [qaError, setQaError] = useState<string | null>(null);
  const [qaAnswer, setQaAnswer] = useState<string | null>(null);

  const send = useCallback(async () => {
    setQaLoading(true);
    setQaError(null);
    setQaAnswer(null);
    try {
      const res = await fetch(`${API_BASE}/api/generations/${props.jobId}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!res.ok) throw new Error("Request failed");
      const data = await res.json();
      setQaAnswer(data.answer);
    } catch {
      setQaError("Could not get an answer.");
    } finally {
      setQaLoading(false);
    }
  }, [props.jobId, question]);

  return (
    <section className="card">
      <header className="card__head">
        <span className="icon-chip icon-chip--qa">{ICONS.qa}</span>
        <h3>Ask About Your Floor Plan</h3>
      </header>
      <div className="card__body">
        <div className="qa-box">
          <div className="qa-input-row">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question about this floor plan"
              disabled={qaLoading}
            />
            <button
              type="button"
              className="button button--primary"
              onClick={send}
              disabled={qaLoading || !question.trim()}
            >
              {qaLoading ? "Sending…" : "Send"}
            </button>
          </div>
          {qaError && (
            <p className="result__error" role="alert">
              {qaError}
            </p>
          )}
          {qaAnswer && (
            <p className="qa-answer">{qaAnswer}</p>
          )}
        </div>
      </div>
    </section>
  );
}

function AssumptionsCard(props: {
  assumptions: string[];
  openQuestions: string[];
  provenance: GenerationResult["hypothesis"]["provenance"];
}) {
  const { assumptions, openQuestions, provenance } = props;
  return (
    <section className="card">
      <header className="card__head">
        <span className="icon-chip icon-chip--limit">{ICONS.limit}</span>
        <h3>Assumptions &amp; limitations</h3>
      </header>
      <div className="card__body">
        {assumptions.length > 0 && (
          <>
            <ul className="assumption-list">
              {assumptions.map((a, i) => (
                <li key={i}>{a}</li>
              ))}
            </ul>
            {openQuestions.length > 0 && (
              <>
                <h4
                  className="card__subhead"
                  style={{ color: "var(--text-muted)" }}
                >
                  Open questions
                </h4>
                <ul className="assumption-list">
                  {openQuestions.map((q, i) => (
                    <li key={i}>{q}</li>
                  ))}
                </ul>
              </>
            )}
          </>
        )}
        <p className="result__provenance">
          Reasoning:{ " "}
          {provenance.reasoning_provider === "groq"
            ? "AI-assisted (Groq)"
            : "deterministic fallback"}
          {provenance.detection_model && (
            <>
              {" · "}
              Detector: {provenance.detection_model}
            </>
          )}
          {" · "}
          Depth: {provenance.depth_method}
        </p>
      </div>
    </section>
  );
}

function LegendCard() {
  return (
    <section className="card">
      <header className="card__head">
        <span className="icon-chip icon-chip--legend">{ICONS.legend}</span>
        <h3>Evidence legend</h3>
      </header>
      <div className="card__body">
        <EvidenceLegend />
      </div>
    </section>
  );
}

export function ResultPanel(props: {
  jobId: string;
  result: GenerationResult;
  onReset: () => void;
}) {
  const { result } = props;
  const { hypothesis } = result;
  const objects = hypothesis.objects;
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);

  const downloadPng = useCallback(async () => {
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

  const downloadJson = useCallback(() => {
    setDownloadError(null);
    try {
      const json = JSON.stringify(result, null, 2);
      const blob = new Blob([json], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `floor-plan-layout-${props.jobId}.json`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch {
      setDownloadError("The layout could not be downloaded.");
    }
  }, [result, props.jobId]);

  const highUncertainty =
    hypothesis.uncertainty.overall === "high";

  return (
    <section className="result" aria-label="Floor plan result">
      <header className="result__header">
        <h1 className="result__title">Your approximate floor plan</h1>
        <p className="result__subtitle">
          Schematic approximation from a single photo — not to scale.
        </p>
        {highUncertainty && (
          <div className="result__warn result__warn--high" role="alert">
            <strong>High uncertainty.</strong> This plan is a rough sketch
            from one photo; hidden geometry cannot be recovered and positions
            are approximate. Verify everything by hand.
          </div>
        )}
      </header>

      <div className="result__grid">
        <FloorPlanCard
          jobId={props.jobId}
          onDownloadPng={downloadPng}
          onDownloadJson={downloadJson}
          downloading={downloading}
          downloadError={downloadError}
        />

        <div className="result__summary">
          <DetectedObjectsCard objects={objects} />
          <QACard jobId={props.jobId} />
          <LegendCard />
          <AssumptionsCard
            assumptions={hypothesis.uncertainty.assumptions}
            openQuestions={hypothesis.uncertainty.open_questions}
            provenance={hypothesis.provenance}
          />
          <button
            type="button"
            className="button button--secondary"
            onClick={props.onReset}
            style={{ marginTop: 8 }}
          >
            Generate another
          </button>
        </div>
      </div>
    </section>
  );
}

export default ResultPanel;
