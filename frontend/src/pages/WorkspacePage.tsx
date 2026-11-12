import { useCallback, useRef } from "react";
import { Dropzone } from "../components/Dropzone";
import { ImagePreview } from "../components/ImagePreview";
import { ProcessingPanel } from "../components/ProcessingPanel";
import { ResultPanel } from "../components/ResultPanel";
import { ErrorPanel } from "../components/ErrorPanel";
import { Button } from "../components/Button";
import { useGeneration } from "../hooks/useGeneration";

export default function WorkspacePage() {
  const {
    phase,
    selectFile,
    removeFile,
    replaceFile,
    submit,
    retry,
    reset,
  } = useGeneration();
  const replaceInputRef = useRef<HTMLInputElement>(null);

  const openReplace = useCallback(() => {
    replaceInputRef.current?.click();
  }, []);

  const selectedFileName =
    phase.kind === "selected"
      ? phase.file.name
      : phase.kind === "failed" && phase.file
        ? phase.file.name
        : null;

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="inner">
          <img
            src="/favicon.svg"
            alt=""
            className="logo"
            aria-hidden="true"
            width={36}
            height={36}
          />
          <div className="brand">
            <h1>Floor Plan Generator</h1>
            <p className="subtitle">
              Turn any room photo into an approximate floor plan using AI.
            </p>
          </div>
          <span className="ai-indicator">
            <span className="pulse" aria-hidden="true"></span>
            AI-powered spatial intelligence
          </span>
        </div>
      </header>

      <main className="app-main">
        {phase.kind === "empty" && (
          <section className="workspace" aria-label="Upload a room photo">
            <h2 className="workspace__title">
              Turn one room photo into an approximate floor plan
            </h2>
            <p className="workspace__subtitle">
              Upload a single indoor photograph. You will get a schematic
              top-down view where every element is marked as observed,
              inferred, or uncertain.
            </p>
            <Dropzone onFile={selectFile} />
            <div className="disclaimer">
              <strong>Approximate results.</strong> This tool produces a
              schematic hypothesis, not verified measurements. A single
              photo cannot recover hidden geometry or exact dimensions —
              treat every dimension as unconfirmed until measured by hand.
            </div>
          </section>
        )}

        {phase.kind === "selected" && (
          <section className="workspace" aria-label="Review your photo">
            <h2 className="workspace__title">Review and generate</h2>
            {phase.error && (
              <p className="workspace__error" role="alert">
                {phase.error}
              </p>
            )}
            <ImagePreview
              previewUrl={phase.previewUrl}
              fileName={phase.file.name}
              onReplace={openReplace}
              onRemove={removeFile}
            />
            <input
              ref={replaceInputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp,image/bmp,image/gif"
              className="visually-hidden"
              aria-hidden="true"
              tabIndex={-1}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) replaceFile(file);
                e.target.value = "";
              }}
            />
            <div className="workspace__actions">
              <Button
                variant="primary"
                disabled={phase.error !== null}
                onClick={() => submit(phase.file)}
              >
                Generate floor plan
              </Button>
              <Button variant="secondary" onClick={removeFile}>
                Choose a different photo
              </Button>
            </div>
            <div className="disclaimer">
              <strong>Approximate results.</strong> The floor plan is a
              schematic hypothesis, not verified measurements.
            </div>
          </section>
        )}

        {phase.kind === "submitting" && (
          <ProcessingPanel fileName="Uploading…" />
        )}

        {phase.kind === "processing" && selectedFileName && (
          <ProcessingPanel fileName={selectedFileName} />
        )}

        {phase.kind === "completed" && (
          <ResultPanel
            jobId={phase.jobId}
            result={phase.result}
            onReset={reset}
          />
        )}

        {phase.kind === "failed" && (
          <ErrorPanel
            message={phase.message}
            retryable={phase.retryable}
            onRetry={retry}
            onReset={reset}
          />
        )}

        <section className="how-it-works" aria-label="How it works">
          <div className="inner">
            <h2 className="how-it-works__title">How it works</h2>
            <ol className="how-it-works__steps">
              <li>
                <strong>Upload one indoor photo.</strong> The photo is decoded
                and validated server-side (format, size, readability). Nothing
                is stored permanently.
              </li>
              <li>
                <strong>Objects are detected.</strong> A pretrained, locally run
                object detector (SSDlite-MobileNetV3 on COCO) finds visible
                furniture such as beds, sofas, chairs, tables, and TVs. Each
                detection is marked
                <span className="evidence observed">OBSERVED</span> with a
                confidence score and normalized image bounding box.
              </li>
              <li>
                <strong>Positions are estimated.</strong> Depth and floor cues
                project detected objects into an approximate top-down plan.
                Positions from a single photo are estimates, not measurements:
                anything uncertain is labelled
                <span className="evidence uncertain">UNCERTAIN</span>, never
                presented as fact.
              </li>
              <li>
                <strong>A new floor-plan PNG is rendered</strong> for your
                upload, with a unique job ID, walls, furniture symbols, a
                legend, and uncertainty indicators. Download the PNG or the
                structured JSON layout.
              </li>
            </ol>

            <div className="limitations">
              <h3 className="how-it-works__title">Limitations</h3>
              <ul className="limitations__list">
                <li>
                  Single-image uncertainty. Walls behind the camera, occluded
                  objects, and exact dimensions cannot be recovered from one
                  photograph; all output is approximate and not to scale.
                </li>
                <li>
                  Detector coverage. The detector covers classes such as bed,
                  couch, chair, dining table, and TV. It does not detect desk,
                  cabinet, wardrobe, shelf, door, or window — these are never
                  fabricated.
                </li>
                <li>
                  Not an architectural tool. Use the output as a starting
                  sketch; verify all dimensions by hand before any purchase or
                  build.
                </li>
              </ul>
            </div>
          </div>
        </section>
      </main>

      <footer className="app-footer">
        <div className="inner">
          Floor Plan Generator — local prototype. Images and results stay on
          your machine and are deleted after a short retention window.
        </div>
      </footer>
    </div>
  );
}
