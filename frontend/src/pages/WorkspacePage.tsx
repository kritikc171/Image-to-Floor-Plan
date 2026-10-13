import { Button } from "../components/Button";

/**
 * Landing / workspace shell (Milestone 1).
 * The interactive upload journey is wired in Milestone 2.
 */
export default function WorkspacePage() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="inner">
          <img
            src="/favicon.svg"
            alt=""
            className="logo"
            aria-hidden="true"
            width={32}
            height={32}
          />
          <h1>Floor Plan Generator</h1>
        </div>
      </header>
      <main className="app-main">
        <section className="hero">
          <h2>Turn one room photo into an approximate floor plan</h2>
          <p>
            Upload a single indoor photograph to get a schematic top-down view
            with detected furniture, doors, and windows — each clearly marked
            as observed, inferred, or uncertain.
          </p>
          <div className="disclaimer">
            <strong>Approximate results.</strong> This tool produces a schematic
            hypothesis, not verified measurements. A single photo cannot recover
            hidden geometry or exact dimensions — treat every dimension as
            unconfirmed until measured by hand.
          </div>
          <p style={{ marginTop: 24 }}>
            <Button
              onClick={() => undefined}
              aria-label="Upload a room photo (coming in the next milestone)"
            >
              Upload a photo
            </Button>
          </p>
        </section>
      </main>
      <footer className="app-footer">
        <div className="inner">
          Floor Plan Generator — local prototype. Images and results stay on your
          machine and are deleted after a short retention window.
        </div>
      </footer>
    </div>
  );
}
