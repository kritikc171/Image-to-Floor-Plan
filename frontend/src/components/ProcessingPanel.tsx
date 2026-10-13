/** Processing state with accessible progress feedback. */

export function ProcessingPanel(props: { fileName: string }) {
  return (
    <section className="processing" aria-label="Processing your photo">
      <div
        className="spinner"
        role="progressbar"
        aria-valuetext="Processing"
        aria-label="Processing your photo"
      />
      <h2>Analyzing your photo</h2>
      <p className="processing__file">{props.fileName}</p>
      <p className="processing__hint">
        This usually takes a few seconds. You can close this page and come
        back — the result is not stored long-term.
      </p>
    </section>
  );
}
