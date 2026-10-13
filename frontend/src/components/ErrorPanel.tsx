/** Processing failure with retry / new-image actions. */

export function ErrorPanel(props: {
  message: string;
  retryable: boolean;
  onRetry: () => void;
  onReset: () => void;
}) {
  return (
    <section className="error-panel" role="alert">
      <h2>Processing failed</h2>
      <p>{props.message}</p>
      <div className="error-panel__actions">
        {props.retryable && (
          <button
            type="button"
            className="button button--primary"
            onClick={props.onRetry}
          >
            Try again
          </button>
        )}
        <button
          type="button"
          className="button button--secondary"
          onClick={props.onReset}
        >
          Upload a different photo
        </button>
      </div>
    </section>
  );
}
