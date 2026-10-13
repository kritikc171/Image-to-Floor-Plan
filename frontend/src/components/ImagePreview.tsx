/** Selected-image preview with replace / remove actions. */

export function ImagePreview(props: {
  previewUrl: string;
  fileName: string;
  onReplace: () => void;
  onRemove: () => void;
}) {
  return (
    <figure className="preview">
      <img src={props.previewUrl} alt={`Preview of ${props.fileName}`} />
      <figcaption className="preview__meta">
        <span className="preview__name">{props.fileName}</span>
        <span className="preview__actions">
          <button
            type="button"
            className="button button--secondary"
            onClick={props.onReplace}
          >
            Replace
          </button>
          <button
            type="button"
            className="button button--secondary"
            onClick={props.onRemove}
          >
            Remove
          </button>
        </span>
      </figcaption>
    </figure>
  );
}
