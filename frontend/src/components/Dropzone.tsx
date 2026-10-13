/** Accessible drag-and-drop / click upload zone. */

import { useCallback, useRef, useState } from "react";

export function Dropzone(props: {
  onFile: (file: File) => void;
  disabled?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const dragDepth = useRef(0);

  const open = useCallback(() => {
    if (!props.disabled) inputRef.current?.click();
  }, [props.disabled]);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const first = files?.[0];
      if (first) props.onFile(first);
    },
    [props],
  );

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label="Upload a photo of an indoor room"
      aria-disabled={props.disabled}
      className={`dropzone${dragging ? " dropzone--dragging" : ""}`}
      onClick={open}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          open();
        }
      }}
      onDragEnter={(e) => {
        e.preventDefault();
        dragDepth.current += 1;
        setDragging(true);
      }}
      onDragLeave={(e) => {
        e.preventDefault();
        dragDepth.current = Math.max(0, dragDepth.current - 1);
        if (dragDepth.current === 0) setDragging(false);
      }}
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        dragDepth.current = 0;
        setDragging(false);
        handleFiles(e.dataTransfer.files);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp,image/bmp,image/gif"
        className="dropzone__input"
        tabIndex={-1}
        aria-hidden="true"
        onChange={(e) => {
          handleFiles(e.target.files);
          e.target.value = "";
        }}
      />
      <svg
        className="dropzone__icon"
        viewBox="0 0 48 48"
        aria-hidden="true"
        width={48}
        height={48}
      >
        <path
          d="M24 34V14m0 0l-8 8m8-8l8 8"
          stroke="currentColor"
          strokeWidth={3}
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
        />
        <path
          d="M10 34v4a4 4 0 0 0 4 4h20a4 4 0 0 0 4-4v-4"
          stroke="currentColor"
          strokeWidth={3}
          strokeLinecap="round"
          fill="none"
        />
      </svg>
      <p className="dropzone__title">Drag and drop a room photo here</p>
      <p className="dropzone__hint">
        or press Enter / Space to browse — one PNG, JPEG, WebP, BMP, or GIF,
        up to 15 MB
      </p>
    </div>
  );
}
