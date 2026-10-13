/**
 * Generation state machine for the upload → result journey.
 * All AI work happens on the backend; this hook only orchestrates
 * API communication and presentation state.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  createGeneration,
  getGeneration,
  type GenerationResult,
} from "../api/client";

/** Mirrors backend MAX_UPLOAD_BYTES default (15 MiB). Server is source of truth. */
export const MAX_FILE_SIZE = 15 * 1024 * 1024;
export const ACCEPTED_TYPES = [
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/bmp",
  "image/gif",
];

export type WorkspacePhase =
  | { kind: "empty" }
  | { kind: "selected"; file: File; previewUrl: string; error: string | null }
  | { kind: "submitting" }
  | { kind: "processing"; jobId: string }
  | { kind: "completed"; jobId: string; result: GenerationResult }
  | {
      kind: "failed";
      message: string;
      retryable: boolean;
      file: File | null;
    };

const POLL_INTERVAL_MS = 1000;
const POLL_TIMEOUT_MS = 5 * 60 * 1000;

export function validateFile(file: File): string | null {
  if (!ACCEPTED_TYPES.includes(file.type)) {
    return "Please choose a PNG, JPEG, WebP, BMP, or GIF photo.";
  }
  if (file.size === 0) {
    return "The selected file is empty.";
  }
  if (file.size > MAX_FILE_SIZE) {
    return "The photo is larger than 15 MB. Please choose a smaller image.";
  }
  return null;
}

export function useGeneration() {
  const [phase, setPhase] = useState<WorkspacePhase>({ kind: "empty" });
  const pollTimer = useRef<number | null>(null);
  const startedAt = useRef<number>(0);
  const lastFile = useRef<File | null>(null);
  const phaseRef = useRef<WorkspacePhase>({ kind: "empty" });
  phaseRef.current = phase;

  const clearPoll = useCallback(() => {
    if (pollTimer.current !== null) {
      window.clearTimeout(pollTimer.current);
      pollTimer.current = null;
    }
  }, []);

  // Revoke object URLs on unmount.
  useEffect(() => () => clearPoll(), [clearPoll]);

  const selectFile = useCallback((file: File) => {
    const error = validateFile(file);
    const previewUrl = URL.createObjectURL(file);
    lastFile.current = file;
    setPhase({ kind: "selected", file, previewUrl, error });
  }, []);

  const removeFile = useCallback(() => {
    setPhase((p) => {
      if (p.kind === "selected") URL.revokeObjectURL(p.previewUrl);
      return { kind: "empty" };
    });
  }, []);

  const replaceFile = useCallback((file: File) => {
    setPhase((p) => {
      if (p.kind === "selected") URL.revokeObjectURL(p.previewUrl);
      return { kind: "empty" };
    });
    selectFile(file);
  }, [selectFile]);

  const poll = useCallback(
    (jobId: string) => {
      const tick = async () => {
        try {
          const status = await getGeneration(jobId);
          if (status.status === "completed" && status.result) {
            setPhase({ kind: "completed", jobId, result: status.result });
            return;
          }
          if (status.status === "failed") {
            setPhase({
              kind: "failed",
              message: status.error?.message ?? "Processing failed.",
              retryable: true,
              file: null,
            });
            return;
          }
          if (Date.now() - startedAt.current > POLL_TIMEOUT_MS) {
            setPhase({
              kind: "failed",
              message: "Processing timed out. Please try again.",
              retryable: true,
              file: null,
            });
            return;
          }
          pollTimer.current = window.setTimeout(tick, POLL_INTERVAL_MS);
        } catch (err) {
          const message =
            err instanceof Error ? err.message : "Could not reach the server.";
          setPhase({ kind: "failed", message, retryable: true, file: null });
        }
      };
      pollTimer.current = window.setTimeout(tick, POLL_INTERVAL_MS);
    },
    [],
  );

  const reset = useCallback(() => {
    clearPoll();
    setPhase((p) => {
      if (p.kind === "selected") URL.revokeObjectURL(p.previewUrl);
      return { kind: "empty" };
    });
  }, [clearPoll]);

  const submit = useCallback(
    async (file: File) => {
      clearPoll();
      lastFile.current = file;
      setPhase({ kind: "submitting" });
      try {
        const created = await createGeneration(file);
        startedAt.current = Date.now();
        setPhase({ kind: "processing", jobId: created.job_id });
        poll(created.job_id);
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Could not reach the server.";
        setPhase({
          kind: "failed",
          message,
          retryable: true,
          file,
        });
      }
    },
    [clearPoll, poll],
  );

  const retry = useCallback(() => {
    const file = lastFile.current;
    if (file) {
      void submit(file);
    } else {
      reset();
    }
  }, [submit, reset]);

  return {
    phase,
    selectFile,
    removeFile,
    replaceFile,
    submit,
    retry,
    reset,
  };
}
