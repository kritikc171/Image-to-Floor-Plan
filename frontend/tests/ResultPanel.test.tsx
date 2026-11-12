import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import ResultPanel from "../src/components/ResultPanel";
import { type GenerationResult } from "../src/api/client";

vi.mock("../src/api/client", () => ({
  floorplanUrl: vi.fn((jobId: string) => `/api/generations/${jobId}/floorplan`),
}));

const makeResult = (): GenerationResult => ({
  schema_version: "1.0",
  job_id: "job_test",
  created_at: "2026-10-13T00:00:00Z",
  input: { width_px: 640, height_px: 480, format: "PNG" },
  hypothesis: {
    schema_version: "1.0",
    room_boundary: {
      polygon: [{ x: 0.05, y: 0.1 }, { x: 0.95, y: 0.1 }, { x: 0.95, y: 0.9 }, { x: 0.05, y: 0.9 }],
      evidence_status: "INFERRED",
      uncertainty_reason: "test",
    },
    objects: [
      {
        id: "object-001",
        type: "furniture",
        label: "dining table",
        position: { x: 0.4, y: 0.5 },
        orientation: 0,
        dimensions: null,
        image_bbox: { x_min: 0.2, y_min: 0.6, x_max: 0.5, y_max: 0.8 },
        placement_confidence: 0.8,
        evidence_status: "OBSERVED",
        confidence: 0.82,
        evidence_source: ["torchvision-ssdlite-coco"],
        uncertainty_reason: "Detected in the photograph.",
        footprint: { width: 0.34, depth: 0.34, approximate: true, basis: "test" },
        layout_issues: null,
      },
      {
        id: "object-002",
        type: "furniture",
        label: "chair",
        position: null,
        orientation: null,
        dimensions: null,
        image_bbox: null,
        placement_confidence: null,
        evidence_status: "UNCERTAIN",
        confidence: null,
        evidence_source: ["classical-cv"],
        uncertainty_reason: "No floor position.",
        footprint: { width: 0.14, depth: 0.14, approximate: true, basis: "test" },
        layout_issues: ["no floor position; shown as unplaced"],
      },
    ],
    openings: [],
    uncertainty: {
      overall: "medium",
      assumptions: ["single photo assumption"],
      open_questions: ["Where are the boundaries?"],
    },
    provenance: {
      reasoning_provider: "deterministic-fallback",
      depth_method: "classical-relative",
      feature_method: "classical-cv",
      detection_model: "ssdlite320_mobilenet_v3_large (COCO-2017)",
      generated_views_used: 0,
      model_name: null,
    },
  },
  floorplan_image_url: "/api/generations/job_test/floorplan",
});

describe("ResultPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders detected objects with labels, confidence and status chips", () => {
    render(<ResultPanel jobId="job_test" result={makeResult()} onReset={vi.fn()} />);
    expect(screen.getByText("dining table")).toBeInTheDocument();
    expect(screen.getByText("chair")).toBeInTheDocument();
    expect(screen.getByText("82%")).toBeInTheDocument();
    expect(screen.getByText("Observed")).toBeInTheDocument();
    expect(screen.getByText("Uncertain")).toBeInTheDocument();
  });

  it("renders the evidence legend with all four statuses", () => {
    render(<ResultPanel jobId="job_test" result={makeResult()} onReset={vi.fn()} />);
    expect(screen.getByText(/directly visible in your photo/i)).toBeInTheDocument();
    expect(screen.getByText(/estimated from visible evidence/i)).toBeInTheDocument();
    expect(screen.getByText(/derived from an AI-generated view/i)).toBeInTheDocument();
    expect(screen.getByText(/not enough evidence to decide/i)).toBeInTheDocument();
  });

  it("renders the Q&A input and Send button", () => {
    render(<ResultPanel jobId="job_test" result={makeResult()} onReset={vi.fn()} />);
    expect(screen.getByPlaceholderText(/ask a question about this floor plan/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /send/i })).toBeInTheDocument();
  });

  it("renders download buttons", () => {
    render(<ResultPanel jobId="job_test" result={makeResult()} onReset={vi.fn()} />);
    expect(screen.getByRole("button", { name: /download png/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /download json layout/i })).toBeInTheDocument();
  });

  it("calls floorplanUrl when downloading PNG", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      blob: () => Promise.resolve(new Blob(["bytes"], { type: "image/png" })),
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<ResultPanel jobId="job_test" result={makeResult()} onReset={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: /download png/i }));
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith("/api/generations/job_test/floorplan");
    });
  });
});
