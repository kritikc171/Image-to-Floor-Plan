import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import WorkspacePage from "../src/pages/WorkspacePage";
import {
  createGeneration,
  getGeneration,
  type GenerationStatusResponse,
  type GenerationResult,
} from "../src/api/client";

vi.mock("../src/api/client", () => ({
  createGeneration: vi.fn(),
  getGeneration: vi.fn(),
  floorplanUrl: vi.fn((jobId: string) => `/api/generations/${jobId}/floorplan`),
}));

const makeFile = (name = "room.png", type = "image/png") =>
  new File([new Uint8Array([1, 2, 3, 4])], name, { type });

const pickFile = (input: HTMLInputElement, file: File) => {
  fireEvent.change(input, { target: { files: [file] } });
};

const fileInputs = () =>
  Array.from(
    document.querySelectorAll('input[type="file"]'),
  ) as HTMLInputElement[];

const mockResult: GenerationResult = {
  schema_version: "1.0",
  job_id: "job_test",
  created_at: "2026-10-13T00:00:00Z",
  input: { width_px: 640, height_px: 480, format: "PNG" },
  hypothesis: {
    schema_version: "1.0",
    room_boundary: {
      polygon: null,
      evidence_status: "UNCERTAIN",
      uncertainty_reason: "test",
    },
    objects: [
      {
        id: "object-001",
        type: "furniture",
        label: "table",
        position: null,
        orientation: null,
        dimensions: null,
        image_bbox: null,
        placement_confidence: null,
        evidence_status: "INFERRED",
        confidence: null,
        evidence_source: ["test"],
        uncertainty_reason: "test",
        footprint: null,
        layout_issues: null,
      },
    ],
    openings: [],
    uncertainty: {
      overall: "high",
      assumptions: ["single photo"],
      open_questions: ["boundary?"],
    },
    provenance: {
      reasoning_provider: "deterministic-fallback",
      depth_method: "test",
      feature_method: "test",
      generated_views_used: 0,
      model_name: null,
      detection_model: null,
    },
  },
  floorplan_image_url: "/api/generations/job_test/floorplan",
};

const completedStatus = (): GenerationStatusResponse => ({
  job_id: "job_test",
  status: "completed",
  created_at: "2026-10-13T00:00:00Z",
  updated_at: "2026-10-13T00:00:01Z",
  error: null,
  result: mockResult,
});

describe("WorkspacePage journey", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows the empty upload state", () => {
    render(<WorkspacePage />);
    expect(
      screen.getByRole("button", { name: /upload a photo of an indoor room/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/drag and drop a room photo/i)).toBeInTheDocument();
  });

  it("validates file type client-side", () => {
    render(<WorkspacePage />);
    const input = fileInputs()[0];
    pickFile(input, makeFile("notes.txt", "text/plain"));
    expect(
      screen.getByText(/please choose a png, jpeg, webp, bmp, or gif/i),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /generate floor plan/i }),
    ).toBeDisabled();
  });

  it("validates oversize files client-side", () => {
    render(<WorkspacePage />);
    const big = new File([new Uint8Array(16 * 1024 * 1024)], "big.png", {
      type: "image/png",
    });
    Object.defineProperty(big, "size", { value: 16 * 1024 * 1024 });
    const input = fileInputs()[0];
    pickFile(input, big);
    expect(screen.getByText(/larger than 15 mb/i)).toBeInTheDocument();
  });

  it("previews a valid selection and generates a floor plan", async () => {
    vi.mocked(createGeneration).mockResolvedValue({
      job_id: "job_test",
      status: "queued",
      created_at: "2026-10-13T00:00:00Z",
    });
    vi.mocked(getGeneration).mockResolvedValue(completedStatus());

    render(<WorkspacePage />);
    pickFile(fileInputs()[0], makeFile());
    expect(screen.getByAltText(/preview of room\.png/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /generate floor plan/i }));

    await waitFor(
      () => {
        expect(
          screen.getByRole("button", { name: /download png/i }),
        ).toBeInTheDocument();
      },
      { timeout: 5000 },
    );
    expect(
      screen.getByRole("img", { name: /approximate top-down floor plan/i }),
    ).toHaveAttribute("src", "/api/generations/job_test/floorplan");
    expect(screen.getByText("table")).toBeInTheDocument();
    expect(
      screen.getByText(/schematic approximation from a single photo/i),
    ).toBeInTheDocument();
  });

  it("downloads the floor plan PNG", async () => {
    vi.mocked(createGeneration).mockResolvedValue({
      job_id: "job_test",
      status: "queued",
      created_at: "2026-10-13T00:00:00Z",
    });
    vi.mocked(getGeneration).mockResolvedValue(completedStatus());

    const blob = new Blob(["pngbytes"], { type: "image/png" });
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      blob: () => Promise.resolve(blob),
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<WorkspacePage />);
    pickFile(fileInputs()[0], makeFile());
    fireEvent.click(screen.getByRole("button", { name: /generate floor plan/i }));

    await waitFor(
      () => {
        expect(
          screen.getByRole("button", { name: /download png/i }),
        ).toBeInTheDocument();
      },
      { timeout: 5000 },
    );

    fireEvent.click(screen.getByRole("button", { name: /download png/i }));
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/generations/job_test/floorplan",
      );
    });
  });

  it("shows failure state with retry after processing failure", async () => {
    vi.mocked(createGeneration).mockResolvedValue({
      job_id: "job_test",
      status: "queued",
      created_at: "2026-10-13T00:00:00Z",
    });
    vi.mocked(getGeneration).mockResolvedValue({
      job_id: "job_test",
      status: "failed",
      created_at: "2026-10-13T00:00:00Z",
      updated_at: "2026-10-13T00:00:01Z",
      error: { code: "processing_failed", message: "Image processing failed." },
      result: null,
    });

    render(<WorkspacePage />);
    pickFile(fileInputs()[0], makeFile());
    fireEvent.click(screen.getByRole("button", { name: /generate floor plan/i }));

    await waitFor(
      () => {
        expect(screen.getByRole("alert")).toBeInTheDocument();
      },
      { timeout: 5000 },
    );
    expect(screen.getByText("Image processing failed.")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /try again/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /upload a different photo/i }),
    ).toBeInTheDocument();
  });

  it("shows network failure when the API is unreachable", async () => {
    vi.mocked(createGeneration).mockRejectedValue(new Error("Cannot reach the server."));
    render(<WorkspacePage />);
    pickFile(fileInputs()[0], makeFile());
    fireEvent.click(screen.getByRole("button", { name: /generate floor plan/i }));
    await waitFor(
      () => {
        expect(screen.getByText(/cannot reach the server/i)).toBeInTheDocument();
      },
      { timeout: 5000 },
    );
    expect(screen.getByRole("button", { name: /try again/i })).toBeInTheDocument();
  });
});
