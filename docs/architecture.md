# Architecture

## Overview

Floor Plan Generator is a two-process system:

- **Frontend** (React + TypeScript + Vite): presentation, user interaction, API
  communication only. It never holds credentials and never runs AI models.
- **Backend** (Python + FastAPI + LangGraph): image validation, computer-vision
  pipeline, spatial reasoning, uncertainty assessment, deterministic rendering,
  artifact management.

All AI reasoning and image processing happens server-side. The frontend receives
validated JSON and a rendered PNG.

## Processing pipeline

```text
RGB IMAGE
 → validate_input      decode, format/size/pixel checks, EXIF orientation
 → preprocess          RGB normalization, safe resize, artifacts saved
 → estimate_depth      relative depth cues (classical, CPU, no model download)
 → estimate_normals    surface normals derived from depth (assumed pinhole camera)
 → extract_features    deterministic region/edge/color analysis → visual evidence
 → reason_spatial      Groq (if configured) or deterministic evidence-based fallback
 → assess_uncertainty  per-object and global uncertainty, open questions
 → plan_viewpoint      (optional, disabled by default) pick an unresolved question
 → generate_view       (optional, disabled by default) image-conditioned provider
 → verify_candidate    (optional) sanity-check generated candidate
 → update_hypothesis   (optional) merge verified evidence, never overwrite OBSERVED
 → render_floorplan    deterministic Pillow 2D schematic
 → validate_output     schema validation before returning
```

The pipeline is a LangGraph `StateGraph`. Optional nodes are reached through
conditional edges; the viewpoint loop is bounded to at most 4 iterations and stops
early when uncertainty is sufficiently reduced or a provider fails.

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `app.core.config` | Pydantic-settings based configuration from env |
| `app.core.errors` | typed application errors → safe API responses |
| `app.schemas.spatial` | versioned spatial data contract (Pydantic) |
| `app.services.image_validation` | real image decoding and limit enforcement |
| `app.services.preprocessing` | orientation normalization, resize, color conversion |
| `app.services.depth` | `DepthEstimator` interface + classical CPU implementation |
| `app.services.normals` | depth → normals with documented camera convention |
| `app.services.features` | deterministic visual evidence extraction |
| `app.services.reasoning` | spatial hypothesis construction (Groq or fallback) |
| `app.services.renderer` | deterministic floor-plan PNG rendering |
| `app.tools.groq_client` | bounded Groq API client with retries |
| `app.tools.novel_view` | optional image-generation provider interface (disabled) |
| `app.nodes` | thin LangGraph node wrappers |
| `app.graph` | graph topology, conditional edges, termination |
| `app.api` | FastAPI routes, job lifecycle, artifact serving |

## Coordinate conventions

- **Image pixel coordinates**: `(x, y)`, origin top-left, x right, y down.
- **Camera-relative depth**: relative estimate, unitless, larger = farther.
- **Floor-plan coordinates**: `(x, y)` in a normalized plan space (0–1), origin
  top-left of the plan. Metric dimensions are **unknown** unless stated.
- Camera coordinate convention for normals: +X right, +Y down, +Z forward
  (optical axis), derived from assumed pinhole intrinsics.

## Job model

`POST /api/generations` creates an in-memory job (`queued → processing → completed | failed`)
processed by a background asyncio task. In-memory state does **not** survive restarts
and does **not** support multi-worker deployments — documented limitation for this
local prototype.

## Replaceability

- Depth: swap `DepthEstimator` implementation (e.g. MiDaS adapter) without touching
  the graph.
- Reasoning: Groq is behind an interface; the fallback is explicit and identified.
- Novel views: provider interface, disabled unless explicitly configured.

## Security posture

- Secrets only in `backend/.env` (git-ignored); `.env.example` holds placeholders.
- CORS restricted to the configured frontend origin.
- Upload limits (bytes and pixels) enforced after real decoding, never trusting
  filenames or browser MIME types.
- Artifacts are referenced by opaque job IDs, resolved inside a fixed root, and
  deleted after a TTL.
- Provider payloads and raw exceptions never reach the browser.
- This is a local prototype; basic safeguards do **not** make it production-secure.
