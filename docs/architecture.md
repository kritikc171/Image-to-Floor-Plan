# Architecture

## Overview

Floor Plan Generator is a two-process system:

- **Frontend** (React + TypeScript + Vite): presentation, user interaction, API
  communication only. It never holds credentials and never runs AI models.
- **Backend** (Python + FastAPI): image validation, local computer vision,
  spatial-hypothesis construction, uncertainty reporting, deterministic
  rendering, and artifact management.

Image processing and object detection happen server-side. The frontend receives
validated JSON and a rendered PNG; it does not run models or hold provider keys.

## Processing pipeline

```text
RGB IMAGE
 → validate_and_normalize  decode, format/size/pixel checks, EXIF orientation
 → estimate_depth          relative depth cues (classical, CPU)
 → estimate_normals        surface normals derived from relative depth
 → extract_features        deterministic region/edge/color analysis
 → estimate_floor_plane    segment visible floor and project its outline
 → rectify_floor_quad      homography maps usable quadrilaterals; a degenerate
                            quadrilateral falls back to an axis-aligned rectangle
                            with a bounded, explicitly approximate aspect ratio
 → normalize_room_plan     final guard rectifies any remaining non-rectangular
                            outline and composes the transform for object points
 → detect_objects          local Torchvision SSDlite-MobileNetV3 detector
 → build_hypothesis        estimate oriented footprints, keep them inside the
                            room, and search for the nearest clear placement;
                            leave an object unplaced if no clear position exists
 → render_floorplan        deterministic Pillow 2D schematic PNG
 → validate_output         Pydantic response models
```

The current `run_pipeline` implementation calls these services directly. It
uses deterministic hypothesis construction; configured Groq and novel-view
settings are not invoked by this processing path.

The floor outline is a visible-floor estimate, not a verified room perimeter.
The homography uses the visible quadrilateral and its projected edge proportions
to produce a cleaner rectangular approximation; when homography is unavailable,
the axis-aligned ground-plane bounds provide a labeled rectangular approximation
with its aspect ratio clamped to avoid a degenerate sliver.
Neither method recovers hidden walls or proves that the real room is rectangular.
As a final guard, any non-rectangular plan outline is normalized before the
hypothesis is built; the same composed transform is applied to object floor
positions, wall cues, and openings. The result JSON includes geometry diagnostics:
source ground-plane corners, correction method/fallback reason, corrected plan
corners, and the perimeter/wall segments passed to the renderer.
The renderer uses equal pixel scale on both plan axes to avoid stretching the room
or furniture symbols.

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `app.core.config` | Pydantic-settings based configuration from env |
| `app.core.errors` | typed application errors → safe API responses |
| `app.schemas.spatial` | versioned spatial data contract (Pydantic) |
| `app.services.pipeline` | orchestration of the baseline image-to-floor-plan stages |
| `app.services.image_validation` | real image decoding, orientation, and limit enforcement |
| `app.services.depth` | `DepthEstimator` interface + classical CPU implementation |
| `app.services.floorplane` | visible-floor segmentation and projective plan normalization |
| `app.services.normals` | depth → normals with documented camera convention |
| `app.services.features` | deterministic visual evidence extraction; rejects regions spanning opposite image borders to avoid treating photo-wide background as furniture |
| `app.services.detection` | local pretrained Torchvision object detection |
| `app.services.hypothesis` | deterministic object placement and uncertainty construction |
| `app.services.layout` | footprint fitting, rotated bounds, room containment, and nearest-clear placement |
| `app.services.room_geometry` | wall perimeter and uncertain opening attachment |
| `app.services.renderer` | deterministic floor-plan PNG rendering |
| `app.api` | FastAPI routes, job lifecycle, artifact serving |
| `app.schemas` | validated API and spatial data contracts |

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

- Depth, features, and detection use interfaces so implementations can be replaced
  without changing the route or result schema.
- The current pipeline uses deterministic hypothesis construction; optional
  provider settings are not invoked by `run_pipeline`.

## Security posture

- Secrets only in `backend/.env` (git-ignored); `.env.example` holds placeholders.
- CORS restricted to the configured frontend origin.
- Upload limits (bytes and pixels) enforced after real decoding, never trusting
  filenames or browser MIME types.
- Artifacts are referenced by opaque job IDs, resolved inside a fixed root, and
  deleted after a TTL.
- Provider payloads and raw exceptions never reach the browser.
- This is a local prototype; basic safeguards do **not** make it production-secure.
