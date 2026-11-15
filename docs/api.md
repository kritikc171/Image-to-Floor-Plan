# API reference

Base URL (development): `http://127.0.0.1:8000`. OpenAPI docs at `/docs`.

## Conventions

- All responses are JSON unless noted.
- Error shape: `{"error": {"code": string, "message": string, "details": ...}}`
- Request IDs are echoed via `X-Request-ID` response header.

## `GET /api/health`

Health check. Never exposes configuration values or secrets.

**200**

```json
{"status": "ok", "version": "0.1.0"}
```

## `POST /api/generations`

Upload exactly one indoor-room photograph and start processing.

**Request**: `multipart/form-data`, field `file`, single image
(PNG/JPEG/WebP/GIF/BMP), max 15 MB, max 4096 px per dimension (configurable).

**Responses**:

- `202 Accepted` — job accepted:

```json
{
  "job_id": "job_01HF...",
  "status": "queued",
  "created_at": "2026-10-13T15:39:25Z"
}
```

- `400` — no file, multiple files, unsupported format, malformed image,
  oversize image.
- `413` — file exceeds byte limit or image dimensions exceed per-side limit.
- `415` — unsupported media type.
- `429` — too many concurrent jobs (configurable).

## `GET /api/generations/{job_id}`

Poll job status and result metadata.

**200**

```json
{
  "job_id": "job_01HF...",
  "status": "completed",
  "created_at": "...",
  "updated_at": "...",
  "error": null,
  "result": { ... }   // present when status == "completed"
}
```

`status` is one of `queued`, `processing`, `completed`, `failed`.
`error` is `null` or `{"code": "...", "message": "..."}` (safe, no internals).
Processing is asynchronous; poll this endpoint until status reaches a terminal
state.

### Result schema (v1)

The completed result is a `GenerationResult` object nested under `result`:

```jsonc
{
  "schema_version": "1.0",
  "job_id": "job_01HF...",
  "created_at": "2026-10-13T15:39:25Z",
  "input": {"width_px": 1920, "height_px": 1080, "format": "PNG"},
  "hypothesis": {
    "schema_version": "1.0",
    "room_boundary": {
      "polygon": [{"x": 0.1, "y": 0.2}, ...],
      "evidence_status": "INFERRED",
      "uncertainty_reason": "boundary estimated from visible floor region"
    },
    "room_geometry": {
      "perimeter": [{"x": 0.1, "y": 0.2}, ...],
      "corners": [{"x": 0.1, "y": 0.2}, ...],
      "walls": [...],
      "doors": [],
      "windows": [],
      "evidence_status": "INFERRED",
      "uncertainty_reason": "...",
      "rectangular_assumption": true
    },
    "geometry_diagnostics": {
      "geometry_source": "floorplane-estimator",
      "source_ground_corners": [[...], ...],
      "homography_applied": true,
      "correction_method": "projective homography",
      "fallback_reason": null,
      "corrected_room_corners": [...],
      "renderer_perimeter": [...],
      "renderer_wall_segments": [...]
    },
    "objects": [
      {
        "id": "object-001",
        "type": "furniture",
        "label": "dining table",
        "image_bbox": {"x_min": 0.2, "y_min": 0.6, "x_max": 0.5, "y_max": 0.8},
        "position": {"x": 0.4, "y": 0.6},
        "orientation": 0.0,
        "dimensions": null,
        "dimensions_confidence": null,
        "footprint": {"width": 0.34, "depth": 0.34, "approximate": true, "basis": "..."},
        "footprint_polygon": [...],
        "footprint_confidence": 0.62,
        "ground_contact": {"x": 0.4, "y": 0.8},
        "ground_confidence": 0.75,
        "confidence": 0.62,
        "placement_confidence": 0.62,
        "evidence_status": "OBSERVED",
        "evidence_source": ["torchvision-ssdlite-coco"],
        "uncertainty_reason": "Class 'dining table' detected in the photograph...",
        "layout_issues": null
      }
    ],
    "openings": [
      {
        "id": "opening-001",
        "kind": "opening",
        "side": null,
        "position": {"x": 0.05, "y": 0.5},
        "width": null,
        "swing": null,
        "swing_degrees": null,
        "wall_id": null,
        "evidence_status": "UNCERTAIN",
        "uncertainty_reason": "Border-adjacent edge candidate; not confirmed as a door or window."
      }
    ],
    "uncertainty": {
      "overall": "high",
      "assumptions": [
        "Single RGB photograph without camera intrinsics",
        "Camera optical axis assumed parallel to the floor",
        "Depth is a unitless relative estimate, not metric depth",
        "Object labels are generic geometric descriptions",
        "Heuristic scores are not calibrated probabilities"
      ],
      "open_questions": ["Where are the room boundaries?"]
    },
    "provenance": {
      "reasoning_provider": "deterministic-fallback",
      "depth_method": "classical-relative (horizon + ground-plane cue)",
      "feature_method": "classical-cv (gradient regions + floor segmentation)",
      "detection_model": "ssdlite320_mobilenet_v3_large (COCO-2017)",
      "generated_views_used": 0,
      "model_name": null
    }
  },
  "floorplan_image_url": "/api/generations/{job_id}/floorplan"
}
```

### Evidence statuses

- `OBSERVED` — directly supported by the photograph (a pretrained detector hit).
- `INFERRED` — estimated from visible evidence under stated assumptions.
- `GENERATED` — derived from an AI-generated candidate view. **Not used by the
  current pipeline; `generated_views_used` is always `0`.**
- `UNCERTAIN` — not enough evidence; position or dimensions unknown.

Precedence: OBSERVED > INFERRED > GENERATED > UNCERTAIN. Detection and spatial
placement are different concepts: an object whose class is `OBSERVED` may still
have a floor position that is `INFERRED` or `UNCERTAIN`.

## `GET /api/generations/{job_id}/floorplan`

Returns the rendered floor plan as **PNG** (`image/png`).

- `200` — PNG bytes (1400 × 1000).
- `404` — unknown job.
- `409` — job exists but the floor plan is not ready yet (not completed).

## `POST /api/generations/{job_id}/ask`

Ask a simple question about a completed floor plan. This is a **deterministic
text matcher**, not an LLM — it answers counts, positions, confidence,
observed/uncertain objects, and limitations from the hypothesis JSON.

**Request**: `{"question": "..."}` (max 500 characters).

## `GET /api/health`

Health check. Returns `{"status": "ok", "version": "0.1.0"}`.

## Notes

- Jobs and artifacts are in-memory/local and expire after a TTL
  (`ARTIFACT_TTL_SECONDS`, default 3600 s). Do not rely on persistence.
- Processing is asynchronous; poll `GET /api/generations/{job_id}`.
