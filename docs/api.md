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
(pPNG/JPEG/WebP/GIF/BMP), max 15 MB, max 4096 px per dimension (configurable).

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
- `413` — file exceeds byte limit.
- `415` — unsupported media type.
- `429` — too many concurrent jobs (configurable).
- `503` — processing backend unavailable.

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

### Result schema (v1)

```jsonc
{
  "schema_version": "1.0",
  "job_id": "job_01HF...",
  "created_at": "...",
  "input": {"width_px": 1920, "height_px": 1080, "format": "png"},
  "room_boundary": {
    "polygon": [{"x": 0.1, "y": 0.2}, ...],   // normalized 0–1 plan space
    "evidence_status": "INFERRED",
    "uncertainty_reason": "boundary estimated from dominant edges"
  },
  "objects": [
    {
      "id": "object-001",
      "type": "furniture",
      "label": "bed",
      "position": {"x": 0.4, "y": 0.6},
      "orientation": 0.0,
      "dimensions": null,
      "evidence_status": "UNCERTAIN",
      "confidence": null,
      "evidence_source": ["classical-cv"],
      "uncertainty_reason": "Insufficient evidence to estimate floor position"
    }
  ],
  "openings": [
    {"id": "opening-001", "kind": "door", "side": "left",
     "position": {"x": 0.05, "y": 0.5}, "evidence_status": "UNCERTAIN"}
  ],
  "uncertainty": {
    "overall": "high",
    "assumptions": ["pinhole camera, no metric intrinsics available", "..."],
    "open_questions": ["Is there a window on the north wall?"]
  },
  "provenance": {"reasoning_provider": "groq|deterministic-fallback",
                 "depth_method": "classical-relative", "generated_views_used": 0},
  "floorplan_image_url": "/api/generations/{job_id}/floorplan"
}
```

### Evidence statuses

- `OBSERVED` — directly supported by the original photograph.
- `INFERRED` — estimated from evidence and stated assumptions.
- `GENERATED` — derived from an AI-generated candidate view (only when the optional
  provider is enabled).
- `UNCERTAIN` — insufficient or conflicting evidence.

Precedence: OBSERVED > INFERRED > GENERATED > UNCERTAIN. Observed evidence is
never overwritten by generated hypotheses.

## `GET /api/generations/{job_id}/floorplan`

Returns the rendered floor plan as **PNG** (`image/png`).

- `200` — PNG bytes.
- `404` — unknown job or floor plan not yet rendered.

## Notes

- Jobs and artifacts are in-memory/local and expire after a TTL
  (`ARTIFACT_TTL_SECONDS`, default 3600 s). Do not rely on persistence.
- Processing is asynchronous; poll `GET /api/generations/{job_id}`.
