# Floor Plan Generator

A web application that accepts **one RGB photograph of an indoor room** and produces an
**approximate 2D top-down floor plan** as a structured spatial hypothesis plus a
downloadable schematic PNG.

> **This is an approximation system, not an architectural measurement tool.**
> A single RGB image cannot recover hidden geometry or exact metric dimensions.
> The pipeline distinguishes **observed** evidence from **inferred** and **uncertain**
> elements, and never presents inferred geometry as verified.

## Background

### Problem context

Converting a photograph of an indoor room into a usable floor plan is a
spatial-reconstruction problem. A single RGB image records only the light that
reached the camera from a fixed viewpoint: surfaces facing the camera are
visible, but anything behind them, outside the frame, or occluded by furniture
is absent. Conventional object detection can label what is visible (a chair, a
table), but it cannot by itself establish where those objects sit on the floor,
how large the room is, or where walls run — especially walls not facing the
camera.

Manually creating a floor plan requires measuring wall lengths, room proportions,
and furniture positions with a tape measure, then translating those measurements
into a top-down drawing. For a single room this takes minutes to hours; for an
entire property it can take days. The result depends on the measurer's skill and
care.

This project addresses a narrower problem: given one photograph of an indoor
room, produce a **schematic top-down floor plan** that estimates room boundary,
visible furniture positions, and footprint sizes — while explicitly reporting
which parts are observed, which are inferred, and which are uncertain.

### Information categories in the output

Every element in the generated floor plan is assigned one of four evidence
statuses:

| Status | Meaning | Basis |
|---|---|---|
| **Observed** | Directly visible in the photograph | Pretrained object detector matched a COCO class (bed, couch, chair, dining table, TV, etc.) |
| **Inferred** | Estimated from visible evidence under stated assumptions | Shape-based candidate from classical CV; floor position projected from image evidence through an estimated floor plane |
| **Generated** | Derived from an AI-generated candidate view | Not used by the current pipeline — always 0 generated views |
| **Uncertain** | Not enough evidence to decide | Object above floor level, floor plane unavailable, opening candidate not confirmed |

### What the system does NOT do

- **No metric dimensions.** A single RGB photo without camera intrinsics or a
  known reference object cannot yield reliable meters. Room dimensions and object
  sizes are reported as normalized plan units (0–1), never fabricated as meters
  or feet.
- **No hidden geometry.** Walls behind the camera, occluded objects, and closed
  doors are never reconstructed. Hidden objects are never labeled as detected.
- **No guaranteed completeness.** Detectors are heuristic; missed furniture is
  normal. The COCO detector does not recognize wardrobes, cabinets, doors, or
  windows as those classes.
- **Depth is relative, not metric.** The default depth estimator uses classical
  image cues (horizon/gradient geometry). It is a unitless relative estimate
  used only for coarse near/far ordering and floor-plane projection.
- **Rectangular geometry is an approximation.** A projective homography maps a
  visible floor quadrilateral to straight-edged plan coordinates, but this does
  not verify that the real room is rectangular or recover unseen corners.

### Intended use cases

- Producing a rough, schematic floor plan from a single room photo as a
  starting sketch
- Understanding approximate furniture layout and room proportions from one image
- Educational or exploratory use where exact measurements are not required

### Not suitable for

- Construction, renovation, or real-estate documentation
- Any application requiring metric accuracy or verified wall positions
- Rooms where most furniture or walls are occluded or outside the frame

## Quick start

Run the backend and frontend in **two separate PowerShell terminals**. In each
terminal, start from the project root.

### Prerequisites
- Python 3.13 (verified with 3.13.12; required by the pinned CV/AI stack)
- Node.js 24 and npm 12 (verified)
- Git

### Backend

In Terminal 1:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# optional: copy .env.example to .env and set GROQ_API_KEY (not required)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Frontend

In Terminal 2, starting from the project root:

```powershell
cd frontend
npm install
npm run dev        # http://localhost:5173
```

### Tests

Run each test suite from a separate terminal, starting from the project root.

Backend:

```powershell
cd backend
python -m pytest
```

Frontend:

```powershell
cd frontend
npm test
```

## Repository layout

```text
├── frontend/        React + TypeScript + Vite application (presentation only)
├── backend/         FastAPI + Python computer-vision processing pipeline
│   └── app/
│       ├── api/         HTTP routes
│       ├── core/        configuration, logging, error handling
│       ├── schemas/     Pydantic spatial + API contracts
│       ├── services/    CV stages (validation, depth, detection, floor plane, layout, renderer)
│       └── main.py      FastAPI application factory
├── artifacts/       local, temporary processing artifacts (git-ignored)
└── docs/            architecture, API, limitations
```

## Documentation

- [Architecture](docs/architecture.md) — pipeline, modules, data flow
- [API reference](docs/api.md) — endpoints, schemas, status model, errors
- [Limitations](docs/limitations.md) — what the system cannot guarantee

## Configuration

Backend settings live in `backend/.env` (see `backend/.env.example`). The baseline
pipeline works **without any API key**: the Groq reasoning step is optional and a
deterministic, evidence-based fallback is used when no key is configured. Optional
novel-view generation is disabled by default.

No credentials are ever exposed to the frontend bundle.
