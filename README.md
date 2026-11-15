# Floor Plan Generator

A web application that accepts **one RGB photograph of an indoor room** and produces an
**approximate 2D top-down floor plan** as a structured spatial hypothesis plus a
downloadable schematic PNG.

> **This is an approximation system, not an architectural measurement tool.**
> A single RGB image cannot recover hidden geometry or exact metric dimensions.
> The pipeline distinguishes **observed** evidence from **inferred** and **uncertain**
> elements, and never presents inferred geometry as verified.

## Recent improvements (Nov 14–15, 2026)

The existing processing pipeline and upload workflow have been improved; the
project has not been rebuilt or had its API replaced.

- **Rectangular room outline:** when the visible floor provides usable geometry,
  the backend uses a projective transform to map it to straight-edged plan
  coordinates. If that geometry is unsuitable, it uses a labeled rectangular
  approximation when possible. Object positions are transformed along with the
  room outline so they stay in the same plan coordinate system. The result also
  includes geometry diagnostics describing the correction and any fallback.
- **Furniture placement:** object positions are estimated from their image
  bounding boxes and projected floor-contact points. The layout step accounts
  for rotated footprints, keeps furniture inside the estimated room, and
  searches for a nearby clear position when objects would overlap. If there is
  no clear position, the object is retained in the result as unplaced rather
  than being drawn over another item.
- **Furniture symbols:** the renderer draws class-specific top-down symbols
  rather than using the same plain rectangle for every detected object. Sofa
  symbols show one, two, or three seats according to the estimated footprint
  width. This is a rendering heuristic, not verified seat-count recognition.
- **Detection and false-positive handling:** the Torchvision COCO detector
  considers confidence-qualified candidates and retains up to 100 after
  duplicate suppression. The classical edge extractor discards a connected
  region that spans opposing image borders, since it is likely to represent
  photo-wide background or floor rather than one piece of furniture.
- **Regression coverage:** backend tests cover geometry correction, placement,
  symbol rendering, candidate handling, and the photo-wide-region filter.
  Run them from `backend` with `python -m pytest app\tests -q`.

These updates improve the schematic but do not make it an exact floor plan.
Positions, footprints, and room proportions remain estimates from one image.
The COCO model can detect only classes it was trained on; it may miss furniture
or misclassify it, and raising the candidate limit does not add new classes.
See [Limitations](docs/limitations.md) for details.

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
