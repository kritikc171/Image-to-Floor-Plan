# Floor Plan Generator

A web application that accepts **one RGB photograph of an indoor room** and produces an
**approximate 2D top-down floor plan** as a structured spatial hypothesis plus a
downloadable schematic PNG.

> **This is an approximation system, not an architectural measurement tool.**
> A single RGB image cannot recover hidden geometry or exact metric dimensions.
> The pipeline distinguishes **observed** evidence from **inferred** and **uncertain**
> elements, and never presents inferred geometry as verified.

## Quick start

### Prerequisites
- Python 3.13 (verified with 3.13.12; required by the pinned CV/AI stack)
- Node.js 24 and npm 12 (verified)
- Git

### Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# optional: copy .env.example to .env and set GROQ_API_KEY (not required)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Frontend

```powershell
cd frontend
npm install
npm run dev        # http://localhost:5173
```

### Tests

```powershell
cd backend; python -m pytest          # backend test suite
cd frontend; npm test                 # vitest run
```

## Repository layout

```text
├── frontend/        React + TypeScript + Vite application (presentation only)
├── backend/         FastAPI + LangGraph processing pipeline
│   └── app/
│       ├── api/         HTTP routes
│       ├── core/        configuration, logging, error handling
│       ├── schemas/     Pydantic spatial + API contracts
│       ├── services/    CV pipeline services (validation, depth, normals, features, renderer)
│       ├── tools/       external provider clients (Groq, optional novel-view)
│       ├── nodes/       LangGraph node functions
│       ├── graph.py     LangGraph workflow definition
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
