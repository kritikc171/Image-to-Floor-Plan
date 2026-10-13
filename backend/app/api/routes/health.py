"""Health endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.api import HealthResponse

router = APIRouter(tags=["health"])

APP_VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Readiness probe. Exposes no configuration or secrets."""
    get_settings()  # ensure settings load cleanly
    return HealthResponse(status="ok", version=APP_VERSION)
