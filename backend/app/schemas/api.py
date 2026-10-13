"""API-facing request/response models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.spatial import GenerationResult


class HealthResponse(BaseModel):
    status: str
    version: str


class GenerationCreateResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GenerationStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    error: Optional[dict] = None
    result: Optional[GenerationResult] = None


class ErrorResponse(BaseModel):
    error: dict
