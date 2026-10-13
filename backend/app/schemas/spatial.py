"""Spatial data contract: versioned, validated structures shared by
pipeline, reasoning, renderer, and API.

Coordinate conventions
----------------------
- ``ImagePoint``: pixel coordinates, origin top-left, x right, y down.
- ``PlanPoint``: normalized floor-plan coordinates in [0, 1], origin
  top-left of the plan.
- Depth is a unitless relative estimate (larger = farther), never metric.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1.0"


class EvidenceStatus(str, Enum):
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    GENERATED = "GENERATED"
    UNCERTAIN = "UNCERTAIN"


# Precedence: OBSERVED > INFERRED > GENERATED > UNCERTAIN
EVIDENCE_PRECEDENCE: List[EvidenceStatus] = [
    EvidenceStatus.OBSERVED,
    EvidenceStatus.INFERRED,
    EvidenceStatus.GENERATED,
    EvidenceStatus.UNCERTAIN,
]


class ImagePoint(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)


class PlanPoint(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class BoundingBox(BaseModel):
    x_min: float = Field(ge=0)
    y_min: float = Field(ge=0)
    x_max: float = Field(ge=0)
    y_max: float = Field(ge=0)

    @field_validator("x_max")
    @classmethod
    def _x_order(cls, v: float, info) -> float:
        if v < info.data.get("x_min", 0):
            raise ValueError("x_max must be >= x_min")
        return v

    @field_validator("y_max")
    @classmethod
    def _y_order(cls, v: float, info) -> float:
        if v < info.data.get("y_min", 0):
            raise ValueError("y_max must be >= y_min")
        return v


class SpatialObject(BaseModel):
    """One detected or hypothesized object/feature."""

    id: str = Field(min_length=1, max_length=64)
    type: str = Field(min_length=1, max_length=64)
    label: Optional[str] = None
    position: Optional[PlanPoint] = None
    orientation: Optional[float] = None  # radians, 0 = facing "down-plan"
    dimensions: Optional[dict] = None  # only when genuinely known (rare)
    evidence_status: EvidenceStatus
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    evidence_source: List[str] = Field(default_factory=list)
    uncertainty_reason: Optional[str] = None


class RoomBoundary(BaseModel):
    polygon: Optional[List[PlanPoint]] = None
    evidence_status: EvidenceStatus
    uncertainty_reason: Optional[str] = None


class Opening(BaseModel):
    """Door or window candidate."""

    id: str = Field(min_length=1, max_length=64)
    kind: str  # "door" | "window" | "opening"
    side: Optional[str] = None
    position: Optional[PlanPoint] = None
    evidence_status: EvidenceStatus
    uncertainty_reason: Optional[str] = None


class UncertaintySummary(BaseModel):
    overall: str  # "low" | "medium" | "high"
    assumptions: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)


class Provenance(BaseModel):
    reasoning_provider: str  # "groq" | "deterministic-fallback"
    depth_method: str
    feature_method: str
    generated_views_used: int = Field(ge=0, le=4)
    model_name: Optional[str] = None


class SpatialHypothesis(BaseModel):
    """Validated structured spatial hypothesis (pipeline output)."""

    schema_version: str = SCHEMA_VERSION
    room_boundary: RoomBoundary
    objects: List[SpatialObject] = Field(default_factory=list)
    openings: List[Opening] = Field(default_factory=list)
    uncertainty: UncertaintySummary
    provenance: Provenance


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobError(BaseModel):
    code: str
    message: str


class GenerationResult(BaseModel):
    """API result payload returned with a completed job."""

    schema_version: str = SCHEMA_VERSION
    job_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    input: dict
    hypothesis: SpatialHypothesis
    floorplan_image_url: str


class GenerationJob(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[JobError] = None
    result: Optional[GenerationResult] = None
