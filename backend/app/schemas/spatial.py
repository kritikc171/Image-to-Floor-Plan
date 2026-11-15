"""Spatial data contract: versioned, validated structures shared by
pipeline, reasoning, renderer, and API.

Coordinate conventions
----------------------
- ``ImagePixel``: pixel coordinates, origin top-left, x right, y down.
- ``NormalizedImage``: x/y in [0, 1], origin top-left of the image.
- ``GroundPoint``: inverse-perspective (ground-plane) projection units.
  X right, Y into the scene (away from camera). Arbitrary units, not metric.
- ``PlanPoint``: normalized floor-plan coordinates in [0, 1], origin
  top-left of the plan, x right, y down.
- ``CanvasPixel``: renderer output pixels, origin top-left.
- Depth is a unitless relative estimate (larger = farther), never metric.

Image-space coordinates are never passed to the architectural renderer;
every crossing of coordinate spaces is an explicit transformation.

Evidence statuses
-----------------
OBSERVED  — directly supported by the photograph (e.g. a detector hit).
INFERRED  — estimated from visible evidence under stated assumptions.
GENERATED — derived from an AI-generated candidate view (not used here).
UNCERTAIN — not enough evidence; position/dimensions unknown.

Precedence: OBSERVED > INFERRED > GENERATED > UNCERTAIN.
Detection and spatial placement are different concepts: an object whose
class is OBSERVED may still have a floor position that is INFERRED or
UNCERTAIN. Physical dimensions are UNCERTAIN unless genuinely measured.
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
    image_bbox: Optional[BoundingBox] = None  # normalized image coordinates (detections)
    position: Optional[PlanPoint] = None  # normalized plan centroid of the footprint
    orientation: Optional[float] = None  # radians, 0 = facing "down-plan"
    dimensions: Optional[dict] = None  # only when genuinely known (rare)
    dimensions_confidence: Optional[float] = Field(
        default=None, ge=0, le=1
    )  # confidence in ``dimensions``
    footprint: Optional[dict] = None  # approximate plan-space {"width","depth","approximate":true}
    footprint_polygon: Optional[List[PlanPoint]] = None  # top-down footprint corners, plan space
    footprint_confidence: Optional[float] = Field(
        default=None, ge=0, le=1
    )  # confidence in the footprint estimate
    ground_contact: Optional[PlanPoint] = None  # projected bbox bottom-centre, if available
    ground_confidence: Optional[float] = Field(
        default=None, ge=0, le=1
    )  # confidence in the projected contact point
    confidence: Optional[float] = Field(default=None, ge=0, le=1)  # detection strength
    placement_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    evidence_status: EvidenceStatus
    evidence_source: List[str] = Field(default_factory=list)
    uncertainty_reason: Optional[str] = None
    # Optional layout-validation notes (never fabricated measurements).
    layout_issues: Optional[List[str]] = Field(default=None)


class WallSegment(BaseModel):
    """One wall of the estimated room perimeter.

    ``start`` and ``end`` are plan-space points; the segment connects them.
    ``thickness`` is normalized plan width of the wall.
    """

    id: str = Field(min_length=1, max_length=64)
    start: PlanPoint
    end: PlanPoint
    thickness: float = Field(default=0.04, ge=0, le=0.2)
    wall_type: str = Field(default="uncertain")  # "interior" | "exterior" | "uncertain"
    evidence_status: EvidenceStatus
    uncertainty_reason: Optional[str] = None


class GeometryDiagnostics(BaseModel):
    """Source, correction, and final coordinates used to draw the room."""

    geometry_source: str = "unknown"
    source_ground_corners: List[tuple[float, float]] = Field(default_factory=list)
    homography_applied: bool = False
    correction_method: str
    fallback_reason: Optional[str] = None
    corrected_room_corners: List[PlanPoint] = Field(default_factory=list)
    renderer_perimeter: List[PlanPoint] = Field(default_factory=list)
    renderer_wall_segments: List[WallSegment] = Field(default_factory=list)


class Opening(BaseModel):
    """Door or window candidate located on a wall segment."""

    id: str = Field(min_length=1, max_length=64)
    kind: str  # "door" | "window" | "opening"
    side: Optional[str] = None  # which wall side: "north" | "south" | "east" | "west" | null
    position: Optional[PlanPoint] = None  # plan point on the wall centreline
    width: Optional[float] = Field(default=None, ge=0, le=1)  # opening width along the wall
    swing: Optional[str] = Field(
        default=None
    )  # "inward" | "outward" | "left" | "right" | null
    swing_degrees: Optional[float] = Field(default=None, ge=0, le=180)
    wall_id: Optional[str] = None  # references WallSegment.id
    evidence_status: EvidenceStatus
    uncertainty_reason: Optional[str] = None


class RoomGeometry(BaseModel):
    """Coherent room perimeter represented as connected wall segments."""

    perimeter: Optional[List[PlanPoint]] = None  # closed ring, plan space
    corners: Optional[List[PlanPoint]] = None  # room corners
    walls: List[WallSegment] = Field(default_factory=list)
    doors: List[Opening] = Field(default_factory=list)
    windows: List[Opening] = Field(default_factory=list)
    evidence_status: EvidenceStatus
    uncertainty_reason: Optional[str] = None
    rectangular_assumption: bool = Field(
        default=False, description="True when corners were snapped to a rectangle"
    )


class RoomBoundary(BaseModel):
    """Backward-compatible room outline (the projected visible floor region)."""

    polygon: Optional[List[PlanPoint]] = None
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
    detection_model: Optional[str] = None  # object detector used, if any
    generated_views_used: int = Field(ge=0, le=4)
    model_name: Optional[str] = None


class SpatialHypothesis(BaseModel):
    """Validated structured spatial hypothesis (pipeline output)."""

    schema_version: str = SCHEMA_VERSION
    room_boundary: RoomBoundary
    room_geometry: Optional[RoomGeometry] = None
    geometry_diagnostics: Optional[GeometryDiagnostics] = None
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
