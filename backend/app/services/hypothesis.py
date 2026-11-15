"""Deterministic spatial-hypothesis construction (fallback reasoning).

Converts extracted visual evidence into a validated
:class:`SpatialHypothesis` without any AI provider.

Rules
-----
- Pretrained detector outputs are genuine observations: the object class is
  ``OBSERVED`` with the model's detection confidence and normalized image
  bbox. Detection and spatial placement are distinct: a detected object
  whose floor position cannot be estimated keeps ``position = None`` and
  is reported as uncertain, never fabricated.
- Floor-level objects receive an approximate plan position projected from
  the ground-contact point (bbox bottom-centre) through the estimated
  floor plane. Placement is INFERRED (an estimate, not a measurement).
- Floor-plane position, footprint, orientation and dimensions are all
  single-view estimates: heuristic confidences are reported, never
  presented as calibrated measurements.
- The room boundary is the projectively rectified visible floor region or
  an explicitly labeled rectangular approximation (INFERRED); hidden walls
  are absent, not fabricated.
- Openings are UNCERTAIN border-adjacent edge candidates; they are not
  confirmed doors or windows.
"""

from __future__ import annotations

import math
from typing import Optional

from app.schemas.spatial import (
    EvidenceStatus,
    GeometryDiagnostics,
    Opening,
    PlanPoint,
    Provenance,
    RoomBoundary,
    RoomGeometry,
    SpatialHypothesis,
    SpatialObject,
    UncertaintySummary,
)
from app.services.detection import DETECTION_SOURCE
from app.services.footprint import footprint_for, prior_for
from app.services.floorplane import FloorPlaneResult
from app.services import layout
from app.services.room_geometry import build_room_geometry
from app.services.features import FeatureExtractionResult, FLOOR_LEVEL_MARGIN, VisualEvidence

# Map classical geometric labels to the closest supported furniture class
# when no detector label is available. Approximations, not claims.
_CLASSICAL_TO_CLASS = {
    "wide surface": "table",
    "tall object": "cabinet",
    "compact object": "chair",
    "object": "furniture",
}

_FURNITURE_CLASSES = (
    "bed",
    "sofa",
    "couch",
    "chair",
    "dining table",
    "table",
    "desk",
    "wardrobe",
    "cabinet",
    "counter",
    "tv",
    "furniture",
)


def _furniture_class(label: str) -> str:
    text = (label or "").lower()
    for key in _FURNITURE_CLASSES:
        if key in text:
            return key
    for key, mapped in _CLASSICAL_TO_CLASS.items():
        if key in text:
            return mapped
    return "furniture"


def _ground_confidence(ev: VisualEvidence, horizon_norm: float, height: int) -> float:
    """Heuristic confidence in the projected ground-contact point."""
    if not ev.floor_level:
        return 0.0
    margin = ev.bbox[3] - (horizon_norm + FLOOR_LEVEL_MARGIN)
    depth_score = min(max(margin * 2.0, 0.0), 1.0)
    size_score = min(max(ev.area_fraction * 20.0, 0.0), 1.0)
    return round(0.45 + 0.35 * depth_score + 0.20 * size_score * ev.strength, 3)


def _footprint_estimate(
    class_name: str,
    ev: VisualEvidence,
    features: FeatureExtractionResult,
) -> tuple[dict, float]:
    """Approximate footprint (dict) plus heuristic footprint confidence."""
    pw, pd = prior_for(class_name)
    gw = features.ground_width(ev.bbox)
    basis = "approximate class prior; not a measured dimension"
    w = pw
    if gw is not None and gw > 0:
        scale = min(max(gw / pw, 0.5), 2.0)
        w = round(pw * scale, 4)
        basis = "class prior scaled by projected bbox base width"
    conf = round(0.35 + 0.35 * (1.0 if gw is not None else 0.0) + 0.15 * ev.strength, 3)
    return (
        {
            "width": round(w, 4),
            "depth": round(pd, 4),
            "approximate": True,
            "basis": basis,
        },
        conf,
    )


def _orientation_for(
    class_name: str, contact: PlanPoint, others: list[SpatialObject]
) -> Optional[float]:
    """Class-appropriate orientation in radians (0 = facing down-plan).

    Returns None when no spatial evidence supports a specific orientation;
    the caller then assumes down-plan and states it as an assumption.
    """
    if class_name == "chair":
        best, best_dist = None, float("inf")
        for o in others:
            if o.position is None or "table" not in (o.label or "").lower():
                continue
            dist = (
                (o.position.x - contact.x) ** 2 + (o.position.y - contact.y) ** 2
            ) ** 0.5
            if dist < best_dist:
                best, best_dist = o, dist
        if best is not None:
            return round(math.atan2(best.position.y - contact.y, best.position.x - contact.x), 4)
    return None


def build_hypothesis(
    features: FeatureExtractionResult,
    detection_model: Optional[str] = None,
    detection_available: bool = True,
    detection_unavailable_reason: Optional[str] = None,
) -> SpatialHypothesis:
    """Build the deterministic (fallback) spatial hypothesis."""

    floorplane_ok = features.plan_transform is not None and features.floor_polygon
    boundary_poly: list[tuple[float, float]] | None = None
    if features.floor_polygon:
        boundary_poly = [(p[0], p[1]) for p in features.floor_polygon]

    horizon_norm = features.horizon_y / max(features.image_height, 1)

    objects: list[SpatialObject] = []
    placed_rects: list[tuple[float, float, float, float]] = []

    for ev in features.objects:
        class_name = _furniture_class(ev.label)
        ground: Optional[tuple[float, float]] = None
        position: Optional[PlanPoint] = None
        placement_confidence: Optional[float] = None
        ground_confidence: Optional[float] = None
        layout_issues: list[str] = []

        if ev.floor_level and floorplane_ok:
            ground = features.to_plan_ground(ev.bbox)
            if ground is not None:
                position = PlanPoint(x=round(ground[0], 4), y=round(ground[1], 4))
                ground_confidence = _ground_confidence(ev, horizon_norm, features.image_height)
                # The floor position is a deterministic projection of the
                # observed contact point, so placement confidence reflects
                # detection strength; ground_confidence reports projection
                # reliability separately.
                base = ev.strength if ev.source == DETECTION_SOURCE else 0.6
                placement_confidence = round(min(0.95, base), 3)
        elif ev.floor_level and not floorplane_ok:
            layout_issues.append("floor plane unavailable; shown as unplaced")
        else:
            layout_issues.append("above floor level; no floor position")

        footprint, fp_confidence = _footprint_estimate(class_name, ev, features)
        if position is None:
            layout_issues.append("no floor position; shown as unplaced")

        is_detection = ev.source == DETECTION_SOURCE
        if is_detection:
            status = EvidenceStatus.OBSERVED
            model_label = detection_model or "object detector"
            reason = (
                f"Class {class_name!r} detected in the photograph by the "
                f"pretrained object detector ({model_label}). Floor position "
                "is projected from a single view and is approximate, not "
                "measured."
            )
            confidence = round(ev.strength, 3)
            if position is None:
                reason += (
                    " The object projects above the estimated floor line, "
                    "so no floor position is available."
                )
        else:
            status = EvidenceStatus.UNCERTAIN
            reason = (
                "Object is above the floor level; a single photo cannot "
                "determine its floor position."
            )
            confidence = round(ev.strength, 3)
            if ev.floor_level:
                status = EvidenceStatus.INFERRED
                reason = (
                    "Shape-based category candidate only; the object is not "
                    "confirmed as this furniture type. Position is projected "
                    "from the visible floor under a ground-plane assumption."
                )

        orientation = None
        if position is not None:
            orientation = _orientation_for(class_name, position, objects)
            if orientation is None:
                orientation = 0.0  # stated down-plan assumption

        if position is not None and boundary_poly is not None:
            angle = orientation or 0.0
            width = footprint["width"]
            depth = footprint["depth"]
            bounds_w, bounds_d = layout.oriented_dimensions(width, depth, angle)
            cx, cy, fit_w, fit_d, fit_issues = layout.fit_footprint(
                position.x, position.y, bounds_w, bounds_d, boundary_poly,
            )
            layout_issues.extend(fit_issues)
            scale = min(fit_w / bounds_w, fit_d / bounds_d)
            footprint["width"] = round(width * scale, 4)
            footprint["depth"] = round(depth * scale, 4)
            bounds_w, bounds_d = layout.oriented_dimensions(
                footprint["width"], footprint["depth"], angle
            )

            nx, ny, overlap_issue = layout.resolve_overlap(
                layout.rect(cx, cy, bounds_w, bounds_d),
                placed_rects,
                boundary_poly,
            )
            if overlap_issue:
                layout_issues.append(overlap_issue)
                position = None
                reason += (
                    " No non-overlapping in-room position was available, "
                    "so the object is listed as unplaced rather than drawn "
                    "on top of another object."
                )
            else:
                position = PlanPoint(x=round(nx, 4), y=round(ny, 4))
                placed_rects.append(layout.rect(nx, ny, bounds_w, bounds_d))

        footprint_polygon = [
            PlanPoint(x=round(x, 4), y=round(y, 4))
            for x, y in layout.oriented_corners(
                position.x,
                position.y,
                footprint["width"],
                footprint["depth"],
                orientation or 0.0,
            )
        ] if position is not None else None

        objects.append(
            SpatialObject(
                id=ev.id,
                type="furniture",
                label=(
                    ev.label
                    if is_detection
                    else (
                        f"possible {class_name}"
                        if class_name != "furniture"
                        else "unclassified furniture"
                    )
                ),
                image_bbox=(
                    {
                        "x_min": ev.bbox[0],
                        "y_min": ev.bbox[1],
                        "x_max": ev.bbox[2],
                        "y_max": ev.bbox[3],
                    }
                    if is_detection
                    else None
                ),
                position=position,
                orientation=orientation,
                dimensions=None,
                dimensions_confidence=None,
                footprint=footprint,
                footprint_polygon=footprint_polygon,
                footprint_confidence=round(fp_confidence, 3),
                ground_contact=(
                    PlanPoint(x=round(ground[0], 4), y=round(ground[1], 4))
                    if ground is not None else None
                ),
                ground_confidence=ground_confidence,
                confidence=confidence,
                placement_confidence=placement_confidence,
                evidence_status=status,
                evidence_source=[ev.source],
                uncertainty_reason=reason,
                layout_issues=layout_issues or None,
            )
        )

    # ---- Openings --------------------------------------------------------
    openings_out: list[Opening] = []
    for ev in features.openings:
        position = None
        projected = features.to_plan(*ev.centroid)
        if projected is not None:
            position = PlanPoint(x=round(projected[0], 4), y=round(projected[1], 4))
        openings_out.append(
            Opening(
                id=ev.id,
                kind="opening",
                side=None,
                position=position,
                wall_id=None,
                width=None,
                swing=None,
                swing_degrees=None,
                evidence_status=EvidenceStatus.UNCERTAIN,
                uncertainty_reason=(
                    "Border-adjacent edge candidate; not confirmed as a "
                    "door or window."
                ),
            )
        )

    # ---- Room geometry ---------------------------------------------------
    room_geometry: Optional[RoomGeometry] = None
    if features.floor_polygon and features.plan_transform is not None:
        room_geometry = build_room_geometry(
            FloorPlaneResult(
                transform=features.plan_transform,
                floor_polygon=features.floor_polygon,
                wall_lines=features.wall_lines or [],
                vanishing_point=features.vanishing_point,
                camera_pitch_deg=features.camera_pitch_deg,
                quality=features.floorplane_quality or "poor",
                uncertainty_reason=(
                    "classical horizon-based floor plane"
                    if features.floorplane_quality != "good"
                    else None
                ),
                source_ground_corners=features.source_ground_corners or None,
                geometry_correction_method=features.geometry_correction_method,
                geometry_fallback_reason=features.geometry_fallback_reason,
                geometry_source=features.geometry_source,
            ),
            openings_out,
        )

    if room_geometry is not None and room_geometry.perimeter:
        polygon = room_geometry.perimeter[:-1] if (
            len(room_geometry.perimeter) > 1
            and room_geometry.perimeter[0] == room_geometry.perimeter[-1]
        ) else room_geometry.perimeter
        boundary = RoomBoundary(
            polygon=polygon or None,
            evidence_status=room_geometry.evidence_status,
            uncertainty_reason=room_geometry.uncertainty_reason,
        )
    else:
        boundary = RoomBoundary(
            polygon=None,
            evidence_status=EvidenceStatus.UNCERTAIN,
            uncertainty_reason=(
                "No reliable floor plane could be estimated; room boundary "
                "is unavailable."
            ),
        )

    # ---- Layout validation against walls (second pass) -----------------
    wall_conflict_objects = 0
    if room_geometry is not None and room_geometry.walls:
        for obj in objects:
            if obj.position is None or obj.footprint is None:
                continue
            fp = obj.footprint
            r = layout.rect(
                obj.position.x, obj.position.y, fp.get("width", 0.18) or 0.18,
                fp.get("depth", 0.18) or 0.18,
            )
            conflicts = layout.rect_vs_walls(r, room_geometry.walls)
            if conflicts:
                wall_conflict_objects += 1
                obj.layout_issues = (
                    list(obj.layout_issues or [])
                    + [f"footprint near wall(s): {', '.join(conflicts)}"]
                )

    object_count = len(objects)
    overall = "high"
    if object_count >= 4:
        overall = "medium"
    if not objects:
        overall = "high"

    assumptions = [
        "Single RGB photograph without camera intrinsics",
        "Camera optical axis assumed parallel to the floor",
        "Depth is a unitless relative estimate, not metric depth",
        "Object labels are generic geometric descriptions",
        "Heuristic scores are not calibrated probabilities",
    ]
    if room_geometry is not None and room_geometry.rectangular_assumption:
        assumptions.append(
            f"Room outline correction: {features.geometry_correction_method}; "
            "true room shape and dimensions are unknown."
        )
    else:
        assumptions.append(
            "Room boundary is the projected visible floor region, not a "
            "verified complete room outline."
        )
    if features.floor_polygon:
        assumptions.append(
            "Object orientation is assumed to face down-plan (0 rad); "
            "not recoverable from a single photo"
        )
    if not detection_available:
        assumptions.append(
            f"Object detection unavailable ({detection_unavailable_reason}); "
            "classical visual evidence only."
        )

    open_questions: list[str] = []
    if not features.floor_polygon:
        open_questions.append("Where are the room boundaries?")
    if not objects:
        open_questions.append("What furniture is visible in the room?")
    if not openings_out:
        open_questions.append("Are there doors or windows in view?")

    geometry_diagnostics = GeometryDiagnostics(
        geometry_source=features.geometry_source,
        source_ground_corners=features.source_ground_corners,
        homography_applied=(
            features.geometry_correction_method == "projective homography"
        ),
        correction_method=features.geometry_correction_method,
        fallback_reason=features.geometry_fallback_reason,
        corrected_room_corners=(
            room_geometry.corners if room_geometry is not None else []
        ),
        renderer_perimeter=(
            room_geometry.perimeter if room_geometry is not None else []
        ),
        renderer_wall_segments=(
            room_geometry.walls if room_geometry is not None else []
        ),
    )

    return SpatialHypothesis(
        room_boundary=boundary,
        room_geometry=room_geometry,
        geometry_diagnostics=geometry_diagnostics,
        objects=objects,
        openings=openings_out,
        uncertainty=UncertaintySummary(
            overall=overall,
            assumptions=assumptions,
            open_questions=open_questions,
        ),
        provenance=Provenance(
            reasoning_provider="deterministic-fallback",
            depth_method="classical-relative (horizon + ground-plane cue)",
            feature_method="classical-cv (gradient regions + floor segmentation)",
            detection_model=detection_model,
            generated_views_used=0,
        ),
    )
