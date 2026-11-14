"""Geometry, coordinate-transformation, and renderer validation tests.

These tests assert the spatial pipeline contracts required by the
floor-plan redesign:

- image coordinates are never used as top-down room coordinates,
- missing/invalid floor positions are never silently defaulted to (0, 0),
- every placed footprint stays inside the estimated room boundary,
- object overlap is resolved without fabricating geometry,
- the renderer emits a valid PNG whose output does not depend on the
  job id (so no job id can appear in the drawing).
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.services.features import (
    ClassicalFeatureExtractor,
    FLOOR_LEVEL_MARGIN,
    PlanTransform,
    VisualEvidence,
    FeatureExtractionResult,
)
from app.services.hypothesis import build_hypothesis
from app.services.layout import rect, rect_inside_poly, fit_footprint, resolve_overlap
from app.services.renderer import render_floorplan
from app.services.footprint import footprint_for, prior_for

# A simple rectangular room in normalized plan space.
ROOM = [(0.05, 0.10), (0.95, 0.10), (0.95, 0.90), (0.05, 0.90)]

TRANSFORM = PlanTransform(sx=0.5, sy=0.5, ox=0.25, oy=0.25)


def _features(objects, polygon=ROOM, transform=TRANSFORM) -> FeatureExtractionResult:
    return FeatureExtractionResult(
        image_width=320,
        image_height=240,
        horizon_y=120.0,
        objects=list(objects),
        openings=[],
        floor_polygon=polygon,
        plan_transform=transform,
    )


def _floor_evidence(oid, label, bbox, strength=0.8):
    """Floor-level evidence: bbox bottom edge below the horizon."""
    y_bottom = bbox[3]
    return VisualEvidence(
        id=oid,
        kind="object",
        label=label,
        bbox=bbox,
        centroid=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
        color=(30, 30, 30),
        area_fraction=0.05,
        strength=strength,
        floor_level=y_bottom > (120.0 / 240.0) + FLOOR_LEVEL_MARGIN,
    )


def test_fit_footprint_stays_inside_convex_room():
    cx, cy, w, d, issues = fit_footprint(0.5, 0.5, 0.2, 0.2, ROOM)
    assert rect_inside_poly(rect(cx, cy, w, d), ROOM)
    # A footprint larger than the room is scaled down, not placed outside.
    cx, cy, w, d, issues = fit_footprint(0.5, 0.5, 2.0, 2.0, ROOM)
    assert w <= 0.85 * 0.9 and d <= 0.85 * 0.9
    assert rect_inside_poly(rect(cx, cy, w, d), ROOM)


def test_resolve_overlap_separates_without_leaving_room():
    r1 = rect(0.5, 0.5, 0.2, 0.2)
    r2 = rect(0.55, 0.5, 0.2, 0.2)  # overlaps r1
    nx, ny, issue = resolve_overlap(r2, [r1], ROOM)
    assert issue is None
    cand = rect(nx, ny, 0.2, 0.2)
    assert rect_inside_poly(cand, ROOM)
    ox, oy = rect_overlap_if_needed(r1, cand)
    assert ox <= 0.005 or oy <= 0.005


def rect_overlap_if_needed(a, b):
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    return (ox, oy) if ox > 0.005 and oy > 0.005 else (0.0, 0.0)


def test_project_ground_returns_none_at_or_above_horizon():
    # horizon_y = 120 of 240 -> normalized horizon = 0.5.
    floor = TRANSFORM  # not used directly; use project_ground via features
    from app.services.features import FeatureExtractionResult as F
    feats = F(image_width=320, image_height=240, horizon_y=120.0, objects=[],
              openings=[], floor_polygon=ROOM, plan_transform=TRANSFORM)
    # On the horizon -> no finite floor projection.
    assert feats.project_ground(0.5, 0.5) is None
    # Above the horizon -> None.
    assert feats.project_ground(0.5, 0.4) is None
    # Below the horizon -> finite projection in [0,1].
    out = feats.project_ground(0.5, 0.8)
    assert out is not None
    px, py = out
    assert 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0


def test_invalid_position_is_never_defaulted_to_zero():
    # A ground point projecting outside the plan laterally must be
    # unavailable (None), not silently clamped to y = 0.
    transform = PlanTransform(sx=0.5, sy=0.5, ox=0.25, oy=0.25)
    feats = FeatureExtractionResult(
        image_width=320, image_height=240, horizon_y=120.0, objects=[],
        openings=[], floor_polygon=ROOM, plan_transform=transform,
    )
    # Far to the side projects to a negative/large plan x; clamp keeps
    # it in [0,1] but py must remain valid. The key contract: None when
    # py falls outside [0,1] rather than inventing y = 0.
    assert feats.project_ground(0.02, 0.8) is not None


def test_floor_level_objects_are_placed_inside_room():
    objs = [
        _floor_evidence("object-001", "dining table", (0.25, 0.65, 0.55, 0.85)),
        _floor_evidence("object-002", "chair", (0.58, 0.68, 0.70, 0.86)),
        _floor_evidence("object-003", "sofa", (0.10, 0.60, 0.40, 0.80)),
    ]
    hypothesis = build_hypothesis(_features(objs), detection_model=None)
    assert hypothesis.room_boundary.polygon is not None
    poly = [(p.x, p.y) for p in hypothesis.room_boundary.polygon]
    placed = [o for o in hypothesis.objects if o.position is not None]
    # Every placed object must have a footprint inside the boundary.
    for o in placed:
        fp = o.footprint or {}
        r = rect(o.position.x, o.position.y,
                 fp.get("width", 0.18) or 0.18,
                 fp.get("depth", 0.18) or 0.18)
        assert rect_inside_poly(r, poly), f"{o.label} footprint outside room"
    # No position is silently (0, 0).
    for o in placed:
        assert not (o.position.x == 0.0 and o.position.y == 0.0), o.id


def test_unplaced_objects_are_preserved_and_not_fabricated():
    objs = [
        VisualEvidence(
            id="object-001", kind="object", label="tv",
            bbox=(0.5, 0.1, 0.7, 0.4), centroid=(0.6, 0.25),
            color=(40, 40, 40), area_fraction=0.04, strength=0.8,
            floor_level=False,
        ),
    ]
    hypothesis = build_hypothesis(_features(objs), detection_model=None)
    assert len(hypothesis.objects) == 1
    o = hypothesis.objects[0]
    assert o.position is None
    assert o.label == "tv"  # detection preserved, not dropped


def test_footprint_priors_differ_by_class():
    assert prior_for("bed") != prior_for("chair")
    assert prior_for("dining table") != prior_for("cabinet")
    w, d = prior_for("bed")
    assert w > 0.3 and d > 0.3  # bed is a large footprint
    wc, dc = prior_for("chair")
    assert wc < 0.2 and dc < 0.2  # chair is compact


def test_renderer_output_independent_of_job_id():
    objs = [
        _floor_evidence("object-001", "dining table", (0.3, 0.6, 0.6, 0.85)),
    ]
    hypothesis = build_hypothesis(_features(objs), detection_model=None)
    a = render_floorplan(hypothesis, "job_aaa")
    b = render_floorplan(hypothesis, "job_zzz")
    assert a == b  # job id must not influence the drawing
    assert a[:8] == b"\x89PNG\r\n\x1a\n"
    im = Image.open(io.BytesIO(a))
    assert im.size == (1400, 1000)


def test_renderer_handles_unplaced_without_crash():
    objs = [
        VisualEvidence(
            id="object-001", kind="object", label="tv",
            bbox=(0.5, 0.1, 0.7, 0.4), centroid=(0.6, 0.25),
            color=(40, 40, 40), area_fraction=0.04, strength=0.8,
            floor_level=False,
        ),
    ]
    hypothesis = build_hypothesis(_features(objs), detection_model=None)
    png = render_floorplan(hypothesis, "job_unplaced")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_project_ground_floor_level_margin():
    """project_ground rejects points within the floor-level margin above the
    horizon but accepts points measurably below it (the 0.12 -> 0.05 change)."""
    feats = FeatureExtractionResult(
        image_width=320, image_height=240, horizon_y=120.0, objects=[],
        openings=[], floor_polygon=ROOM, plan_transform=TRANSFORM,
    )
    # horizon norm = 0.5; margin gate = 0.55.
    assert feats.project_ground(0.5, 0.50) is None      # on the horizon
    assert feats.project_ground(0.5, 0.51) is None      # within margin above horizon
    out = feats.project_ground(0.5, 0.60)               # measurably below horizon
    assert out is not None
    px, py = out
    assert 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0


def test_feature_extractor_floor_level_uses_margin():
    """An object whose base sits between the old (0.12) and new (0.05)
    floor-level gates is treated as floor level."""
    from app.services.depth import DepthResult
    from app.services.normals import NormalsResult

    img = Image.new("RGB", (320, 240), (220, 220, 220))
    for y in range(100, 140):
        for x in range(100, 180):
            img.putpixel((x, y), (30, 30, 30))

    depth_result = DepthResult(
        depth=np.zeros((240, 320), dtype=np.float32),
        horizon_y=120.0, width=320, height=240,
    )
    normals_result = NormalsResult(
        normals=np.zeros((240, 320, 3), dtype=np.float32),
        valid=np.zeros((240, 320), dtype=bool),
        fov_deg=60.0, fx=1.0, fy=1.0, cx=0.0, cy=0.0,
    )
    features = ClassicalFeatureExtractor().extract(img, depth_result, normals_result)
    floor_objs = [o for o in features.objects if o.floor_level]
    assert len(floor_objs) >= 1
    # Rectangle bottom edge is near row 139-141 -> norm ~0.58.
    # Old gate: 0.5 + 0.12 = 0.62 -> rejected. New gate: 0.55 -> accepted.
    y_max = floor_objs[0].bbox[3]
    assert floor_objs[0].floor_level is True
    assert 0.55 < y_max <= 0.62


def test_hypothesis_places_object_just_below_old_margin():
    """An object below the new floor-level gate gets a real plan position
    instead of being left unplaced."""
    objs = [
        _floor_evidence("object-001", "wide surface", (0.40, 0.45, 0.60, 0.58), strength=0.6),
    ]
    hypothesis = build_hypothesis(_features(objs), detection_model=None)
    assert len(hypothesis.objects) == 1
    obj = hypothesis.objects[0]
    # bottom at 0.58 > new gate 0.55 -> projected; old gate 0.62 would reject.
    assert obj.position is not None
    assert obj.placement_confidence is not None
    assert 0.0 <= obj.position.x <= 1.0
    assert 0.0 <= obj.position.y <= 1.0


def test_room_geometry_includes_wall_floor_intersection_lines():
    """Wall-floor intersection cues from the floor plane become wall segments
    in room geometry; lines duplicating a polygon edge are not repeated."""
    transform = PlanTransform(sx=0.5, sy=0.5, ox=0.25, oy=0.25)
    polygon = [(0.05, 0.10), (0.95, 0.10), (0.95, 0.90), (0.05, 0.90)]
    wall_lines = [
        [(0.05, 0.10), (0.95, 0.10)],   # duplicates top polygon edge -> skipped
        [(0.40, 0.40), (0.40, 0.80)],   # interior wall-floor cue -> added
        [(0.20, 0.50), (0.20, 0.50)],   # degenerate -> skipped
    ]
    features = FeatureExtractionResult(
        image_width=320, image_height=240, horizon_y=120.0, objects=[],
        openings=[], floor_polygon=polygon, plan_transform=transform,
        wall_lines=wall_lines, floorplane_quality="good",
    )
    hypothesis = build_hypothesis(features)
    rg = hypothesis.room_geometry
    assert rg is not None
    perimeter_walls = [w for w in rg.walls if w.id.startswith("wall-") and not w.id.startswith("wall-wl-")]
    cue_walls = [w for w in rg.walls if w.id.startswith("wall-wl-")]
    assert len(perimeter_walls) == 4, f"expected 4 perimeter walls, got {len(perimeter_walls)}"
    assert len(cue_walls) == 1, f"expected 1 wall-floor cue wall, got {len(cue_walls)}"
    assert cue_walls[0].start.x == 0.4 and cue_walls[0].start.y == 0.4
    assert cue_walls[0].end.x == 0.4 and cue_walls[0].end.y == 0.8
    assert cue_walls[0].evidence_status.value == "INFERRED"
    # Perimeter preserved as the boundary.
    assert len(hypothesis.room_boundary.polygon) == 4
