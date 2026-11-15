"""M3 tests: depth, normals, features, hypothesis, renderer, pipeline."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.services.depth import (
    ClassicalDepthCueEstimator,
    DepthResult,
    _build_depth,
    _estimate_horizon,
)
from app.services.features import (
    ClassicalFeatureExtractor,
    FeatureExtractionResult,
    VisualEvidence,
    PlanTransform,
    normalize_floor_plan_geometry,
)
from app.services.floorplane import FloorPlaneEstimator
from app.services.hypothesis import build_hypothesis
from app.services.normals import (
    DepthNormalEstimator,
    NormalsResult,
    floor_pixel_ratio,
)
from app.services.pipeline import run_pipeline
from app.services.renderer import render_floorplan
from app.services.image_validation import ValidatedImage
from app.tests.conftest import make_png


def test_horizon_detection_finds_strong_edge_row():
    # Synthetic image: dark top (wall), bright bottom (floor)
    # with a sharp boundary at row 100.
    arr = np.zeros((200, 200), dtype=np.uint8)
    arr[:100, :] = 60
    arr[100:, :] = 200
    gray = arr.astype(np.float32) / 255.0
    horizon = _estimate_horizon(gray, 200)
    assert 90 <= horizon <= 110


def test_depth_map_is_relative_and_bounded():
    depth = _build_depth(200, 200, 100.0)
    assert depth.shape == (200, 200)
    assert depth.min() >= 0.0
    assert depth.max() <= 1.0
    # Floor pixels near the bottom are nearer (smaller depth)
    # than floor pixels near the horizon.
    assert depth[190, 100] < depth[105, 100]


def test_depth_estimator_on_image():
    img = Image.new("RGB", (320, 240), (100, 110, 120))
    result = ClassicalDepthCueEstimator().estimate(img)
    assert result.depth.shape == (240, 320)
    assert 0 < result.horizon_y < 240
    assert result.width == 320 and result.height == 240


def test_normals_shape_and_validity():
    from app.services.depth import DepthResult

    depth = np.random.default_rng(0).random((40, 40)).astype(np.float32)
    result = DepthResult(
        depth=depth, horizon_y=20.0, width=40, height=40
    )
    normals = DepthNormalEstimator().estimate(result)
    assert normals.normals.shape == (40, 40, 3)
    assert normals.valid.shape == (40, 40)
    # Borders are invalid
    assert not normals.valid[0, 0]
    # Valid normals are unit length
    valid = normals.normals[normals.valid]
    if valid.size:
        lengths = np.linalg.norm(valid, axis=-1)
        assert np.allclose(lengths, 1.0, atol=1e-3)


def test_feature_extraction_finds_synthetic_object():
    # Image with a dark rectangle on a light background below
    # the "horizon" (strong edges -> component).
    img = Image.new("RGB", (320, 240), (220, 220, 220))
    for y in range(140, 200):
        for x in range(80, 180):
            img.putpixel((x, y), (30, 30, 30))

    from app.services.depth import DepthResult

    depth_result = DepthResult(
        depth=np.zeros((240, 320), dtype=np.float32),
        horizon_y=120.0,
        width=320,
        height=240,
    )
    from app.services.normals import NormalsResult

    normals_result = NormalsResult(
        normals=np.zeros((240, 320, 3), dtype=np.float32),
        valid=np.zeros((240, 320), dtype=bool),
        fov_deg=60.0,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
    )

    features = ClassicalFeatureExtractor().extract(
        img, depth_result, normals_result
    )
    assert features.horizon_y == 120.0
    assert len(features.objects) >= 1
    obj = features.objects[0]
    assert obj.floor_level is True
    assert 0 <= obj.centroid[0] <= 1
    assert 0 <= obj.centroid[1] <= 1


def test_feature_extraction_ignores_regions_spanning_opposite_image_borders():
    image = Image.new("RGB", (320, 240), (180, 180, 180))
    draw = ImageDraw.Draw(image)
    for y in range(0, 240, 24):
        for x in range(0, 320, 24):
            if (x // 24 + y // 24) % 2:
                draw.rectangle((x, y, x + 23, y + 23), fill=(30, 30, 30))

    depth = DepthResult(
        depth=np.zeros((240, 320), dtype=np.float32),
        horizon_y=120.0,
        width=320,
        height=240,
    )
    normals = NormalsResult(
        normals=np.zeros((240, 320, 3), dtype=np.float32),
        valid=np.zeros((240, 320), dtype=bool),
        fov_deg=60.0,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
    )

    features = ClassicalFeatureExtractor().extract(image, depth, normals)

    assert features.objects == []


def test_hypothesis_marks_floor_objects_inferred():
    from app.services.features import (
        PlanTransform,
        VisualEvidence,
        FeatureExtractionResult,
    )

    transform = PlanTransform(sx=0.15, sy=10.0, ox=0.5, oy=0.4)
    result = FeatureExtractionResult(
        image_width=320,
        image_height=240,
        horizon_y=120.0,
        objects=[
            VisualEvidence(
                id="object-001",
                kind="object",
                label="compact object",
                bbox=(0.2, 0.6, 0.4, 0.8),
                centroid=(0.3, 0.7),
                color=(30, 30, 30),
                area_fraction=0.05,
                strength=0.4,
                floor_level=True,
            ),
            VisualEvidence(
                id="object-002",
                kind="object",
                label="vertical structure",
                bbox=(0.5, 0.1, 0.6, 0.4),
                centroid=(0.55, 0.25),
                color=(60, 60, 60),
                area_fraction=0.04,
                strength=0.3,
                floor_level=False,
            ),
        ],
        openings=[],
        floor_polygon=[(0.05, 0.1), (0.95, 0.1), (0.95, 0.9), (0.05, 0.9)],
        plan_transform=transform,
    )
    hypothesis = build_hypothesis(result)
    assert hypothesis.objects[0].evidence_status.value == "INFERRED"
    assert hypothesis.objects[0].position is not None
    assert hypothesis.objects[1].evidence_status.value == "UNCERTAIN"
    assert hypothesis.objects[1].position is None
    assert hypothesis.room_boundary.evidence_status.value == "INFERRED"
    assert len(hypothesis.room_boundary.polygon) == 4
    assert hypothesis.provenance.reasoning_provider == (
        "deterministic-fallback"
    )


def test_renderer_produces_valid_png():
    from app.services.features import (
        PlanTransform,
        VisualEvidence,
        FeatureExtractionResult,
    )

    transform = PlanTransform(sx=0.5, sy=0.5, ox=0.25, oy=0.25)
    result = FeatureExtractionResult(
        image_width=320,
        image_height=240,
        horizon_y=120.0,
        objects=[
            VisualEvidence(
                id="object-001",
                kind="object",
                label="compact object",
                bbox=(0.2, 0.6, 0.4, 0.8),
                centroid=(0.3, 0.7),
                color=(30, 30, 30),
                area_fraction=0.05,
                strength=0.4,
                floor_level=True,
            )
        ],
        openings=[],
        floor_polygon=[(0.05, 0.1), (0.95, 0.1), (0.95, 0.9), (0.05, 0.9)],
        plan_transform=transform,
    )
    hypothesis = build_hypothesis(result)
    png = render_floorplan(hypothesis, "job_test")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(io.BytesIO(png))
    assert img.size == (1400, 1000)
    # Deterministic: same hypothesis -> same bytes
    assert render_floorplan(hypothesis, "job_test") == png


def test_pipeline_end_to_end():
    image = ValidatedImage(
        image=Image.open(io.BytesIO(make_png(320, 240))),
        format="PNG",
        width=320,
        height=240,
    )
    output = run_pipeline("job_pipeline_test", image)
    assert output.result.job_id == "job_pipeline_test"
    assert output.result.schema_version == "1.0"
    assert output.result.hypothesis is not None
    assert output.floorplan_png[:8] == b"\x89PNG\r\n\x1a\n"


def test_floor_plane_projects_below_horizon_to_valid_plan():
    """Image-to-floor transformation must map below-horizon image points
    to valid normalized plan coordinates (the bug was a resolution mismatch
    between where the PlanTransform was fit and where project_ground
    projected)."""
    h, w = 240, 320
    horizon = 150
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[:horizon, :] = 60   # wall above horizon
    arr[horizon:, :] = 200  # floor below horizon
    img = Image.fromarray(np.stack([arr, arr, arr], axis=-1), "RGB")

    depth = ClassicalDepthCueEstimator().estimate(img)
    normals = DepthNormalEstimator().estimate(depth)
    floorplane = FloorPlaneEstimator().estimate(
        img, depth, normals, depth.horizon_y
    )
    assert floorplane.transform is not None
    assert floorplane.floor_polygon is not None
    assert len(floorplane.floor_polygon) >= 3
    assert floorplane.transform.homography is not None
    assert len(floorplane.floor_polygon) == 4
    corners = floorplane.floor_polygon
    assert corners[0][1] == pytest.approx(corners[1][1])
    assert corners[1][0] == pytest.approx(corners[2][0])
    assert corners[2][1] == pytest.approx(corners[3][1])
    assert corners[3][0] == pytest.approx(corners[0][0])

    feats = FeatureExtractionResult(
        image_width=w,
        image_height=h,
        horizon_y=depth.horizon_y,
        objects=[],
        openings=[],
        floor_polygon=floorplane.floor_polygon,
        plan_transform=floorplane.transform,
    )
    # A point on the floor (near bottom, below horizon) projects in-plan.
    out = feats.project_ground(0.5, 0.9)
    assert out is not None
    px, py = out
    assert 0.0 <= px <= 1.0
    assert 0.0 <= py <= 1.0
    # Ground width of a floor-level span is finite and positive.
    gw = feats.ground_width((0.2, 0.9, 0.8, 0.9))
    assert gw is not None and gw > 0
    # A point on/above the horizon has no finite floor projection.
    assert feats.project_ground(0.5, horizon / h) is None


def test_projective_floor_transform_keeps_left_right_object_order():
    h, w = 120, 160
    horizon = 55
    mask = np.zeros((h, w), dtype=bool)
    for y in range(60, h - 1):
        half_width = 20 + (y - 60) * 0.45
        center = (w - 1) / 2
        mask[y, int(center - half_width):int(center + half_width) + 1] = True

    from app.services.floorplane import _floor_polygon_and_transform

    polygon, transform, _ = _floor_polygon_and_transform(
        w, h, horizon, mask, w, h, horizon
    )
    assert polygon is not None
    assert transform is not None
    assert len(polygon) == 4
    assert polygon[0][1] == pytest.approx(polygon[1][1])
    assert polygon[1][0] == pytest.approx(polygon[2][0])
    assert polygon[2][1] == pytest.approx(polygon[3][1])
    assert polygon[3][0] == pytest.approx(polygon[0][0])

    features = FeatureExtractionResult(
        image_width=w,
        image_height=h,
        horizon_y=float(horizon),
        objects=[],
        openings=[],
        floor_polygon=polygon,
        plan_transform=transform,
    )
    left = features.project_ground(0.35, 0.82)
    right = features.project_ground(0.65, 0.82)
    assert left is not None and right is not None
    assert left[0] < right[0]


def test_rejected_floor_quad_uses_rectangular_fallback_through_rendering(monkeypatch):
    from app.services.floorplane import _floor_polygon_and_transform

    h, w, horizon = 120, 160, 55
    mask = np.zeros((h, w), dtype=bool)
    for y in range(60, h - 1):
        half_width = 20 + (y - 60) * 0.45
        center = (w - 1) / 2
        mask[y, int(center - half_width):int(center + half_width) + 1] = True

    monkeypatch.setattr(
        "app.services.floorplane.rectify_ground_quad",
        lambda _corners: None,
    )
    polygon, transform, info = _floor_polygon_and_transform(
        w, h, horizon, mask, w, h, horizon
    )
    assert polygon is not None and transform is not None
    assert info["correction_method"] == "axis-aligned rectangular approximation"
    assert info["fallback_reason"]
    assert [point[0] for point in polygon] == pytest.approx(
        [polygon[0][0], polygon[1][0], polygon[1][0], polygon[0][0]]
    )
    assert [point[1] for point in polygon] == pytest.approx(
        [polygon[0][1], polygon[0][1], polygon[2][1], polygon[2][1]]
    )
    aspect = (polygon[1][0] - polygon[0][0]) / (
        polygon[2][1] - polygon[1][1]
    )
    assert 0.65 <= aspect <= 1.6

    evidence = VisualEvidence(
        id="fallback-object",
        kind="object",
        label="wide surface",
        bbox=(0.25, 0.6, 0.48, 0.8),
        centroid=(0.365, 0.7),
        color=(80, 90, 100),
        area_fraction=0.05,
        strength=0.8,
        floor_level=True,
    )
    features = FeatureExtractionResult(
        image_width=w,
        image_height=h,
        horizon_y=float(horizon),
        objects=[evidence],
        openings=[],
        floor_polygon=polygon,
        plan_transform=transform,
        source_ground_corners=info["source_ground_corners"],
        geometry_correction_method=info["correction_method"],
        geometry_fallback_reason=info["fallback_reason"],
    )
    hypothesis = build_hypothesis(features)
    geometry = hypothesis.room_geometry
    assert geometry is not None
    diagnostics = hypothesis.geometry_diagnostics
    assert diagnostics is not None
    assert diagnostics.homography_applied is False
    assert diagnostics.fallback_reason == info["fallback_reason"]
    assert diagnostics.corrected_room_corners == geometry.corners
    assert diagnostics.renderer_perimeter == geometry.perimeter
    assert len(diagnostics.renderer_wall_segments) == 4
    assert hypothesis.objects[0].position is not None
    assert hypothesis.objects[0].label == "possible table"

    output = Image.open(io.BytesIO(render_floorplan(hypothesis, "fallback")))
    assert output.size == (1400, 1000)
    assert output.format == "PNG"


def test_classical_floor_geometry_fallback_is_rectangular(monkeypatch):
    from app.services.features import _floor_geometry

    monkeypatch.setattr(
        "app.services.features.rectify_ground_quad",
        lambda _corners: None,
    )
    polygon, transform, info = _floor_geometry(160, 120, 55.0, [])
    assert polygon is not None and transform is not None
    assert info["correction_method"] == "axis-aligned rectangular approximation"
    assert info["fallback_reason"]
    assert [point[0] for point in polygon] == pytest.approx(
        [polygon[0][0], polygon[1][0], polygon[1][0], polygon[0][0]]
    )
    assert [point[1] for point in polygon] == pytest.approx(
        [polygon[0][1], polygon[0][1], polygon[2][1], polygon[2][1]]
    )
    aspect = (polygon[1][0] - polygon[0][0]) / (
        polygon[2][1] - polygon[1][1]
    )
    assert 0.65 <= aspect <= 1.6


def test_pipeline_normalizes_trapezoid_before_hypothesis_and_rendering():
    features = FeatureExtractionResult(
        image_width=100,
        image_height=100,
        horizon_y=20.0,
        objects=[],
        openings=[],
        floor_polygon=[
            (0.20, 0.20),
            (0.80, 0.30),
            (0.75, 0.90),
            (0.25, 0.80),
        ],
        plan_transform=PlanTransform(
            sx=0.15,
            sy=30.0,
            ox=0.5,
            oy=0.05,
        ),
    )
    left_before = features.project_ground(0.3, 0.8)
    right_before = features.project_ground(0.7, 0.8)
    assert left_before is not None and right_before is not None
    assert left_before[0] < right_before[0]

    normalized = normalize_floor_plan_geometry(features)
    left_after = normalized.project_ground(0.3, 0.8)
    right_after = normalized.project_ground(0.7, 0.8)
    assert left_after is not None and right_after is not None
    assert left_after[0] < right_after[0]

    hypothesis = build_hypothesis(normalized)
    geometry = hypothesis.room_geometry
    assert geometry is not None and geometry.rectangular_assumption
    assert all(
        abs(a.x - b.x) < 1e-4 or abs(a.y - b.y) < 1e-4
        for a, b in zip(geometry.corners, geometry.corners[1:] + geometry.corners[:1])
    )
    diagnostics = hypothesis.geometry_diagnostics
    assert diagnostics is not None
    assert diagnostics.correction_method == "projective homography"
    assert diagnostics.renderer_perimeter == geometry.perimeter

    png = render_floorplan(hypothesis, "rectangularized-trapezoid")
    assert Image.open(io.BytesIO(png)).size == (1400, 1000)
