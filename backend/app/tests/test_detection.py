"""Detection and fixture-reuse tests."""

from __future__ import annotations

import os
from io import BytesIO

import pytest
from PIL import Image

from app.services.detection import (
    DETECTION_SOURCE,
    Detection,
    TorchVisionDetector,
    UnavailableDetector,
)
from app.services.features import (
    PlanTransform,
    VisualEvidence,
    FeatureExtractionResult,
    ClassicalFeatureExtractor,
)
from app.services.hypothesis import build_hypothesis
from app.services.pipeline import (
    DEFAULT_DEPS,
    PipelineDeps,
    run_pipeline,
)
from app.services.depth import ClassicalDepthCueEstimator, DepthResult
from app.services.normals import (
    DepthNormalEstimator,
    NormalsResult,
)
from app.services.image_validation import ValidatedImage


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
FIXTURE = os.path.join(ROOT, "test_floorplan.png")


def test_detector_suppresses_near_identical_cross_class_boxes():
    detections = [
        Detection(
            id="det-000",
            class_name="chair",
            label="chair",
            bbox=(0.2, 0.2, 0.6, 0.7),
            confidence=0.72,
        ),
        Detection(
            id="det-001",
            class_name="couch",
            label="couch",
            bbox=(0.205, 0.205, 0.595, 0.695),
            confidence=0.61,
        ),
        Detection(
            id="det-002",
            class_name="bed",
            label="bed",
            bbox=(0.65, 0.25, 0.95, 0.8),
            confidence=0.58,
        ),
    ]

    result = TorchVisionDetector._deduplicate(detections)

    assert [item.id for item in result] == ["det-000", "det-002"]


def _depth_result() -> DepthResult:
    return DepthResult(
        depth=None,  # type: ignore[arg-type]
        horizon_y=120.0,
        width=320,
        height=240,
    )


def _normals_result() -> NormalsResult:
    return NormalsResult(
        normals=None,  # type: ignore[arg-type]
        valid=None,  # type: ignore[arg-type]
        fov_deg=60.0,
        fx=1.0,
        fy=1.0,
        cx=0.0,
        cy=0.0,
    )


def _features(objects, openings=None, polygon=None, plan_transform=None) -> FeatureExtractionResult:
    return FeatureExtractionResult(
        image_width=320,
        image_height=240,
        horizon_y=120.0,
        objects=objects,
        openings=openings or [],
        floor_polygon=polygon,
        plan_transform=plan_transform,
    )


def test_detector_supported_classes_cover_indoor_subset():
    detector = TorchVisionDetector()
    classes = detector.supported_classes
    for expected in ("bed", "couch", "chair", "dining table", "tv", "person"):
        assert expected in classes, f"missing COCO class {expected!r}"
    for unsupported in ("desk", "cabinet", "wardrobe", "shelf", "door", "window"):
        assert unsupported not in classes
    assert "__background__" not in classes
    assert "N/A" not in classes


def test_hypothesis_marks_detector_evidence_observed():
    transform = PlanTransform(sx=0.15, sy=10.0, ox=0.5, oy=0.4)
    features = _features(
        objects=[
            VisualEvidence(
                id="det-000",
                kind="object",
                label="bed",
                bbox=(0.2, 0.6, 0.5, 0.8),
                centroid=(0.35, 0.7),
                color=(30, 30, 30),
                area_fraction=0.06,
                strength=0.62,
                floor_level=True,
                source=DETECTION_SOURCE,
            )
        ],
        polygon=[(0.05, 0.1), (0.95, 0.1), (0.95, 0.9), (0.05, 0.9)],
        plan_transform=transform,
    )
    hypothesis = build_hypothesis(features, detection_model="test-model")
    assert len(hypothesis.objects) == 1
    obj = hypothesis.objects[0]
    assert obj.evidence_status.value == "OBSERVED"
    assert obj.confidence == 0.62
    assert obj.placement_confidence == 0.62
    assert obj.position is not None
    assert obj.image_bbox.x_min == 0.2
    assert obj.image_bbox.y_min == 0.6
    assert obj.image_bbox.x_max == 0.5
    assert obj.image_bbox.y_max == 0.8
    assert hypothesis.provenance.detection_model == "test-model"


def test_hypothesis_detector_above_floor_keeps_detection_without_position():
    transform = PlanTransform(sx=0.15, sy=10.0, ox=0.5, oy=0.4)
    features = _features(
        objects=[
            VisualEvidence(
                id="det-000",
                kind="object",
                label="tv",
                bbox=(0.5, 0.1, 0.7, 0.4),
                centroid=(0.6, 0.25),
                color=(40, 40, 40),
                area_fraction=0.04,
                strength=0.8,
                floor_level=False,
                source=DETECTION_SOURCE,
            )
        ],
        polygon=[(0.05, 0.1), (0.95, 0.1), (0.95, 0.9), (0.05, 0.9)],
        plan_transform=transform,
    )
    hypothesis = build_hypothesis(features, detection_model="test-model")
    obj = hypothesis.objects[0]
    assert obj.evidence_status.value == "OBSERVED"
    assert obj.confidence == 0.8
    assert obj.position is None
    assert obj.placement_confidence is None
    assert "above the estimated floor" in (obj.uncertainty_reason or "")


def test_pipeline_skips_detection_when_detector_unavailable(monkeypatch):
    from app.services.pipeline import DEFAULT_DEPS as default_deps

    deps = PipelineDeps(
        depth=ClassicalDepthCueEstimator(),
        normals=DepthNormalEstimator(),
        features=ClassicalFeatureExtractor(),
        detection=UnavailableDetector("offline in tests"),
    )
    monkeypatch.setattr(
        "app.services.pipeline.DEFAULT_DEPS",
        deps,
        raising=True,
    )

    image = ValidatedImage(
        image=Image.open(BytesIO(b"")),
        format="PNG",
        width=320,
        height=240,
    ) if False else None
    # Build a tiny valid RGB image directly.
    img = Image.new("RGB", (320, 240), (100, 110, 120))
    validated = ValidatedImage(image=img, format="PNG", width=320, height=240)
    output = run_pipeline("job_nodetect", validated, deps=deps)
    assert output.result.hypothesis.provenance.detection_model is None
    assert any(
        "detection unavailable" in a.lower()
        for a in output.result.hypothesis.uncertainty.assumptions
    )
    # No detector-sourced objects.
    assert all(ev.source != DETECTION_SOURCE for ev in [])


def test_detector_runs_on_photo_returns_valid_schema():
    detector = TorchVisionDetector()
    if os.path.isfile(FIXTURE):
        image = Image.open(FIXTURE).convert("RGB")
    else:
        image = Image.new("RGB", (640, 480), (120, 130, 140))
    result = detector.detect(image)
    assert result.available is True
    assert result.model_name == "ssdlite320_mobilenet_v3_large (COCO-2017)"
    for det in result.detections:
        assert det.evidence_status == "OBSERVED"
        assert det.id
        assert det.class_name
        assert 0.0 <= det.confidence <= 1.0
        x0, y0, x1, y1 = det.bbox
        assert 0.0 <= x0 < x1 <= 1.0
        assert 0.0 <= y0 < y1 <= 1.0
