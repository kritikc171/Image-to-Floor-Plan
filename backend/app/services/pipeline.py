"""Processing pipeline entry point.

Orchestrates: validation (done by the route), preprocessing,
depth estimation, normal estimation, feature extraction,
deterministic hypothesis construction, and floor-plan
rendering. Each stage is independently replaceable and
independently testable.

The pipeline is synchronous and CPU-only. The route runs it
via ``asyncio.to_thread`` to avoid blocking the event loop.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO

import numpy as np
from PIL import Image

from app.schemas.spatial import GenerationResult
from app.services.artifacts import ARTIFACT_UPLOAD, artifact_store
from app.services.detection import Detection, DETECTION_SOURCE, ObjectDetector, TorchVisionDetector
from app.services.depth import (
    ClassicalDepthCueEstimator,
    DepthEstimator,
)
from app.services.features import (
    ClassicalFeatureExtractor,
    FeatureExtractor,
    FLOOR_LEVEL_MARGIN,
    VisualEvidence,
    normalize_floor_plan_geometry,
)
from app.services.hypothesis import build_hypothesis
from app.services.image_ops import dominant_color
from app.services.normals import (
    DepthNormalEstimator,
    NormalEstimator,
)
from app.services.renderer import render_floorplan
from app.services.floorplane import FloorPlaneEstimator
from app.services.image_validation import ValidatedImage


@dataclass(frozen=True)
class PipelineDeps:
    """Injectable pipeline stages (for testing)."""

    depth: DepthEstimator
    normals: NormalEstimator
    features: FeatureExtractor
    detection: ObjectDetector


DEFAULT_DEPS = PipelineDeps(
    depth=ClassicalDepthCueEstimator(),
    normals=DepthNormalEstimator(),
    features=ClassicalFeatureExtractor(),
    detection=TorchVisionDetector(),
)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(ax1 - ax0, 0.0) * max(ay1 - ay0, 0.0)
    area_b = max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0)
    return inter / max(area_a + area_b - inter, 1e-9)


def _detection_to_evidence(
    det: Detection,
    image: Image.Image,
    horizon_y: float,
) -> VisualEvidence:
    width, height = image.size
    x0, y0, x1, y1 = det.bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    # Decide floor-level on the bbox bottom edge (y1), the
    # ground-contact row, not the centre. A tall object's centre can
    # sit above the horizon while its base rests on the floor.
    floor_level = y1 > (horizon_y / height) + FLOOR_LEVEL_MARGIN

    mask = np.zeros((height, width), dtype=bool)
    x0_px = max(0, min(int(round(x0 * width)), width - 1))
    y0_px = max(0, min(int(round(y0 * height)), height - 1))
    x1_px = max(x0_px + 1, min(int(round(x1 * width)), width))
    y1_px = max(y0_px + 1, min(int(round(y1 * height)), height))
    mask[y0_px:y1_px, x0_px:x1_px] = True
    color = dominant_color(image, mask)
    area_fraction = (x1 - x0) * (y1 - y0)

    return VisualEvidence(
        id=det.id,
        kind="object",
        label=det.class_name,
        bbox=det.bbox,
        centroid=(cx, cy),
        color=color,
        area_fraction=area_fraction,
        strength=det.confidence,
        floor_level=floor_level,
        source=det.source,
    )


def run_pipeline(
    job_id: str,
    image: ValidatedImage,
    deps: PipelineDeps = DEFAULT_DEPS,
) -> PipelineOutput:
    """Run the baseline pipeline synchronously."""
    # Persist the normalized upload for provenance within the TTL.
    with image.image as im:
        buf = BytesIO()
        im.save(buf, format="PNG")
    artifact_store.save(job_id, ARTIFACT_UPLOAD, buf.getvalue())

    # 1. Relative depth (classical cues).
    depth_result = deps.depth.estimate(image.image)

    # 2. Surface normals (derived from depth).
    normals_result = deps.normals.estimate(depth_result)

    # 3. Visual evidence extraction (classical CV).
    features = deps.features.extract(image.image, depth_result,
                                     normals_result)

    # 3b. Floor-plane estimation: depth/normals-aware segmentation of the
    #     visible floor, with wall-floor intersection cues. Falls back to
    #     the classical horizon-based plane when the evidence is weak.
    floorplane = FloorPlaneEstimator().estimate(
        image.image, depth_result, normals_result, depth_result.horizon_y
    )
    if floorplane.quality == "good" and floorplane.transform is not None:
        features = replace(
            features,
            floor_polygon=floorplane.floor_polygon,
            plan_transform=floorplane.transform,
            wall_lines=floorplane.wall_lines or [],
            vanishing_point=floorplane.vanishing_point,
            camera_pitch_deg=floorplane.camera_pitch_deg,
            floorplane_quality=floorplane.quality,
            source_ground_corners=floorplane.source_ground_corners or [],
            geometry_correction_method=floorplane.geometry_correction_method,
            geometry_fallback_reason=floorplane.geometry_fallback_reason,
            geometry_source="floorplane-estimator",
        )
    features = normalize_floor_plan_geometry(features)

    # 4. Object detection (pretrained COCO detector -> OBSERVED evidence).
    detection_result = deps.detection.detect(image.image)
    det_evidences = [
        _detection_to_evidence(det, image.image, depth_result.horizon_y)
        for det in detection_result.detections
    ]

    # Suppress classical regions a detector already explained semantically.
    if detection_result.detections:
        filtered = [
            ev for ev in features.objects
            if not any(
                _iou(ev.bbox, det.bbox) >= 0.5
                for det in detection_result.detections
            )
        ]
        features = replace(features, objects=filtered + det_evidences)

    # 5. Spatial hypothesis (deterministic fallback reasoning;
    #    Groq reasoning is layered on in Milestone 4).
    hypothesis = build_hypothesis(
        features,
        detection_model=(
            detection_result.model_name if detection_result.available else None
        ),
        detection_available=detection_result.available,
        detection_unavailable_reason=detection_result.unavailable_reason,
    )

    # 5. Deterministic floor-plan rendering.
    floorplan_png = render_floorplan(hypothesis, job_id)

    result = GenerationResult(
        job_id=job_id,
        input={
            "width_px": image.width,
            "height_px": image.height,
            "format": image.format,
        },
        hypothesis=hypothesis,
        floorplan_image_url=f"/api/generations/{job_id}/floorplan",
    )
    return PipelineOutput(result=result, floorplan_png=floorplan_png)


class PipelineOutput:
    """Pipeline result payload."""

    def __init__(self, result: GenerationResult, floorplan_png: bytes):
        self.result = result
        self.floorplan_png = floorplan_png
