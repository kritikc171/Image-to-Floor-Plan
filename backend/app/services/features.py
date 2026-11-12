"""Deterministic visual feature extraction (classical CV).

Identifies useful *evidence* from the photograph using only
NumPy/Pillow: gradient-edge regions, connected components,
color, and geometry. No model download, no network, fully
deterministic.

Important conventions
---------------------
- Bounding boxes and centroids are in **normalized image
  coordinates** (0–1, origin top-left).
- ``strength`` is a heuristic edge-strength score, NOT a
  calibrated probability.
- Labels are generic geometric descriptions ("wide surface",
  "vertical structure"). Specific furniture claims require the
  reasoning stage and are always marked INFERRED.
- Image-space boxes are never treated as world-space positions;
  floor-plane projection happens in the plan transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from PIL import Image

from app.services.depth import DepthResult
from app.services.image_ops import (
    binary_closing,
    connected_components,
    dominant_color,
    sobel_magnitude,
    to_gray,
)
from app.services.normals import NormalsResult, floor_pixel_ratio

MAX_WORKING_DIMENSION = 256
MIN_AREA_FRACTION = 0.004
MIN_FILL_RATIO = 0.05


@dataclass(frozen=True)
class VisualEvidence:
    """One piece of visual evidence."""

    id: str
    kind: str  # "object" | "opening"
    label: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 normalized
    centroid: tuple[float, float]  # normalized
    color: tuple[int, int, int]
    area_fraction: float
    strength: float  # heuristic score, not calibrated
    floor_level: bool
    source: str = "classical-cv"


@dataclass(frozen=True)
class PlanTransform:
    """Affine map from ground-plane (IPM) coords to plan coords.

    Built from an inverse-perspective (ground-plane) projection
    under the assumption that the camera optical axis is
    parallel to the floor (no pitch) — a stated assumption, not
    a measurement.
    """

    sx: float
    sy: float
    ox: float
    oy: float

    def apply(self, x: float, y: float) -> tuple[float, float]:
        return (self.sx * x + self.ox, self.sy * y + self.oy)


@dataclass(frozen=True)
class FeatureExtractionResult:
    image_width: int
    image_height: int
    horizon_y: float
    objects: list[VisualEvidence] = field(default_factory=list)
    openings: list[VisualEvidence] = field(default_factory=list)
    floor_polygon: list[tuple[float, float]] | None = None  # plan coords
    plan_transform: PlanTransform | None = None
    floor_pixel_ratio: float = 0.0

    def to_plan(
        self, x_norm: float, y_norm: float
    ) -> tuple[float, float] | None:
        """Project a normalized image point to normalized plan coords.

        Applies inverse-perspective (ground-plane) projection
        followed by the stored affine normalization. Returns
        ``None`` for points at/above the horizon (no finite
        floor projection exists).
        """
        return self.project_ground(x_norm, y_norm)

    def to_plan_ground(self, bbox: tuple[float, float, float, float]) -> tuple[float, float] | None:
        """Project the bbox bottom-centre (ground-contact point) to plan coords.

        A bbox centre is not a ground-contact point: the centre of a
        tall object (e.g. a chair) lies above the floor and projects
        too far away. The bottom-centre is the defensible contact
        point for a floor-standing object.
        """
        x0, _y0, x1, y1 = bbox
        return self.project_ground((x0 + x1) / 2.0, y1)

    def ground_width(self, bbox: tuple[float, float, float, float]) -> float | None:
        """Project the bbox bottom edge and return its plan-space width."""
        x0, _y0, x1, y1 = bbox
        a = self.project_ground(x0, y1)
        b = self.project_ground(x1, y1)
        if a is None or b is None:
            return None
        return abs(b[0] - a[0])

    def is_floor_level(self, y_norm: float) -> bool:
        """True when the given normalized image row is below the horizon band."""
        if self.plan_transform is None:
            return False
        horizon = self.horizon_y / self.image_height
        return y_norm > horizon + 0.12

    def project_ground(
        self, x_norm: float, y_norm: float
    ) -> tuple[float, float] | None:
        """Project a normalized image point to normalized plan coords.

        Applies inverse-perspective (ground-plane) projection
        followed by the stored affine normalization. Returns
        ``None`` for points at/above the horizon (no finite
        floor projection exists).
        """
        if self.plan_transform is None:
            return None
        horizon = self.horizon_y / self.image_height
        if y_norm <= horizon + 0.12:
            return None
        y_px = y_norm * self.image_height
        x_px = x_norm * self.image_width
        cx = (self.image_width - 1) / 2.0
        denom = max(y_px - self.horizon_y, 0.5)
        ipm_x = (x_px - cx) / denom
        ipm_y = 1.0 / denom
        px, py = self.plan_transform.apply(ipm_x, ipm_y)
        # A projection whose vertical coordinate falls outside the unit
        # plan cannot be a defensible top-down position; mark it as
        # unavailable (None) instead of silently clamping it to y=0.
        if not (0.0 <= py <= 1.0):
            return None
        px = min(max(px, 0.0), 1.0)
        return (px, py)


class FeatureExtractor(Protocol):
    """Replaceable feature-extraction interface."""

    def extract(
        self,
        image: Image.Image,
        depth: DepthResult,
        normals: NormalsResult,
    ) -> FeatureExtractionResult: ...


class ClassicalFeatureExtractor:
    """Gradient-region feature extractor (deterministic)."""

    def __init__(self, max_dimension: int = MAX_WORKING_DIMENSION) -> None:
        self.max_dimension = max_dimension

    def extract(
        self,
        image: Image.Image,
        depth: DepthResult,
        normals: NormalsResult,
    ) -> FeatureExtractionResult:
        working = _downscale(image, self.max_dimension)
        ww, wh = working.size
        scale_x = image.size[0] / ww
        scale_y = image.size[1] / wh
        horizon = depth.horizon_y / scale_y

        gray = to_gray(working)
        edges = sobel_magnitude(gray)
        threshold = float(np.percentile(edges, 65))
        binary = edges > max(threshold, 0.05)
        binary = binary_closing(binary, iterations=2)

        components = connected_components(binary)
        total_pixels = float(ww * wh)

        objects: list[VisualEvidence] = []
        openings: list[VisualEvidence] = []
        counter = 0

        for comp in components:
            area = int(comp.sum())
            if area < MIN_AREA_FRACTION * total_pixels:
                continue
            ys, xs = np.where(comp)
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            bw = x1 - x0 + 1
            bh = y1 - y0 + 1
            if bw <= 0 or bh <= 0:
                continue
            aspect = bw / bh
            if aspect < 0.2 or aspect > 5.0:
                continue
            fill = area / float(bw * bh)
            if fill < MIN_FILL_RATIO:
                continue

            cx_n = (x0 + x1) / 2.0 / ww
            cy_n = (y0 + y1) / 2.0 / wh
            strength = float(edges[comp].mean())
            # floor_level is decided on the bbox bottom edge (y1), the
            # ground-contact row, not the centre. A tall object's centre
            # can sit above the horizon while its base rests on the floor.
            floor_level = (y1 / wh) > (horizon / wh) + 0.12
            color = dominant_color(working, comp)

            counter += 1
            touches_border = x0 <= 1 or x1 >= ww - 2
            near_top = cy_n < 0.7

            if touches_border and near_top and aspect < 0.5:
                evidence = VisualEvidence(
                    id=f"opening-{counter:03d}",
                    kind="opening",
                    label="door or window candidate",
                    bbox=(
                        x0 / ww,
                        y0 / wh,
                        x1 / ww,
                        y1 / wh,
                    ),
                    centroid=(cx_n, cy_n),
                    color=color,
                    area_fraction=area / total_pixels,
                    strength=strength,
                    floor_level=False,
                )
                openings.append(evidence)
                continue

            label = _geometric_label(aspect, floor_level)
            objects.append(
                VisualEvidence(
                    id=f"object-{counter:03d}",
                    kind="object",
                    label=label,
                    bbox=(x0 / ww, y0 / wh, x1 / ww, y1 / wh),
                    centroid=(cx_n, cy_n),
                    color=color,
                    area_fraction=area / total_pixels,
                    strength=strength,
                    floor_level=floor_level,
                )
            )

        # Floor geometry: approximate the visible floor region as the
        # region below the horizon, bounded by strong vertical edges
        # when present, else the full frame width.
        floor_polygon, transform = _floor_geometry(
            image.size[0], image.size[1], horizon, objects
        )

        return FeatureExtractionResult(
            image_width=image.size[0],
            image_height=image.size[1],
            horizon_y=depth.horizon_y,
            objects=objects,
            openings=openings,
            floor_polygon=floor_polygon,
            plan_transform=transform,
            floor_pixel_ratio=floor_pixel_ratio(normals),
        )


def _geometric_label(aspect: float, floor_level: bool) -> str:
    """Generic geometric description — deliberately not furniture claims."""
    if floor_level:
        if aspect > 1.5:
            return "wide surface"
        if aspect < 0.67:
            return "tall object"
        return "compact object"
    if aspect < 0.67:
        return "vertical structure"
    if aspect > 1.5:
        return "wall feature"
    return "object"


def _floor_geometry(
    width: int,
    height: int,
    horizon: float,
    objects: list[VisualEvidence],
) -> tuple[list[tuple[float, float]] | None, PlanTransform | None]:
    """Project the visible floor region to plan space.

    Uses inverse-perspective mapping for a ground plane:
        X = (x − cx) / (y − horizon),   Y = 1 / (y − horizon)
    The far edge is placed a small margin below the horizon to
    keep the projection finite.

    The projection is evaluated at *original* image resolution so that it
    stays consistent with ``FeatureExtractionResult.project_ground``.
    """
    if horizon <= 0 or horizon >= height - 4:
        return None, None

    margin = max(2.0, 0.05 * height)
    far_y = horizon + margin
    cx = (width - 1) / 2.0

    def ipm(x_px: float, y_px: float) -> tuple[float, float]:
        denom = max(y_px - horizon, 0.5)
        return ((x_px - cx) / denom, 1.0 / denom)

    corners = [
        ipm(0.0, far_y),        # far-left
        ipm(float(width), far_y),  # far-right
        ipm(float(width), height - 1),  # near-right
        ipm(0.0, height - 1),      # near-left
    ]

    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x if max_x > min_x else 1.0
    span_y = max_y - min_y if max_y > min_y else 1.0

    transform = PlanTransform(
        sx=0.9 / span_x,
        sy=0.9 / span_y,
        ox=0.05 - min_x * (0.9 / span_x),
        oy=0.05 - min_y * (0.9 / span_y),
    )

    polygon = [transform.apply(*c) for c in corners]
    return polygon, transform


def _downscale(image: Image.Image, max_dimension: int) -> Image.Image:
    width, height = image.size
    scale = min(1.0, max_dimension / max(width, height))
    if scale >= 1.0:
        return image
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.BILINEAR,
    )
