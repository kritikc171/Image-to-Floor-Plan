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
  "vertical structure"). Shape-based furniture candidates are
  explicitly labeled as possible rather than confirmed.
- Image-space boxes are never treated as world-space positions;
  floor-plane projection happens in the plan transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Protocol

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

# Ground-contact gate: a pixel row is treated as floor level only when it sits
# clearly below the estimated horizon. The margin keeps ill-conditioned,
# near-horizon projections out while still recovering objects whose base is
# measurably below the horizon line. The projection itself (project_ground) is
# the final safety net: anything projecting outside [0, 1] is returned as None.
FLOOR_LEVEL_MARGIN = 0.05


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
    """Map from ground-plane (IPM) coords to plan coords.

    The baseline affine form is used for weak-evidence fallback.
    A homography is used when a visible floor quadrilateral can be rectified;
    it maps the boundary and object contact points consistently.
    """

    sx: float
    sy: float
    ox: float
    oy: float
    homography: tuple[float, ...] | None = None

    def apply(self, x: float, y: float) -> tuple[float, float]:
        if self.homography is not None:
            h = self.homography
            denominator = h[6] * x + h[7] * y + h[8]
            if abs(denominator) < 1e-12:
                return (float("inf"), float("inf"))
            return (
                (h[0] * x + h[1] * y + h[2]) / denominator,
                (h[3] * x + h[4] * y + h[5]) / denominator,
            )
        return (self.sx * x + self.ox, self.sy * y + self.oy)


def rectify_ground_quad(
    corners: list[tuple[float, float]],
) -> tuple[PlanTransform, list[tuple[float, float]]] | None:
    """Projectively rectify an ordered floor quadrilateral to a rectangle.

    The rectangle's aspect ratio comes from the average lengths of opposite
    edges in the estimated ground plane. Its longer side occupies 90% of the
    normalized plan range. This is a clearly approximate room-shape assumption,
    not recovery of hidden walls or architectural dimensions.
    """
    if len(corners) != 4 or not all(
        np.isfinite(value) for point in corners for value in point
    ):
        return None

    edges = [
        (corners[(i + 1) % 4][0] - corners[i][0],
         corners[(i + 1) % 4][1] - corners[i][1])
        for i in range(4)
    ]
    lengths = [float(np.hypot(dx, dy)) for dx, dy in edges]
    if min(lengths) <= 1e-9:
        return None

    turns = [
        edges[i][0] * edges[(i + 1) % 4][1]
        - edges[i][1] * edges[(i + 1) % 4][0]
        for i in range(4)
    ]
    if not (all(turn > 1e-9 for turn in turns) or all(turn < -1e-9 for turn in turns)):
        return None

    width = (lengths[0] + lengths[2]) / 2.0
    depth = (lengths[1] + lengths[3]) / 2.0
    if width <= 1e-9 or depth <= 1e-9:
        return None

    scale = 0.9 / max(width, depth)
    plan_width = width * scale
    plan_depth = depth * scale
    left = (1.0 - plan_width) / 2.0
    top = (1.0 - plan_depth) / 2.0
    targets = [
        (left, top),
        (left + plan_width, top),
        (left + plan_width, top + plan_depth),
        (left, top + plan_depth),
    ]

    source = np.asarray(corners, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)

    def normalize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        center = points.mean(axis=0)
        mean_distance = float(np.linalg.norm(points - center, axis=1).mean())
        if mean_distance <= 1e-12:
            return None
        scale = np.sqrt(2.0) / mean_distance
        matrix = np.asarray(
            [
                [scale, 0.0, -scale * center[0]],
                [0.0, scale, -scale * center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        normalized = (points - center) * scale
        return normalized, matrix

    normalized_source = normalize(source)
    normalized_target = normalize(target)
    if normalized_source is None or normalized_target is None:
        return None
    src, source_matrix = normalized_source
    dst, target_matrix = normalized_target

    equations: list[list[float]] = []
    values: list[float] = []
    for (x, y), (u, v) in zip(src, dst):
        equations.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        values.append(u)
        equations.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.append(v)

    matrix = np.asarray(equations, dtype=np.float64)
    condition = np.linalg.cond(matrix)
    if not np.isfinite(condition) or condition > 1e12:
        return None
    try:
        normalized_coefficients = np.linalg.solve(
            matrix, np.asarray(values, dtype=np.float64)
        )
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(normalized_coefficients).all():
        return None

    normalized_homography = np.append(normalized_coefficients, 1.0).reshape(3, 3)
    homography = np.linalg.inv(target_matrix) @ normalized_homography @ source_matrix
    if not np.isfinite(homography).all() or np.max(np.abs(homography)) <= 1e-12:
        return None
    coefficients = tuple(float(value) for value in homography.ravel())
    transform = PlanTransform(
        sx=1.0,
        sy=1.0,
        ox=0.0,
        oy=0.0,
        homography=coefficients,
    )
    polygon = [transform.apply(*point) for point in corners]
    if not all(np.isfinite(value) for point in polygon for value in point):
        return None
    return transform, polygon


def approximate_ground_quad(
    corners: list[tuple[float, float]],
) -> tuple[PlanTransform, list[tuple[float, float]]] | None:
    """Map finite ground-plane bounds to a centered rectangle approximation."""
    if len(corners) != 4 or not all(
        np.isfinite(value) for point in corners for value in point
    ):
        return None

    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x, span_y = max_x - min_x, max_y - min_y
    if span_x <= 1e-9 or span_y <= 1e-9:
        return None

    aspect = min(max(span_x / span_y, 0.65), 1.6)
    if aspect >= 1.0:
        plan_width, plan_height = 0.9, 0.9 / aspect
    else:
        plan_width, plan_height = 0.9 * aspect, 0.9
    left, top = (1.0 - plan_width) / 2.0, (1.0 - plan_height) / 2.0
    transform = PlanTransform(
        sx=plan_width / span_x,
        sy=plan_height / span_y,
        ox=left - min_x * (plan_width / span_x),
        oy=top - min_y * (plan_height / span_y),
    )
    rectangle = [
        (left, top),
        (left + plan_width, top),
        (left + plan_width, top + plan_height),
        (left, top + plan_height),
    ]
    return transform, rectangle


def normalize_floor_plan_geometry(
    features: FeatureExtractionResult,
) -> FeatureExtractionResult:
    """Ensure every usable room outline is rectangular before placing objects."""
    polygon = features.floor_polygon
    source_transform = features.plan_transform
    if not polygon or source_transform is None:
        return features

    points = list(polygon)
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points.pop()
    if len(points) < 3 or _is_axis_aligned_rectangle(points):
        return features

    correction_method = "projective homography"
    fallback_reason = None
    correction = rectify_ground_quad(points) if len(points) == 4 else None
    if correction is None:
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
        bounds = [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
        ]
        correction = approximate_ground_quad(bounds)
        correction_method = "axis-aligned rectangular approximation"
        fallback_reason = (
            "The room outline was not a usable convex quadrilateral; "
            "used bounded ground-plane bounds for a rectangular approximation."
        )
    if correction is None:
        return features

    plan_correction, corrected_polygon = correction
    original_matrix = (
        np.asarray(source_transform.homography, dtype=np.float64).reshape(3, 3)
        if source_transform.homography is not None
        else np.asarray(
            [
                [source_transform.sx, 0.0, source_transform.ox],
                [0.0, source_transform.sy, source_transform.oy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )
    correction_matrix = np.asarray(
        plan_correction.homography, dtype=np.float64
    ).reshape(3, 3)
    combined = correction_matrix @ original_matrix
    transform = PlanTransform(
        sx=1.0,
        sy=1.0,
        ox=0.0,
        oy=0.0,
        homography=tuple(float(value) for value in combined.ravel()),
    )

    return replace(
        features,
        floor_polygon=corrected_polygon,
        plan_transform=transform,
        wall_lines=[
            [plan_correction.apply(*point) for point in line]
            for line in features.wall_lines
        ],
        vanishing_point=(
            plan_correction.apply(*features.vanishing_point)
            if features.vanishing_point is not None
            else None
        ),
        geometry_correction_method=correction_method,
        geometry_fallback_reason=fallback_reason,
    )


def _is_axis_aligned_rectangle(
    points: list[tuple[float, float]],
) -> bool:
    if len(points) != 4:
        return False
    return all(
        abs(points[index][0] - points[(index + 1) % 4][0]) <= 1e-6
        or abs(points[index][1] - points[(index + 1) % 4][1]) <= 1e-6
        for index in range(4)
    )


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
    # Floor-plane outputs carried through so the hypothesis stage can build
    # room geometry from the actual wall-floor intersection cues instead of
    # discarding them. All optional; absent when the floor plane is weak.
    wall_lines: list[list[tuple[float, float]]] = field(default_factory=list)  # plan coords
    vanishing_point: Optional[tuple[float, float]] = None
    camera_pitch_deg: Optional[float] = None
    floorplane_quality: Optional[str] = None  # "good" | "poor" | "none"
    source_ground_corners: list[tuple[float, float]] = field(default_factory=list)
    geometry_correction_method: str = "unknown"
    geometry_fallback_reason: Optional[str] = None
    geometry_source: str = "classical-fallback"

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
        return y_norm > horizon + FLOOR_LEVEL_MARGIN

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
        if y_norm <= horizon + FLOOR_LEVEL_MARGIN:
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
        if not (
            np.isfinite(px)
            and np.isfinite(py)
            and 0.0 <= px <= 1.0
            and 0.0 <= py <= 1.0
        ):
            return None
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
            spans_opposite_borders = (
                (x0 <= 1 and x1 >= ww - 2)
                or (y0 <= 1 and y1 >= wh - 2)
            )
            if spans_opposite_borders:
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
            floor_level = (y1 / wh) > (horizon / wh) + FLOOR_LEVEL_MARGIN
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
        floor_polygon, transform, geometry_info = _floor_geometry(
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
            source_ground_corners=geometry_info["source_ground_corners"],
            geometry_correction_method=geometry_info["correction_method"],
            geometry_fallback_reason=geometry_info["fallback_reason"],
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
) -> tuple[
    list[tuple[float, float]] | None,
    PlanTransform | None,
    dict[str, object],
]:
    """Project the visible floor region to plan space.

    Uses inverse-perspective mapping for a ground plane:
        X = (x − cx) / (y − horizon),   Y = 1 / (y − horizon)
    The far edge is placed a small margin below the horizon to
    keep the projection finite.

    The projection is evaluated at *original* image resolution so that it
    stays consistent with ``FeatureExtractionResult.project_ground``.
    """
    if horizon <= 0 or horizon >= height - 4:
        return None, None, {
            "source_ground_corners": [],
            "correction_method": "unavailable",
            "fallback_reason": "The horizon estimate is outside the usable image.",
        }

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

    rectified = rectify_ground_quad(corners)
    if rectified is not None:
        transform, polygon = rectified
        return polygon, transform, {
            "source_ground_corners": corners,
            "correction_method": "projective homography",
            "fallback_reason": None,
        }

    approximation = approximate_ground_quad(corners)
    if approximation is None:
        return None, None, {
            "source_ground_corners": corners,
            "correction_method": "unavailable",
            "fallback_reason": "The ground-plane corner bounds are degenerate.",
        }
    transform, polygon = approximation
    return polygon, transform, {
        "source_ground_corners": corners,
        "correction_method": "axis-aligned rectangular approximation",
        "fallback_reason": (
            "The floor quadrilateral was degenerate or non-convex; "
            "used its ground-plane bounds as a rectangular approximation."
        ),
    }


def _downscale(image: Image.Image, max_dimension: int) -> Image.Image:
    width, height = image.size
    scale = min(1.0, max_dimension / max(width, height))
    if scale >= 1.0:
        return image
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.BILINEAR,
    )
