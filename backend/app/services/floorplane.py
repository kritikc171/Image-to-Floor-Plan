"""Floor-plane estimation from a single indoor photograph.

Produces a defensible *approximate* ground-plane coordinate system using
only the available visual evidence: relative depth, surface normals,
gradient edges, the horizon proxy, and object contact points.

The result is never presented as metric geometry. Where the evidence is
thin, the floor plane is marked uncertain and the caller keeps objects
unplaced rather than fabricating coordinates.

Coordinate spaces (documented in app.schemas.spatial)
-----------------------------------------------------
1. ImagePixel      — camera pixels, origin top-left.
2. NormalizedImage — x/y in [0, 1], origin top-left of the image.
3. GroundPoint     — inverse-perspective ground-plane units
   (X right, Y into the scene). Arbitrary units, not metric.
4. PlanPoint       — normalized plan coords in [0, 1], origin top-left.
5. CanvasPixel     — renderer output pixels.

Stage 3 → 4 is an explicit affine normalization; nothing in the renderer
ever consumes image or ground coordinates directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.services.depth import DepthResult
from app.services.features import PlanTransform
from app.services.image_ops import sobel_components, to_gray
from app.services.normals import NormalsResult

# Rows above this fraction of the image height can never be floor.
_FLOOR_MIN_Y_FRAC = 0.55
# A pixel is treated as floor when its normal points roughly upward
# (camera-space +Y is down, so floor normals have negative Y).
_NORMAL_FLOOR_Y = -0.45
_NORMAL_FLOOR_Z = 0.5


@dataclass(frozen=True)
class FloorPlaneResult:
    """Estimated ground-plane coordinate system.

    ``transform`` maps ground-plane (IPM) points to normalized plan
    coordinates. ``floor_polygon`` is the visible floor region in plan
    space — the *visible* floor, not a verified complete room boundary.
    ``wall_lines`` are candidate wall-floor intersection lines in plan
    space (each as [(x1, y1), (x2, y2)]). ``vanishing_point`` is the
    estimated camera vanishing point in plan space, or None.
    ``camera_pitch_deg`` is the estimated pitch (0 = level), or None.
    ``quality`` is one of "good" | "poor" | "none".
    """

    transform: Optional[PlanTransform]
    floor_polygon: Optional[list[tuple[float, float]]]  # plan coords
    wall_lines: list[list[tuple[float, float]]]
    vanishing_point: Optional[tuple[float, float]]
    camera_pitch_deg: Optional[float]
    quality: str  # "good" | "poor" | "none"
    uncertainty_reason: Optional[str]


class FloorPlaneEstimator:
    """Estimate the visible floor plane from classical CV cues."""

    def __init__(self, max_dimension: int = 256) -> None:
        self.max_dimension = max_dimension

    def estimate(
        self,
        image: Image.Image,
        depth: DepthResult,
        normals: NormalsResult,
        horizon_y: float,
    ) -> FloorPlaneResult:
        width, height = image.size
        working = _downscale(image, self.max_dimension)
        ww, wh = working.size
        scale_x = width / ww
        scale_y = height / wh
        horizon = horizon_y / scale_y

        gray = to_gray(working)
        floor_mask = _segment_floor(working, depth, normals, horizon, wh, ww)

        if floor_mask is None or not floor_mask.any():
            return FloorPlaneResult(
                transform=None,
                floor_polygon=None,
                wall_lines=[],
                vanishing_point=None,
                camera_pitch_deg=None,
                quality="none",
                uncertainty_reason=(
                    "No reliable floor region could be segmented "
                    "(depth/normals/horizon cues all weak)."
                ),
            )

        # Wall-floor intersection: rows where floor gives way to wall,
        # tracked as candidate vertical wall lines in image space.
        edge_candidates = _wall_floor_edges(floor_mask, gray, horizon, wh)

        # Ground-plane projection of the visible floor region.
        polygon, transform, ground_info = _floor_polygon_and_transform(
            ww, wh, horizon, floor_mask, width, height, horizon_y
        )

        wall_lines_plan: list[list[tuple[float, float]]] = []
        if transform is not None and polygon is not None:
            cx = (width - 1) / 2.0
            for line in edge_candidates:
                # Project at ORIGINAL resolution so the points line up with the
                # transform (which was fit on original-resolution IPM corners).
                x0o, y0o = line[0][0] * scale_x, line[0][1] * scale_y
                x1o, y1o = line[1][0] * scale_x, line[1][1] * scale_y
                p0 = transform.apply(*_ipm(x0o, y0o, horizon_y, cx))
                p1 = transform.apply(*_ipm(x1o, y1o, horizon_y, cx))
                wall_lines_plan.append([p0, p1])
        vanishing_point = _vanishing_point(wall_lines_plan)
        pitch = _camera_pitch(horizon, wh)

        quality = "good" if transform is not None and len(polygon or []) >= 3 else "poor"
        if transform is None:
            quality = "none"

        return FloorPlaneResult(
            transform=transform,
            floor_polygon=polygon,
            wall_lines=wall_lines_plan,
            vanishing_point=vanishing_point,
            camera_pitch_deg=pitch,
            quality=quality,
            uncertainty_reason=(
                None
                if quality == "good"
                else "Floor plane estimated from weak cues; treat as approximate."
            ),
        )


# ---------------------------------------------------------------------------
# Floor segmentation
# ---------------------------------------------------------------------------

def _segment_floor(
    image: Image.Image,
    depth: DepthResult,
    normals: NormalsResult,
    horizon: float,
    wh: int,
    ww: int,
) -> Optional[np.ndarray]:
    """Boolean mask of likely floor pixels in the working image."""

    depth_w = depth.width
    depth_h = depth.height
    depth_img = np.asarray(depth.depth)

    # Depth resized to working resolution (relative ordering preserved).
    if (depth_w, depth_h) != (ww, wh):
        from PIL import Image as _Img

        depth_img = np.asarray(
            _Img.fromarray((depth_img * 255).clip(0, 255).astype(np.uint8)).resize(
                (ww, wh), _Img.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0

    normals_valid = np.zeros((wh, ww), dtype=bool)
    normals_up = np.zeros((wh, ww), dtype=bool)
    if normals.normals is not None and normals.valid is not None:
        n = np.asarray(normals.normals)
        v = np.asarray(normals.valid)
        if n.shape[:2] != (wh, ww):
            from PIL import Image as _Img

            n_small = np.asarray(
                _Img.fromarray((n * 255).clip(0, 255).astype(np.uint8)).resize(
                    (ww, wh), _Img.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0
            v_small = np.asarray(
                _Img.fromarray((v * 255).astype(np.uint8)).resize(
                    (ww, wh), _Img.NEAREST
                ),
                dtype=bool,
            )
        else:
            n_small, v_small = n, v
        normals_valid = v_small
        normals_up = (
            v_small
            & (n_small[:, :, 1] < _NORMAL_FLOOR_Y)
            & (np.abs(n_small[:, :, 2]) < _NORMAL_FLOOR_Z)
        )

    ys, xs = np.meshgrid(np.arange(wh, dtype=np.float32), np.arange(ww, dtype=np.float32), indexing="ij")
    below_horizon = ys > horizon + max(2.0, 0.05 * wh)
    near_enough = ys < wh - 1  # keep away from the bottom frame edge

    if normals_valid.any() and normals_up.any():
        mask = below_horizon & normals_up & near_enough
    else:
        # Normals unavailable or unusable: fall back to depth-ordered
        # region below the horizon. Farther floor pixels have smaller
        # relative depth; the nearest floor is at the bottom of the frame.
        mask = below_horizon & near_enough

    if not mask.any():
        return None

    # Keep the largest connected component below the horizon (the main
    # visible floor), discarding small isolated patches.
    from app.services.image_ops import connected_components

    comps = connected_components(mask)
    if not comps:
        return None
    comps.sort(key=lambda c: int(c.sum()), reverse=True)
    return comps[0]


def _wall_floor_edges(
    floor_mask: np.ndarray, gray: np.ndarray, horizon: float, wh: int
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Candidate wall-floor intersection lines in working-image pixels.

    Scans each row for the floor/wall transition; dominant near-vertical
    transitions become wall line candidates.
    """
    h, w = floor_mask.shape
    lines: list[tuple[tuple[float, float], tuple[float, float]]] = []
    transitions: dict[int, list[float]] = {}
    for y in range(int(horizon) + 5, h - 2):
        row = floor_mask[y]
        if not row.any():
            continue
        # Rightmost floor pixel in the row = floor/wall edge on the right;
        # leftmost = edge on the left.
        xs = np.where(row)[0]
        transitions.setdefault(int(xs.min()), []).append(float(y))
        transitions.setdefault(int(xs.max()), []).append(float(y))

    # Cluster transition x positions into a few vertical lines.
    xs_sorted = sorted(transitions)
    clusters: list[list[int]] = []
    for x in xs_sorted:
        for cluster in clusters:
            if abs(x - cluster[0]) <= 6:
                cluster.append(x)
                break
        else:
            clusters.append([x])

    for cluster in clusters:
        if len(cluster) < max(4, int(0.02 * wh)):
            continue
        cx = float(sum(cluster)) / len(cluster)
        ys = []
        for x in cluster:
            ys.extend(transitions[x])
        if not ys:
            continue
        y0, y1 = min(ys), max(ys)
        if y1 - y0 < 8:
            continue
        lines.append(((cx, max(y0, int(horizon) + 5)), (cx, y1)))
    return lines[:8]


# ---------------------------------------------------------------------------
# Ground-plane projection
# ---------------------------------------------------------------------------

def _ipm(x_px: float, y_px: float, horizon: float, cx: float) -> tuple[float, float]:
    """Inverse-perspective ground-plane point for an image pixel.

    X = (x − cx) / (y − horizon), Y = 1 / (y − horizon).
    """
    denom = max(y_px - horizon, 0.5)
    return ((x_px - cx) / denom, 1.0 / denom)


def _floor_polygon_and_transform(
    ww: int,
    wh: int,
    horizon: float,
    floor_mask: np.ndarray,
    width: int,
    height: int,
    horizon_y: float,
) -> tuple[Optional[list[tuple[float, float]]], Optional[PlanTransform], dict]:
    """Project the visible floor region to normalized plan coordinates.

    The floor mask's extreme projected corners define the visible floor
    polygon; an affine normalization maps it into [0, 1]^2 without
    forcing a rectangle.

    The inverse-perspective projection is evaluated at *original* image
    resolution so that it stays consistent with
    ``FeatureExtractionResult.project_ground`` (which also projects at
    original resolution). The working-resolution mask corners are scaled
    back to original pixels before projection.
    """
    if horizon_y <= 0 or horizon_y >= height - 4:
        return None, None, {}

    ux = width / ww
    uy = height / wh
    cx = (width - 1) / 2.0

    def ipm(x_px: float, y_px: float) -> tuple[float, float]:
        denom = max(y_px - horizon_y, 0.5)
        return ((x_px - cx) / denom, 1.0 / denom)

    # Candidate corners of the visible floor region: the near edge (bottom
    # of the mask) and the far edge (topmost mask rows), at left/right
    # extremes of the mask. Mask pixels are in working resolution; scale to
    # original before projecting.
    ys, xs = np.where(floor_mask)
    if len(xs) == 0:
        return None, None, {}
    near_y = float(wh - 1) * uy
    far_y = float(ys.min()) * uy
    mask_xs = xs[ys == ys.min()] if (ys == ys.min()).any() else xs
    left_x = float(xs.min()) * ux
    right_x = float(xs.max()) * ux
    far_left_x = float(mask_xs.min()) * ux
    far_right_x = float(mask_xs.max()) * ux

    corners = [
        ipm(far_left_x, far_y),      # far-left
        ipm(far_right_x, far_y),     # far-right
        ipm(right_x, near_y),        # near-right
        ipm(left_x, near_y),         # near-left
    ]

    gxs = [c[0] for c in corners]
    gys = [c[1] for c in corners]
    min_x, max_x = min(gxs), max(gxs)
    min_y, max_y = min(gys), max(gys)
    span_x = max_x - min_x if max_x > min_x else 1.0
    span_y = max_y - min_y if max_y > min_y else 1.0

    if span_x <= 0 or span_y <= 0:
        return None, None, {}

    transform = PlanTransform(
        sx=0.9 / span_x,
        sy=0.9 / span_y,
        ox=0.05 - min_x * (0.9 / span_x),
        oy=0.05 - min_y * (0.9 / span_y),
    )

    polygon = [transform.apply(*c) for c in corners]
    return polygon, transform, {}


def _vanishing_point(
    lines: list[list[tuple[float, float]]],
) -> Optional[tuple[float, float]]:
    """Intersection of the two most distant parallel wall-line pairs.

    Very weak cue; returned only when two non-parallel lines cross
    inside the plan.
    """
    if len(lines) < 2:
        return None

    def line_coeffs(a: tuple[float, float], b: tuple[float, float]):
        x1, y1 = a
        x2, y2 = b
        return (y2 - y1, x1 - x2, y1 * x2 - x1 * y2)

    best = None
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a1, b1, c1 = line_coeffs(*lines[i])
            a2, b2, c2 = line_coeffs(*lines[j])
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-6:
                continue
            x = (b1 * c2 - b2 * c1) / det
            y = (c1 * a2 - c2 * a1) / det
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                dist = (x - 0.5) ** 2 + (y - 0.5) ** 2
                if best is None or dist < best[0]:
                    best = (dist, (x, y))
    return best[1] if best else None


def _camera_pitch(horizon: float, wh: int) -> Optional[float]:
    """Rough pitch estimate from horizon position (degrees, 0 = level).

    Purely indicative; assumes a roughly level camera with the horizon
    near the image middle when pitch is zero.
    """
    if wh <= 0:
        return None
    mid = wh / 2.0
    frac = (horizon - mid) / mid  # positive = horizon below middle
    # Heuristic mapping, not a measurement.
    return round(float(frac * 25.0), 1)


def _downscale(image: Image.Image, max_dimension: int) -> Image.Image:
    width, height = image.size
    scale = min(1.0, max_dimension / max(width, height))
    if scale >= 1.0:
        return image
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.BILINEAR,
    )
