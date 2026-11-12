"""Surface-normal estimation derived from relative depth.

Camera coordinate convention (documented):
    +X → image right, +Y → image down, +Z → optical axis forward.

Method
------
Given the relative depth map ``d(x, y)`` and an *assumed* pinhole
camera model with field of view ``fov_deg`` (default 60°, a common
phone-camera approximation; true intrinsics are unknown), each pixel
is back-projected to a camera-space point:

    X = (x − cx) · d / fx,   Y = (y − cy) · d / fy,   Z = d

Surface normals are estimated from the cross product of neighbouring
back-projected points:

    n = normalize(cross(p₊ₓ − p, p₊ᵧ − p))

Border pixels and pixels with near-zero depth are marked invalid.

The result is an **estimate** derived from a relative depth map and
assumed intrinsics. It is not ground-truth geometry and is used only
as a weak cue (e.g., to distinguish near-horizontal floor surfaces
from vertical walls).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.services.depth import DepthResult


@dataclass(frozen=True)
class NormalsResult:
    """Camera-space normals estimate plus a validity mask."""

    normals: np.ndarray  # (H, W, 3) float32, unit vectors where valid
    valid: np.ndarray  # (H, W) bool
    fov_deg: float
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.normals.shape[:2]


class NormalEstimator(Protocol):
    """Replaceable normal-estimation interface."""

    def estimate(self, depth_result: DepthResult) -> NormalsResult: ...


class DepthNormalEstimator:
    """Normals from relative depth under an assumed pinhole model."""

    def __init__(self, fov_deg: float = 60.0) -> None:
        self.fov_deg = fov_deg

    def estimate(self, depth_result: DepthResult) -> NormalsResult:
        depth = depth_result.depth
        height, width = depth.shape
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        fov = float(np.deg2rad(self.fov_deg))
        focal = (width / 2.0) / np.tan(fov / 2.0)

        # Back-projected camera-space points.
        ys, xs = np.indices((height, width), dtype=np.float32)
        z = np.maximum(depth, 1e-6).astype(np.float32)
        points = np.stack(
            [
                (xs - cx) * z / focal,
                (ys - cy) * z / focal,
                z,
            ],
            axis=-1,
        ).astype(np.float32)

        # Central differences for the tangent plane.
        p_right = np.empty_like(points)
        p_right[:, :-1] = points[:, 1:]
        p_right[:, -1] = points[:, -1]
        p_down = np.empty_like(points)
        p_down[:-1, :] = points[1:, :]
        p_down[-1, :] = points[-1, :]

        tx = p_right - points
        ty = p_down - points
        normals = np.cross(tx, ty)
        length = np.linalg.norm(normals, axis=-1, keepdims=True)
        length = np.maximum(length, 1e-9)
        normals = (normals / length).astype(np.float32)

        valid = (
            (depth > 1e-4)
            & (np.abs(normals[:, :, 2]) > 1e-6)
        )
        # Border pixels have one-sided differences; mark invalid.
        valid[0, :] = False
        valid[-1, :] = False
        valid[:, 0] = False
        valid[:, -1] = False

        return NormalsResult(
            normals=normals,
            valid=valid,
            fov_deg=self.fov_deg,
            fx=float(focal),
            fy=float(focal),
            cx=float(cx),
            cy=float(cy),
        )


def floor_pixel_ratio(normals_result: NormalsResult) -> float:
    """Fraction of valid pixels whose normal is near-horizontal.

    A camera-space normal with small |Z| and negative Y points
    roughly toward the floor plane. Used as a weak cue only.
    """
    normals = normals_result.normals
    valid = normals_result.valid
    if not valid.any():
        return 0.0
    n = normals[valid]
    horizontal = (np.abs(n[:, 2]) < 0.5) & (n[:, 1] < -0.5)
    return float(horizontal.mean())
