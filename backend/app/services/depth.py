"""Relative depth estimation (classical, CPU-only, deterministic).

Interface
---------
``DepthEstimator`` is the replaceable abstraction. The default
implementation, ``ClassicalDepthCueEstimator``, needs no model
download and no GPU. A MiDaS/DPT adapter can implement the same
interface later without touching the pipeline.

Method (documented assumptions)
-------------------------------
Indoor photographs usually contain a wall/floor boundary near a
horizon line. The estimator:

1. finds the strongest horizontal-edge row in the middle band of
   the image (a horizon proxy),
2. assigns *floor* pixels (below the horizon) a ground-plane depth
   cue ``d ∝ 1 / (y − horizon)`` — pixels near the horizon are far,
   pixels at the bottom of the frame are near,
3. assigns *upper* pixels (walls/ceiling) a depth cue that grows
   with distance from the horizon.

The result is a **unitless relative estimate** — larger means
farther. It is NOT metric depth and must not be used for
measurements. It is used only for coarse near/far ordering and
floor-plane projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from app.services.image_ops import sobel_components

MAX_WORKING_DIMENSION = 320


@dataclass(frozen=True)
class DepthResult:
    """Relative depth estimate. Larger value = farther away."""

    depth: np.ndarray  # (H, W) float32, relative
    horizon_y: float  # pixel row of the estimated horizon
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


class DepthEstimator(Protocol):
    """Replaceable depth estimation interface."""

    def estimate(self, image: Image.Image) -> DepthResult: ...


class ClassicalDepthCueEstimator:
    """Deterministic relative-depth estimator (no model download)."""

    def __init__(self, max_dimension: int = MAX_WORKING_DIMENSION) -> None:
        self.max_dimension = max_dimension

    def estimate(self, image: Image.Image) -> DepthResult:
        working = _downscale(image, self.max_dimension)
        width, height = working.size
        gray = np.asarray(working.convert("L"), dtype=np.float32) / 255.0

        horizon = _estimate_horizon(gray, height)
        depth = _build_depth(height, width, horizon)

        # Restore estimate to original resolution for consistency.
        if (width, height) != image.size:
            from PIL import Image as _Img

            depth_img = _Img.fromarray(
                (depth * 255).clip(0, 255).astype(np.uint8)
            ).resize(image.size, _Img.BILINEAR)
            depth = np.asarray(depth_img, dtype=np.float32) / 255.0
            horizon = horizon * image.size[1] / height

        return DepthResult(
            depth=depth,
            horizon_y=float(horizon),
            width=image.size[0],
            height=image.size[1],
        )


def _downscale(image: Image.Image, max_dimension: int) -> Image.Image:
    width, height = image.size
    scale = min(1.0, max_dimension / max(width, height))
    if scale >= 1.0:
        return image
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.BILINEAR,
    )


def _estimate_horizon(gray: np.ndarray, height: int) -> float:
    """Horizon proxy: row with the strongest *wall-length* horizontal edge.

    The wall/floor boundary is a horizontal structure that runs across a
    large fraction of the image width, so we score each row by how many
    strong horizontal edge pixels it contains (edge span), not by mean
    energy alone. A single noisy row no longer wins.
    """
    gx, _gy = sobel_components(gray)
    strong = np.abs(gx) > 0.15
    span = strong.mean(axis=1)  # fraction of strong horizontal pixels per row
    energy = np.abs(gx).mean(axis=1)
    score = span * (energy + 0.01)

    # Smooth with a small moving average to avoid single-row noise.
    window = 5
    if len(score) >= window:
        kernel = np.ones(window) / window
        score = np.convolve(score, kernel, mode="same")

    lo, hi = int(0.15 * height), int(0.85 * height)
    if hi <= lo:
        lo, hi = 0, height
    search = score[lo:hi]
    if search.size == 0 or float(search.max()) <= 0.0:
        return float(height) * 0.5
    return float(lo + int(np.argmax(search)))


def _build_depth(height: int, width: int, horizon: float) -> np.ndarray:
    """Construct the relative depth map from the horizon proxy."""
    ys = np.arange(height, dtype=np.float32)
    xs = np.arange(width, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    eps = 1.0
    floor_mask = yy > horizon + eps
    upper_mask = ~floor_mask

    depth = np.zeros((height, width), dtype=np.float32)

    # Floor: ground-plane cue d ∝ 1/(y − horizon), normalized.
    floor_dist = np.maximum(yy - horizon, eps)
    floor_depth = 1.0 / floor_dist
    if floor_depth.max() > 0:
        floor_depth = floor_depth / floor_depth.max()

    # Upper region: farther from the horizon = farther away.
    upper_depth = np.zeros_like(depth)
    denom = max(horizon, 1.0)
    upper_depth = 0.35 + 0.65 * np.maximum(horizon - yy, 0.0) / denom

    depth[floor_mask] = floor_depth[floor_mask]
    depth[upper_mask] = upper_depth[upper_mask]

    # Blend the boundary rows to avoid a hard seam.
    blend = np.zeros((height, width), dtype=np.float32)
    band = 3
    for dy in range(-band, band + 1):
        row = int(np.clip(int(horizon) + dy, 0, height - 1))
        blend[row, :] = 1.0
    if blend.any():
        smooth = _build_depth_reference(height, width, horizon)
        depth = depth * (1 - blend) + smooth * blend
    return depth.astype(np.float32)


def _build_depth_reference(
    height: int, width: int, horizon: float
) -> np.ndarray:
    """Linear reference field used only to smooth the horizon band."""
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    ref = np.repeat(ys, width, axis=1)
    return ref
