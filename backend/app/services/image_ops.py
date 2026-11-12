"""Shared deterministic image operations (NumPy/Pillow only).

No OpenCV dependency: gradients and connected components are
implemented directly so the pipeline stays dependency-light,
deterministic, and CPU-only.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

# 3x3 Sobel kernels (Pillow, separable not required at these sizes)
_SOBEL_X = ImageFilter.Kernel((3, 3), (-1, 0, 1, -2, 0, 2, -1, 0, 1), scale=1)
_SOBEL_Y = ImageFilter.Kernel((3, 3), (-1, -2, -1, 0, 0, 0, 1, 2, 1), scale=1)


def to_gray(image: Image.Image) -> np.ndarray:
    """Convert to a float32 grayscale array in [0, 1]."""
    gray = image.convert("L")
    return np.asarray(gray, dtype=np.float32) / 255.0


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Gradient magnitude via Pillow Sobel kernels."""
    img = Image.fromarray((gray * 255.0).clip(0, 255).astype(np.uint8))
    gx = np.asarray(img.filter(_SOBEL_X), dtype=np.float32) / 255.0
    gy = np.asarray(img.filter(_SOBEL_Y), dtype=np.float32) / 255.0
    return np.sqrt(gx * gx + gy * gy)


def sobel_components(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (gx, gy) directional gradients as float32 arrays in [0, 1]."""
    img = Image.fromarray((gray * 255.0).clip(0, 255).astype(np.uint8))
    gx = np.asarray(img.filter(_SOBEL_X), dtype=np.float32) / 255.0
    gy = np.asarray(img.filter(_SOBEL_Y), dtype=np.float32) / 255.0
    return gx, gy


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected component label masks (boolean arrays) for True pixels.

    Deterministic BFS; ``mask`` is a boolean 2D array.
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for y in range(h):
        for x in range(w):
            if mask[y, x] and not visited[y, x]:
                comp = np.zeros_like(mask, dtype=bool)
                stack = [(y, x)]
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    comp[cy, cx] = True
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < h
                            and 0 <= nx < w
                            and mask[ny, nx]
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                components.append(comp)
    return components


def binary_closing(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Morphological closing (dilate then erode) with a 3x3 cross kernel."""
    result = mask.copy()
    for _ in range(iterations):
        dilated = _dilate(result)
        result = _erode(dilated)
    return result


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _erode(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] &= mask[:-1, :]
    out[:-1, :] &= mask[1:, :]
    out[:, 1:] &= mask[:, :-1]
    out[:, :-1] &= mask[:, 1:]
    return out


def dominant_color(image: Image.Image, mask: np.ndarray) -> tuple[int, int, int]:
    """Median RGB color of the masked region (deterministic)."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    if mask.shape != rgb.shape[:2]:
        mask = _resize_mask(mask, rgb.shape[:2])
    pixels = rgb[mask]
    if pixels.size == 0:
        return (128, 128, 128)
    med = np.median(pixels, axis=0).astype(int)
    return (int(med[0]), int(med[1]), int(med[2]))


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image as _Img

    m = _Img.fromarray((mask * 255).astype(np.uint8))
    m = m.resize((shape[1], shape[0]), _Img.NEAREST)
    return np.asarray(m) > 127
