"""Image validation and preprocessing entry point.

Trusts nothing from the client: the file is decoded with Pillow, format and
dimensions are checked after decoding, and orientation is normalized from
EXIF. Filenames and browser-supplied MIME types are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageOps

from app.core.config import get_settings
from app.core.errors import AppError, ErrorCode

ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "BMP", "GIF"}


@dataclass(frozen=True)
class ValidatedImage:
    image: Image.Image
    format: str
    width: int
    height: int


def validate_and_normalize(data: bytes) -> ValidatedImage:
    """Decode, validate, and normalize an uploaded image.

    Raises AppError with a user-safe message on any problem.
    """
    settings = get_settings()

    if not data:
        raise AppError(ErrorCode.INVALID_IMAGE, "The uploaded file is empty.")

    if len(data) > settings.max_upload_bytes:
        raise AppError(
            ErrorCode.IMAGE_TOO_LARGE,
            f"Image exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB limit.",
            status_code=413,
        )

    try:
        with Image.open(__import__("io").BytesIO(data)) as probe:
            probe.verify()
    except Exception:
        raise AppError(ErrorCode.INVALID_IMAGE, "The file is not a valid image.")

    try:
        with Image.open(__import__("io").BytesIO(data)) as img:
            fmt = (img.format or "").upper()
            if fmt not in ALLOWED_FORMATS:
                raise AppError(
                    ErrorCode.UNSUPPORTED_MEDIA_TYPE,
                    "Unsupported image format. Use PNG, JPEG, WebP, BMP, or GIF.",
                    status_code=415,
                )
            if img.width > settings.max_image_pixels or img.height > settings.max_image_pixels:
                raise AppError(
                    ErrorCode.IMAGE_TOO_LARGE,
                    f"Image dimensions exceed {settings.max_image_pixels}px per side.",
                    status_code=413,
                )
            # Normalize EXIF orientation, then convert to RGB.
            image = ImageOps.exif_transpose(img).convert("RGB")
    except AppError:
        raise
    except Exception:
        raise AppError(ErrorCode.INVALID_IMAGE, "The file could not be decoded.")

    return ValidatedImage(image=image, format=fmt, width=image.width, height=image.height)
