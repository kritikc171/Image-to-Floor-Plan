"""Typed application errors mapped to safe API responses."""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    """Stable, user-safe error codes. Never include internals."""

    INVALID_IMAGE = "invalid_image"
    IMAGE_TOO_LARGE = "image_too_large"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    VALIDATION_FAILED = "validation_failed"
    JOB_NOT_FOUND = "job_not_found"
    ARTIFACT_NOT_FOUND = "artifact_not_found"
    ARTIFACT_UNAVAILABLE = "artifact_unavailable"
    PROCESSING_FAILED = "processing_failed"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RATE_LIMITED = "rate_limited"
    INTERNAL_ERROR = "internal_error"


class AppError(Exception):
    """Application error carrying a safe, client-facing message."""

    def __init__(self, code: ErrorCode, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code

    def to_dict(self) -> dict:
        return {"error": {"code": self.code.value, "message": self.message}}
