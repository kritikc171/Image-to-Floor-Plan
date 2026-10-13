"""Artifact storage: local, temporary, path-safe.

Artifacts live under ``<artifact_root>/<job_id>/``. Every path is resolved
and checked to remain inside the root; user-controlled values never
escape it. Artifacts expire after ARTIFACT_TTL_SECONDS and are deleted
lazily on access and eagerly by a background sweep.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from app.core.config import get_settings
from app.core.errors import AppError, ErrorCode


class ArtifactStore:
    """Filesystem-backed artifact store rooted at a fixed directory."""

    def __init__(self, root: str | Path | None = None) -> None:
        settings = get_settings()
        self.root = Path(root or settings.artifact_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        """Resolve a job directory, enforcing containment in the root."""
        job_path = (self.root / job_id).resolve()
        if self.root not in job_path.parents and job_path != self.root:
            raise AppError(ErrorCode.ARTIFACT_UNAVAILABLE, "Invalid artifact path.")
        return job_path

    def save(self, job_id: str, filename: str, data: bytes) -> Path:
        """Save artifact bytes; returns the written path."""
        if not filename or "/" in filename or "\\" in filename or filename != filename.strip():
            raise AppError(ErrorCode.ARTIFACT_UNAVAILABLE, "Invalid artifact name.")
        job_path = self.job_dir(job_id)
        job_path.mkdir(parents=True, exist_ok=True)
        target = job_path / filename
        target.write_bytes(data)
        return target

    def read(self, job_id: str, filename: str) -> bytes:
        """Read artifact bytes, enforcing TTL and containment."""
        target = self.job_dir(job_id) / filename
        if not target.is_file():
            raise AppError(ErrorCode.ARTIFACT_NOT_FOUND, "Artifact not found.", status_code=404)
        self._enforce_ttl(job_id)
        return target.read_bytes()

    def exists(self, job_id: str, filename: str) -> bool:
        return (self.job_dir(job_id) / filename).is_file()

    def _enforce_ttl(self, job_id: str) -> None:
        settings = get_settings()
        job_path = self.job_dir(job_id)
        if not job_path.exists():
            return
        mtime = job_path.stat().st_mtime
        if time.time() - mtime > settings.artifact_ttl_seconds:
            shutil.rmtree(job_path, ignore_errors=True)

    def cleanup_expired(self) -> int:
        """Delete job directories older than the TTL. Returns count removed."""
        settings = get_settings()
        removed = 0
        now = time.time()
        for child in self.root.iterdir():
            if child.is_dir():
                try:
                    if now - child.stat().st_mtime > settings.artifact_ttl_seconds:
                        shutil.rmtree(child, ignore_errors=True)
                        removed += 1
                except OSError:
                    continue
        return removed


artifact_store = ArtifactStore()

ARTIFACT_UPLOAD = "upload.png"
ARTIFACT_FLOORPLAN = "floorplan.png"
