"""In-memory job store for the local prototype.

State does not survive restarts and does not support multi-worker
deployment. Job records hold safe metadata only; uploaded bytes are kept
solely as artifacts on local disk under the artifact TTL.
"""

from __future__ import annotations

import uuid
from typing import Optional

from app.schemas.spatial import GenerationJob, JobError, JobStatus


class JobStore:
    """Thread-safe-enough job registry for a single-process prototype."""

    def __init__(self) -> None:
        self._jobs: dict[str, GenerationJob] = {}
        self._active = 0

    def create(self) -> GenerationJob:
        job_id = f"job_{uuid.uuid4().hex[:24]}"
        job = GenerationJob(job_id=job_id, status=JobStatus.QUEUED)
        self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[GenerationJob]:
        return self._jobs.get(job_id)

    def set_status(self, job_id: str, status: JobStatus) -> None:
        job = self._jobs.get(job_id)
        if job is not None:
            job.status = status
            job.updated_at = job.updated_at.__class__.now(job.updated_at.tzinfo)

    def mark_failed(self, job_id: str, code: str, message: str) -> None:
        job = self._jobs.get(job_id)
        if job is not None:
            job.status = JobStatus.FAILED
            job.error = JobError(code=code, message=message)
            job.updated_at = job.updated_at.__class__.now(job.updated_at.tzinfo)

    @property
    def active_count(self) -> int:
        return self._active

    @active_count.setter
    def active_count(self, value: int) -> None:
        self._active = value


job_store = JobStore()
