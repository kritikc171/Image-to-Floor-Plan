"""Generation endpoints: create job, poll status, fetch floor plan."""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, File, Request, Response, UploadFile, Header

from app.core.config import get_settings
from app.core.errors import AppError, ErrorCode
from app.core.logging import logger
from app.schemas.api import GenerationCreateResponse, GenerationStatusResponse
from app.schemas.spatial import JobStatus
from app.services.artifacts import ARTIFACT_FLOORPLAN, artifact_store
from app.services.image_validation import validate_and_normalize
from app.services.jobs import job_store
from app.services.pipeline import run_pipeline

router = APIRouter(tags=["generations"])


async def _process_job(job_id: str, file_bytes: bytes) -> None:
    """Background processing: validate, run pipeline, persist artifacts."""
    job_store.set_status(job_id, JobStatus.PROCESSING)
    try:
        image = validate_and_normalize(file_bytes)
    except AppError as exc:
        job_store.mark_failed(job_id, exc.code.value, exc.message)
        return
    except Exception:
        logger.exception("pipeline_error job=%s", job_id)
        job_store.mark_failed(job_id, "processing_failed", "Image processing failed.")
        return

    try:
        output = await asyncio.to_thread(run_pipeline, job_id, image)
        artifact_store.save(job_id, ARTIFACT_FLOORPLAN, output.floorplan_png)
        job_store.get(job_id).result = output.result  # type: ignore[index]
        job_store.set_status(job_id, JobStatus.COMPLETED)
        logger.info("job_completed job=%s", job_id)
    except Exception:
        logger.exception("pipeline_error job=%s", job_id)
        job_store.mark_failed(job_id, "processing_failed", "Image processing failed.")
    finally:
        job_store.active_count = max(0, job_store.active_count - 1)


@router.post("/generations", response_model=GenerationCreateResponse, status_code=202)
async def create_generation(
    request: Request,
    file: UploadFile = File(...),
):
    """Accept exactly one indoor-room photograph and start processing."""
    settings = get_settings()

    if job_store.active_count >= settings.max_concurrent_jobs:
        raise AppError(ErrorCode.RATE_LIMITED, "Too many jobs in progress. Retry shortly.",
                       status_code=429)

    data = await file.read()
    if not data:
        raise AppError(ErrorCode.INVALID_IMAGE, "No image data was provided.")

    job = job_store.create()
    job_store.active_count = job_store.active_count + 1
    request_id = request.state.request_id if hasattr(request.state, "request_id") else job.job_id
    logger.info("job_created job=%s request=%s bytes=%d", job.job_id, request_id, len(data))

    asyncio.create_task(_process_job(job.job_id, data))
    return GenerationCreateResponse(
        job_id=job.job_id,
        status=job.status.value,
        created_at=job.created_at,
    )


@router.get("/generations/{job_id}", response_model=GenerationStatusResponse)
async def get_generation(job_id: str):
    """Poll job status; result is present once status is completed."""
    job = job_store.get(job_id)
    if job is None:
        raise AppError(ErrorCode.JOB_NOT_FOUND, "Unknown job.", status_code=404)
    return GenerationStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        created_at=job.created_at,
        updated_at=job.updated_at,
        error=job.error.model_dump(mode="json") if job.error else None,
        result=job.result,
    )


@router.get("/generations/{job_id}/floorplan")
async def get_floorplan(job_id: str):
    """Return the rendered floor-plan PNG for a completed job."""
    job = job_store.get(job_id)
    if job is None:
        raise AppError(ErrorCode.JOB_NOT_FOUND, "Unknown job.", status_code=404)
    if job.status != JobStatus.COMPLETED:
        raise AppError(ErrorCode.ARTIFACT_UNAVAILABLE,
                       "Floor plan is not ready yet.", status_code=409)
    try:
        png = artifact_store.read(job_id, ARTIFACT_FLOORPLAN)
    except AppError:
        raise
    return Response(content=png, media_type="image/png",
                    headers={"Content-Disposition": f'inline; filename="{job_id}-floorplan.png"'})
