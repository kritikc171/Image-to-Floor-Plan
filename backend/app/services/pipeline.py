"""Processing pipeline entry point.

Milestone 1 baseline: validate input, persist the normalized upload,
construct an honest minimal hypothesis (no fabricated geometry), and
render a deterministic schematic. Milestones 3–4 replace the internals
with depth/normals/feature extraction, reasoning, and uncertainty
assessment — the external contract stays identical.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.spatial import (
    EvidenceStatus,
    GenerationResult,
    Provenance,
    RoomBoundary,
    SpatialHypothesis,
    UncertaintySummary,
)
from app.services.artifacts import ARTIFACT_UPLOAD, artifact_store
from app.services.image_validation import ValidatedImage
from app.services.renderer import render_floorplan


@dataclass(frozen=True)
class PipelineOutput:
    result: GenerationResult
    floorplan_png: bytes


def run_pipeline(job_id: str, image: ValidatedImage) -> PipelineOutput:
    """Run the full pipeline synchronously (called via asyncio.to_thread)."""
    # Persist the normalized upload for provenance within the artifact TTL.
    with image.image as im:
        from io import BytesIO

        buf = BytesIO()
        im.save(buf, format="PNG")
    artifact_store.save(job_id, ARTIFACT_UPLOAD, buf.getvalue())

    hypothesis = _minimal_hypothesis(image)
    floorplan_png = render_floorplan(hypothesis, job_id)

    result = GenerationResult(
        job_id=job_id,
        input={"width_px": image.width, "height_px": image.height, "format": image.format},
        hypothesis=hypothesis,
        floorplan_image_url=f"/api/generations/{job_id}/floorplan",
    )
    return PipelineOutput(result=result, floorplan_png=floorplan_png)


def _minimal_hypothesis(image: ValidatedImage) -> SpatialHypothesis:
    """Construct the honest M1 hypothesis: everything is UNCERTAIN."""
    return SpatialHypothesis(
        room_boundary=RoomBoundary(
            polygon=None,
            evidence_status=EvidenceStatus.UNCERTAIN,
            uncertainty_reason="Baseline pipeline: geometric extraction not yet enabled.",
        ),
        objects=[],
        openings=[],
        uncertainty=UncertaintySummary(
            overall="high",
            assumptions=[
                "Single RGB photograph without camera intrinsics",
                "Metric dimensions cannot be recovered from one image",
                "Milestone 1 baseline: depth, feature, and reasoning stages are minimal",
            ],
            open_questions=["What are the room boundaries and visible objects?"],
        ),
        provenance=Provenance(
            reasoning_provider="deterministic-fallback",
            depth_method="none (milestone 1 baseline)",
            feature_method="none (milestone 1 baseline)",
            generated_views_used=0,
        ),
    )
