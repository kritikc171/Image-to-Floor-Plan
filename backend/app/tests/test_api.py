"""Milestone 1 tests: health, validation, job lifecycle, artifact serving."""

from __future__ import annotations

import time

import pytest

from app.core.errors import AppError, ErrorCode
from app.services.image_validation import validate_and_normalize
from app.tests.conftest import make_png


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "groq" not in response.text.lower() or body["status"] == "ok"


def test_validate_accepts_png():
    image = validate_and_normalize(make_png())
    assert image.format == "PNG"
    assert image.width == 64
    assert image.height == 48
    assert image.image.mode == "RGB"


def test_validate_rejects_garbage():
    with pytest.raises(AppError) as exc_info:
        validate_and_normalize(b"not an image at all")
    assert exc_info.value.code == ErrorCode.INVALID_IMAGE


def test_validate_rejects_empty():
    with pytest.raises(AppError):
        validate_and_normalize(b"")


def test_validate_rejects_oversize_bytes(monkeypatch):
    from app.core import config

    settings = config.get_settings()
    monkeypatch.setattr(settings, "max_upload_bytes", 10)
    with pytest.raises(AppError) as exc_info:
        validate_and_normalize(make_png())
    assert exc_info.value.status_code == 413


def test_generation_lifecycle(client):
    response = client.post(
        "/api/generations",
        files={"file": ("room.png", make_png(320, 240), "image/png")},
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert response.headers.get("X-Request-ID")

    status = None
    for _ in range(50):
        poll = client.get(f"/api/generations/{job_id}")
        assert poll.status_code == 200
        status = poll.json()["status"]
        if status in ("completed", "failed"):
            break
        time.sleep(0.05)

    assert status == "completed", poll.json()
    result = poll.json()["result"]
    assert result["schema_version"] == "1.0"
    assert result["hypothesis"]["uncertainty"]["overall"] == "high"
    assert result["hypothesis"]["provenance"]["reasoning_provider"] == (
        "deterministic-fallback"
    )

    img_response = client.get(f"/api/generations/{job_id}/floorplan")
    assert img_response.status_code == 200
    assert img_response.headers["content-type"] == "image/png"
    assert img_response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_generation_rejects_empty_upload(client):
    response = client.post(
        "/api/generations",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code in (400, 422)
    body = response.json()
    assert "error" in body


def test_generation_rejects_non_image(client):
    response = client.post(
        "/api/generations",
        files={"file": ("fake.png", b"definitely not an image", "image/png")},
    )
    # Async validation fails the job, or sync validation rejects early.
    job_id = response.json().get("job_id")
    if job_id:
        for _ in range(50):
            poll = client.get(f"/api/generations/{job_id}")
            if poll.json()["status"] in ("completed", "failed"):
                break
            time.sleep(0.05)
        assert poll.json()["status"] == "failed"
        assert poll.json()["error"]["code"] == "invalid_image"
    else:
        assert response.status_code == 400


def test_unknown_job_404(client):
    response = client.get("/api/generations/job_does_not_exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "job_not_found"
