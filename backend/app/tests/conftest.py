"""Shared fixtures: isolate artifacts and settings for tests."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator

_TMP = tempfile.mkdtemp(prefix="floorplan-test-")
os.environ.setdefault("artifact_root", os.path.join(_TMP, "artifacts"))
os.environ.setdefault("artifact_ttl_seconds", "3600")
os.environ.setdefault("groq_api_key", "")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def make_png(width: int = 64, height: int = 48) -> bytes:
    """Create a small valid PNG for tests."""
    from io import BytesIO

    from PIL import Image

    img = Image.new("RGB", (width, height), color=(120, 130, 140))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
