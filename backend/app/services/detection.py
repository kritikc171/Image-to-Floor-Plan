"""Object detection for indoor-room photographs.

Uses a torchvision-pretrained SSDlite-MobileNetV3-Large model trained on
COCO-2017. Runs locally on CPU; no paid API is involved.

Model weights
-------------
The pretrained weights (~22 MB) download from the PyTorch model zoo on
first use and are cached in the torch hub cache directory
(``$TORCH_HOME/hub/checkpoints``, typically
``~/.cache/torch/hub/checkpoints``). Set ``TORCH_HOME`` to relocate the
cache. If the download fails (offline machine), detection reports itself
as unavailable and the pipeline continues with classical visual evidence
only — it never fabricates detections.

Supported classes
-----------------
Every COCO-2017 class the model was trained on is detectable. The
indoor-relevant subset includes: bed, couch, chair, dining table, tv,
plus person, bottle, cup, laptop, potted plant, and toilet.

Classes a floor-plan user might expect but the model does NOT support:
desk, cabinet, wardrobe, shelf, door, window. These are listed in
:attr:`unsupported_examples` and are never fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PIL import Image

try:
    import torch  # noqa: F401  (availability flag)
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )
    from torchvision.transforms.v2.functional import to_dtype, to_image
except Exception:  # torch/torchvision not installed
    torch = None  # type: ignore[assignment]
    SSDLite320_MobileNet_V3_Large_Weights = None  # type: ignore[assignment]
    ssdlite320_mobilenet_v3_large = None  # type: ignore[assignment]
    to_dtype = None  # type: ignore[assignment]
    to_image = None  # type: ignore[assignment]

DEFAULT_CONFIDENCE_THRESHOLD = 0.35
MAX_DETECTIONS = 20
MODEL_NAME = "ssdlite320_mobilenet_v3_large"
DATASET = "COCO-2017"

DETECTION_SOURCE = "torchvision-ssdlite-coco"

# COCO classes the model does not cover but users often expect on a floor plan.
UNSUPPORTED_EXAMPLES = ("desk", "cabinet", "wardrobe", "shelf", "door", "window")


@dataclass(frozen=True)
class Detection:
    """One object detection from the pretrained model."""

    id: str
    class_name: str
    label: str
    bbox: tuple[float, float, float, float]  # normalized [0, 1]: x0, y0, x1, y1
    confidence: float  # model score in [0, 1], not a calibrated probability
    evidence_status: str = "OBSERVED"
    source: str = DETECTION_SOURCE


@dataclass(frozen=True)
class DetectionResult:
    """Outcome of a detection run."""

    detections: list[Detection]
    model_name: str
    available: bool
    unavailable_reason: str | None = None


@runtime_checkable
class ObjectDetector(Protocol):
    """Replaceable object-detection interface."""

    def detect(self, image: Image.Image) -> DetectionResult: ...

    @property
    def model_name(self) -> str: ...

    @property
    def supported_classes(self) -> list[str]: ...


class TorchVisionDetector:
    """COCO-pretrained SSDlite detector; weights load lazily on first use."""

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_detections: int = MAX_DETECTIONS,
    ) -> None:
        if torch is None or ssdlite320_mobilenet_v3_large is None:
            raise RuntimeError("torch and torchvision must be installed")
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self._model = None
        self._categories: list[str] | None = None

    @property
    def model_name(self) -> str:
        return f"{MODEL_NAME} ({DATASET})"

    @property
    def supported_classes(self) -> list[str]:
        self._ensure_loaded()
        assert self._categories is not None
        return [c for c in self._categories if c not in ("__background__", "N/A")]

    def _ensure_loaded(self) -> None:
        if self._model is None:
            weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            self._categories = [str(c) for c in weights.meta["categories"]]
            model = ssdlite320_mobilenet_v3_large(weights=weights)
            model.eval()
            self._model = model

    @staticmethod
    def _deduplicate(detections: list[Detection]) -> list[Detection]:
        """Suppress near-identical cross-class boxes, keeping the best score."""
        kept: list[Detection] = []
        for candidate in sorted(
            detections, key=lambda item: item.confidence, reverse=True
        ):
            x0, y0, x1, y1 = candidate.bbox
            duplicate = False
            for existing in kept:
                a0, b0, a1, b1 = existing.bbox
                ix = max(0.0, min(x1, a1) - max(x0, a0))
                iy = max(0.0, min(y1, b1) - max(y0, b0))
                intersection = ix * iy
                area_candidate = (x1 - x0) * (y1 - y0)
                area_existing = (a1 - a0) * (b1 - b0)
                union = area_candidate + area_existing - intersection
                if union > 0 and intersection / union >= 0.85:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        return kept

    def detect(self, image: Image.Image) -> DetectionResult:
        """Run detection on ``image``; never raises for model errors."""
        if torch is None or ssdlite320_mobilenet_v3_large is None or to_image is None:
            return DetectionResult(
                [], "torch unavailable", False, "torch/torchvision not installed"
            )
        try:
            self._ensure_loaded()
        except Exception as exc:  # weights download / load failure
            return DetectionResult([], self.model_name, False, str(exc))

        import torch as _torch

        rgb = image.convert("RGB")
        width, height = rgb.size
        tensor = to_dtype(to_image(rgb), dtype=_torch.float32, scale=True)
        try:
            with _torch.inference_mode():
                outputs = self._model([tensor])
        except Exception as exc:  # inference failure
            return DetectionResult([], self.model_name, False, str(exc))

        output = outputs[0]
        boxes = output["boxes"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        categories = self._categories or []

        detections: list[Detection] = []
        for index in range(min(len(boxes), self.max_detections)):
            confidence = float(scores[index])
            if confidence < self.confidence_threshold:
                continue
            x0, y0, x1, y1 = (float(v) for v in boxes[index])
            x0 = min(max(x0, 0.0), width)
            y0 = min(max(y0, 0.0), height)
            x1 = min(max(x1, 0.0), width)
            y1 = min(max(y1, 0.0), height)
            if x1 <= x0 or y1 <= y0:
                continue
            class_index = int(labels[index])
            class_name = categories[class_index] if class_index < len(categories) else f"class-{class_index}"
            detections.append(
                Detection(
                    id=f"det-{len(detections):03d}",
                    class_name=class_name,
                    label=class_name,
                    bbox=(
                        x0 / width,
                        y0 / height,
                        x1 / width,
                        y1 / height,
                    ),
                    confidence=confidence,
                )
            )

        detections = self._deduplicate(detections)
        return DetectionResult(detections, self.model_name, True)


class UnavailableDetector:
    """Detector substitute used when the model cannot run."""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    @property
    def model_name(self) -> str:
        return "unavailable"

    @property
    def supported_classes(self) -> list[str]:
        return []

    def detect(self, image: Image.Image) -> DetectionResult:  # noqa: ARG002
        return DetectionResult(
            [], "unavailable", False, self._reason
        )
