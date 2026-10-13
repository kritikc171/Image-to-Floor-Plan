"""Deterministic floor-plan renderer (Milestone 1 baseline).

Consumes a validated SpatialHypothesis and produces a PNG. M1 renders an
honest schematic: title, evidence legend, boundary (if any), objects,
assumptions, and an explicit "schematic, not to scale" disclaimer.
Milestone 3 replaces the drawing internals with the full geometry renderer;
the contract (hypothesis in, PNG bytes out) stays identical.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

from app.schemas.spatial import EvidenceStatus, SpatialHypothesis

CANVAS_W, CANVAS_H = 1400, 1000
BG = (255, 255, 255)
INK = (31, 41, 55)
MUTED = (107, 114, 128)
ACCENT = (79, 70, 229)
STATUS_COLORS = {
    EvidenceStatus.OBSERVED: (22, 163, 74),
    EvidenceStatus.INFERRED: (217, 119, 6),
    EvidenceStatus.GENERATED: (139, 92, 246),
    EvidenceStatus.UNCERTAIN: (156, 163, 175),
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        ["C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"]
        + ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
           else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_floorplan(hypothesis: SpatialHypothesis, job_id: str) -> bytes:
    """Render the hypothesis to a deterministic PNG byte stream."""
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(img)

    title = _font(44, bold=True)
    body = _font(26)
    small = _font(22)

    # Header
    draw.text((60, 48), "Floor Plan Generator", font=title, fill=INK)
    draw.text((60, 108), "Approximate schematic — not to scale", font=small, fill=MUTED)
    draw.text((60, 140), f"Job {job_id}", font=small, fill=MUTED)

    # Plan area frame
    plan_box = (60, 200, 900, 880)
    draw.rectangle(plan_box, outline=(209, 213, 219), width=2)

    boundary = hypothesis.room_boundary
    if boundary.polygon:
        pts = [
            (
                plan_box[0] + 40 + p.x * (plan_box[2] - plan_box[0] - 80),
                plan_box[1] + 40 + p.y * (plan_box[3] - plan_box[1] - 80),
            )
            for p in boundary.polygon
        ]
        if len(pts) >= 3:
            color = STATUS_COLORS.get(boundary.evidence_status, MUTED)
            draw.polygon(pts, outline=color, width=4)
            note = f"boundary: {boundary.evidence_status.value.lower()}"
    else:
        draw.text(
            (plan_box[0] + 40, plan_box[1] + 40),
            "Room boundary could not be determined from this image.",
            font=body,
            fill=MUTED,
        )
        note = "boundary: unknown"

    # Objects summary
    y = 920
    if hypothesis.objects:
        summary = ", ".join(
            f"{o.label or o.type} ({o.evidence_status.value.lower()})"
            for o in hypothesis.objects[:6]
        )
        draw.text((60, y), f"Objects: {summary}", font=small, fill=INK)
        y += 32
    else:
        draw.text((60, y), "No objects could be reliably identified from this image.",
                  font=small, fill=MUTED)
        y += 32

    # Assumptions
    for assumption in hypothesis.uncertainty.assumptions[:3]:
        draw.text((60, y), f"Assumption: {assumption}", font=small, fill=MUTED)
        y += 30

    # Legend
    lx, ly = 960, 220
    draw.text((lx, ly - 40), "Evidence legend", font=body, fill=INK)
    for status in EvidenceStatus:
        draw.rectangle((lx, ly, lx + 28, ly + 28), fill=STATUS_COLORS[status])
        draw.text((lx + 44, ly + 2), status.value.title(), font=small, fill=INK)
        ly += 48

    draw.text(
        (lx, ly + 16),
        "Unknown geometry is shown as uncertain,",
        font=small,
        fill=MUTED,
    )
    draw.text((lx, ly + 48), "never as verified measurement.", font=small, fill=MUTED)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
