"""Architectural floor-plan renderer.

Consumes a validated :class:`SpatialHypothesis` and produces a clean,
deterministic PNG. Drawing is a pure function of the hypothesis.

The PNG is an *architectural schematic*, not a computer-vision
diagnostic: it contains room walls, openings, furniture footprints and
a legend only. It never contains job IDs, debug text, model names,
reasoning traces or duplicated headings.

Coordinate convention
---------------------
- ``PlanPoint`` x/y are normalized floor-plan coords in [0, 1], origin
  top-left of the plan, x right, y down.
- The renderer maps the normalized plan onto the main drawing box.
- Image-space coordinates are never used as drawing coordinates.

Style
-----
- White/very light background; conventional architectural linework.
- Walls: thick solid outlines with an inner edge cue, connected at
  corners; room interior tinted very lightly.
- Doors/windows: openings in the wall centreline with a leaf / window
  mark and a swing arc, drawn only when the hypothesis carries them.
- Furniture: class-specific top-down silhouettes with restrained material
  colors and details, scaled from the estimated footprint and rotated by
  the estimated orientation. Evidence status remains marked independently.
- Numbered markers are supplementary identifiers next to furniture;
  they never replace a symbol.
- OBSERVED/INFERRED/UNCERTAIN colour key in the legend, in its own
  right-hand region that does not overlap the plan.
- Footer states "not to scale" once.
"""

from __future__ import annotations

import math
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

from app.schemas.spatial import (
    EvidenceStatus,
    Opening,
    SpatialHypothesis,
)

CANVAS_W, CANVAS_H = 1400, 1000

# Main architectural plan (left). Walls and footprints live here.
PLAN_BOX = (80, 150, 920, 850)
PLAN_PAD = 40.0

# Right-hand information column (own region, never overlaps the plan).
COL_X = 970
COL_W = CANVAS_W - COL_X - 60
FOOTER_Y = 950

# Palette.
INK = (20, 30, 48)          # walls / text
PAPER = (255, 255, 255)     # background
ROOM_TINT = (247, 250, 253)  # room interior fill
LINE_LIGHT = (223, 230, 240)
MUTED = (100, 116, 139)

WALL = (30, 41, 59)
WALL_WIDTH = 14
WALL_INNER = 5

DOOR_LEAF = (30, 41, 59)
SWING = (120, 135, 155)
WINDOW = (60, 75, 100)

MARKER_R = 11
MARKER_MIN_DIST = 30

SYM_MIN = 28.0
SYM_MAX = 200.0

STATUS_COLORS = {
    EvidenceStatus.OBSERVED: (34, 197, 94),
    EvidenceStatus.INFERRED: (245, 158, 11),
    EvidenceStatus.GENERATED: (168, 85, 247),
    EvidenceStatus.UNCERTAIN: (148, 163, 184),
}
STATUS_FILLS = {
    EvidenceStatus.OBSERVED: (220, 252, 231),
    EvidenceStatus.INFERRED: (254, 243, 199),
    EvidenceStatus.GENERATED: (237, 233, 254),
    EvidenceStatus.UNCERTAIN: (243, 244, 246),
}
FURNITURE_FILLS = {
    "bed": (224, 214, 197),
    "sofa": (193, 176, 154),
    "chair": (205, 184, 151),
    "table": (196, 160, 112),
    "desk": (190, 156, 112),
    "cabinet": (183, 145, 101),
    "tv": (37, 48, 65),
    "generic": (207, 211, 216),
}
LIGHT_DETAIL = (248, 245, 238)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_floorplan(hypothesis: SpatialHypothesis, job_id: str) -> bytes:
    """Render the hypothesis to a deterministic PNG byte stream.

    ``job_id`` is kept for signature compatibility and is never drawn.
    """
    del job_id  # never appears in the drawing
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), PAPER)
    draw = ImageDraw.Draw(img)

    title = _font(40, bold=True)
    subtitle = _font(19)
    label_font = _font(24, bold=True)
    small = _font(17)
    tiny = _font(14)

    # ---- 1. Header (single title, single notice) ----------------------
    draw.text((60, 38), "Floor Plan Generator", font=title, fill=INK)
    draw.text(
        (60, 90),
        "Approximate schematic from a single photo — not to scale.",
        font=subtitle,
        fill=MUTED,
    )

    # ---- 2. Main architectural plan ----------------------------------
    plan = _plan_region(draw)
    _draw_room(draw, hypothesis, plan)
    _draw_openings(draw, hypothesis, plan)

    placed = []
    for index, obj in enumerate(hypothesis.objects, start=1):
        if obj.position is None:
            continue  # unplaced: never fabricate a position
        px, py = plan_to_px(obj.position.x, obj.position.y, plan)
        _draw_furniture(
            draw, obj, px, py, plan
        )
        placed.append((index, obj, px, py))

    _draw_numbered_markers(draw, placed, plan)

    # ---- 3. Right column: object legend (own region) -----------------
    _draw_legend(draw, hypothesis, placed)

    # ---- 4. Footer: single disclaimer --------------------------------
    draw.text(
        (60, FOOTER_Y),
        "Schematic approximation from a single photo — verify all "
        "dimensions by hand before any purchase or build.",
        font=small,
        fill=MUTED,
    )

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Plan coordinate mapping
# ---------------------------------------------------------------------------

def _plan_region(draw: ImageDraw.ImageDraw) -> dict:
    x0, y0, x1, y1 = PLAN_BOX
    draw.rectangle(PLAN_BOX, fill=PAPER, outline=LINE_LIGHT, width=2)
    available_w = x1 - x0 - 2 * PLAN_PAD
    available_h = y1 - y0 - 2 * PLAN_PAD
    # A plan coordinate unit must occupy the same number of pixels on both
    # axes; otherwise rectangular rooms and furniture are stretched.
    inner_w = inner_h = min(available_w, available_h)
    return {
        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
        "inner_x0": x0 + PLAN_PAD + (available_w - inner_w) / 2,
        "inner_y0": y0 + PLAN_PAD + (available_h - inner_h) / 2,
        "inner_w": inner_w, "inner_h": inner_h,
    }


def plan_to_px(x: float, y: float, plan: dict) -> tuple[float, float]:
    return (
        plan["inner_x0"] + x * plan["inner_w"],
        plan["inner_y0"] + y * plan["inner_h"],
    )


def _sym_px(unit: float, span: float) -> float:
    px = unit * span
    return min(max(px, SYM_MIN), SYM_MAX) / 2.0


# ---------------------------------------------------------------------------
# Room and walls
# ---------------------------------------------------------------------------

def _draw_room(
    draw: ImageDraw.ImageDraw,
    hypothesis: SpatialHypothesis,
    plan: dict,
) -> None:
    geometry = hypothesis.room_geometry
    boundary = hypothesis.room_boundary

    if geometry is None or not geometry.perimeter:
        if boundary and boundary.polygon:
            # Fallback: draw the raw boundary polygon without walls.
            pts = [plan_to_px(p.x, p.y, plan) for p in boundary.polygon]
            if len(pts) >= 3:
                draw.polygon(pts, fill=ROOM_TINT)
                draw.polygon(pts, outline=WALL, width=WALL_WIDTH)
        else:
            draw.text(
                (plan["inner_x0"] + 40, plan["inner_y0"] + 40),
                "Room geometry could not be estimated from this photo.",
                font=_font(24),
                fill=MUTED,
            )
        return

    pts = [plan_to_px(p.x, p.y, plan) for p in geometry.perimeter]
    if len(pts) < 3:
        return

    # Room interior fill.
    draw.polygon(pts, fill=ROOM_TINT)
    if geometry.rectangular_assumption:
        min_x = min(point[0] for point in pts)
        min_y = min(point[1] for point in pts)
        draw.text(
            (min_x + 16, min_y - 36),
            "RECTANGULAR APPROXIMATION",
            font=_font(14, bold=True),
            fill=MUTED,
        )

    # Openings projected to pixels, grouped by wall.
    openings_on_wall: dict[str, list[tuple[float, float]]] = {}
    for opening in list(geometry.doors) + list(geometry.windows):
        if opening.position is None or not opening.wall_id:
            continue
        openings_on_wall.setdefault(opening.wall_id, []).append(
            plan_to_px(opening.position.x, opening.position.y, plan)
        )

    # Walls: drawn as sub-segments split at openings so doors and
    # windows appear as genuine gaps in the wall line.
    for wall in geometry.walls:
        p0 = plan_to_px(wall.start.x, wall.start.y, plan)
        p1 = plan_to_px(wall.end.x, wall.end.y, plan)
        cuts = [0.0, 1.0]
        for op in openings_on_wall.get(wall.id, []):
            t = _project_t(op, p0, p1)
            if 0.02 < t < 0.98:
                cuts.append(t)
        cuts.sort()
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a < 0.02:
                continue
            pa = (p0[0] + (p1[0] - p0[0]) * a, p0[1] + (p1[1] - p0[1]) * a)
            pb = (p0[0] + (p1[0] - p0[0]) * b, p0[1] + (p1[1] - p0[1]) * b)
            _draw_wall_segment(draw, pa, pb)

    # Clean corner dots.
    corner_r = 5
    for px, py in pts:
        draw.ellipse(
            (px - corner_r, py - corner_r, px + corner_r, py + corner_r),
            fill=WALL,
        )


def _project_t(p, a, b):
    """Parameter t of point p projected onto segment ab."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    if den < 1e-12:
        return 0.0
    return max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / den))


def _draw_wall_segment(
    draw: ImageDraw.ImageDraw, p0: tuple[float, float], p1: tuple[float, float]
) -> None:
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux  # normal
    half = WALL_WIDTH / 2.0
    quad = [
        (x0 + nx * half, y0 + ny * half),
        (x0 - nx * half, y0 - ny * half),
        (x1 - nx * half, y1 - ny * half),
        (x1 + nx * half, y1 + ny * half),
    ]
    draw.polygon(quad, fill=WALL)
    # Inner edge cue for wall thickness.
    inner = WALL_INNER
    draw.line(
        (x0 + nx * inner, y0 + ny * inner, x1 + nx * inner, y1 + ny * inner),
        fill=(110, 125, 145),
        width=2,
    )
    draw.line(
        (x0 - nx * inner, y0 - ny * inner, x1 - nx * inner, y1 - ny * inner),
        fill=(110, 125, 145),
        width=2,
    )
    # Cap discs at corners so joins stay closed.
    r = half
    draw.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), fill=WALL)
    draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=WALL)


# ---------------------------------------------------------------------------
# Openings: doors and windows
# ---------------------------------------------------------------------------

def _draw_openings(
    draw: ImageDraw.ImageDraw,
    hypothesis: SpatialHypothesis,
    plan: dict,
) -> None:
    geometry = hypothesis.room_geometry
    if geometry is None:
        return
    for opening in list(geometry.doors) + list(geometry.windows):
        if opening.position is None:
            continue
        wall = None
        if opening.wall_id:
            wall = next((w for w in geometry.walls if w.id == opening.wall_id), None)
        if wall is None:
            continue
        px, py = plan_to_px(opening.position.x, opening.position.y, plan)

        p0 = plan_to_px(wall.start.x, wall.start.y, plan)
        p1 = plan_to_px(wall.end.x, wall.end.y, plan)
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-6:
            continue
        ux, uy = dx / length, dy / length
        nx, ny = -uy, ux

        if opening.kind == "door":
            _draw_door(draw, px, py, nx, ny, opening)
        else:
            _draw_window(draw, px, py, nx, ny)


def _draw_door(
    draw, px: float, py: float, nx: float, ny: float, opening: Opening
) -> None:
    # Door leaf: line from the opening into the room (away from the wall
    # normal; plan y is down so the room interior is the -normal side).
    leaf_len = 55.0
    leaf = (px - nx * leaf_len * 0.5, py - ny * leaf_len * 0.5)
    draw.line((px, py, leaf), fill=DOOR_LEAF, width=3)
    # Swing arc at the opening, swept into the room.
    r = 22.0
    start = math.atan2(ny, nx)
    sweep = math.radians(opening.swing_degrees if opening.swing_degrees else 90)
    prev = None
    for i in range(25):
        ang = start - sweep + sweep * 2 * i / 24
        point = (px + r * math.cos(ang), py + r * math.sin(ang))
        if prev is not None:
            draw.line((prev, point), fill=SWING, width=2)
        prev = point
    # Opening mark on the wall centreline.
    draw.line(
        (px - nx * 9, py - ny * 9, px + nx * 9, py + ny * 9),
        fill=WALL,
        width=3,
    )


def _draw_window(draw, px: float, py: float, nx: float, ny: float) -> None:
    # Conventional window mark: thin double line across the wall.
    half_w = 10.0
    draw.line(
        (px - nx * half_w, py - ny * half_w, px + nx * half_w, py + ny * half_w),
        fill=WINDOW,
        width=3,
    )
    draw.line(
        (px - nx * (half_w - 3), py - ny * (half_w - 3),
         px + nx * (half_w - 3), py + ny * (half_w - 3)),
        fill=WINDOW,
        width=1,
    )


# ---------------------------------------------------------------------------
# Furniture symbols (top-down geometric footprints)
# ---------------------------------------------------------------------------

def _rotated_rect(cx, cy, w, h, angle):
    """Rectangle centred at (cx,cy), size w x h, rotated by angle rad."""
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    # Screen coords: y down.
    pts = [
        (-w / 2, -h / 2), (w / 2, -h / 2),
        (w / 2, h / 2), (-w / 2, h / 2),
    ]
    return [
        (cx + (px_ * cos_a - py_ * sin_a), cy + (px_ * sin_a + py_ * cos_a))
        for px_, py_ in pts
    ]


def _local_to_px(cx, cy, x, y, angle):
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return (
        cx + x * cos_a - y * sin_a,
        cy + x * sin_a + y * cos_a,
    )


def _rounded_rect_points(cx, cy, w, h, radius, angle=0.0, steps=5):
    """Approximate a rotated rounded rectangle with a closed polygon."""
    half_w, half_h = w / 2.0, h / 2.0
    r = min(max(radius, 0.0), half_w, half_h)
    if r <= 0.0:
        return _rotated_rect(cx, cy, w, h, angle)

    corners = (
        (half_w - r, -half_h + r, -90, 0),
        (half_w - r, half_h - r, 0, 90),
        (-half_w + r, half_h - r, 90, 180),
        (-half_w + r, -half_h + r, 180, 270),
    )
    points = []
    for corner_x, corner_y, start, end in corners:
        for index in range(steps + 1):
            theta = math.radians(start + (end - start) * index / steps)
            x = corner_x + r * math.cos(theta)
            y = corner_y + r * math.sin(theta)
            points.append(_local_to_px(cx, cy, x, y, angle))
    return points


def _draw_rounded_rect(
    draw, cx, cy, w, h, radius, angle, color, fill, width=3
):
    points = _rounded_rect_points(cx, cy, w, h, radius, angle)
    draw.polygon(points, fill=fill, outline=color, width=width)


def _draw_furniture(
    draw: ImageDraw.ImageDraw,
    obj,
    px: float,
    py: float,
    plan: dict,
) -> None:
    label = (obj.label or obj.type or "").lower()
    color = INK
    category = "generic"
    if "bed" in label:
        category = "bed"
    elif "sofa" in label or "couch" in label:
        category = "sofa"
    elif "chair" in label:
        category = "chair"
    elif "dining table" in label or "table" in label:
        category = "table"
    elif "desk" in label:
        category = "desk"
    elif "wardrobe" in label or "cabinet" in label or "counter" in label:
        category = "cabinet"
    elif "tv" in label:
        category = "tv"
    fill = FURNITURE_FILLS[category]

    fp = obj.footprint or {}
    w = float(fp.get("width", 0.18) or 0.18)
    d = float(fp.get("depth", 0.18) or 0.18)
    half_w = _sym_px(w, plan["inner_w"])
    half_d = _sym_px(d, plan["inner_h"])

    angle = obj.orientation if obj.orientation is not None else 0.0

    if "bed" in label:
        _draw_bed(draw, px, py, half_w, half_d, angle, color, fill)
    elif "sofa" in label or "couch" in label:
        _draw_sofa(draw, px, py, half_w, half_d, angle, color, fill)
    elif "chair" in label:
        _draw_chair(draw, px, py, half_w, half_d, angle, color, fill)
    elif "dining table" in label or "table" in label:
        _draw_table(draw, px, py, half_w, half_d, angle, color, fill)
    elif "desk" in label:
        _draw_desk(draw, px, py, half_w, half_d, angle, color, fill)
    elif "wardrobe" in label or "cabinet" in label or "counter" in label:
        _draw_cabinet(draw, px, py, half_w, half_d, angle, color, fill)
    elif "tv" in label:
        _draw_tv(draw, px, py, half_w, half_d, angle, color, fill)
    else:
        _draw_generic(draw, px, py, half_w, half_d, angle, color, fill)

    status_x, status_y = _local_to_px(px, py, half_w * 0.72, half_d * 0.68, angle)
    status_radius = 4
    draw.ellipse(
        (
            status_x - status_radius,
            status_y - status_radius,
            status_x + status_radius,
            status_y + status_radius,
        ),
        fill=STATUS_COLORS[obj.evidence_status],
        outline=PAPER,
        width=1,
    )


def _draw_bed(draw, px, py, hw, hd, angle, color, fill):
    pts = _rounded_rect_points(px, py, hw * 2, hd * 2, 10, angle)
    draw.polygon(pts, fill=fill, outline=color, width=3)
    headboard_h = max(10, hd * 0.18)
    _draw_rounded_rect(
        draw, *_local_to_px(px, py, 0, -hd + headboard_h / 2, angle),
        hw * 1.86, headboard_h, 4, angle, color, (166, 126, 87), 2,
    )
    _draw_rounded_rect(
        draw, *_local_to_px(px, py, 0, hd * 0.28, angle),
        hw * 1.72, hd * 1.18, 8, angle, (197, 190, 178), LIGHT_DETAIL, 1,
    )
    pillow_w = hw * 0.70
    pillow_h = max(9, hd * 0.20)
    for side in (-1, 1):
        _draw_rounded_rect(
            draw, *_local_to_px(px, py, side * hw * 0.48, -hd * 0.48, angle),
            pillow_w, pillow_h, 5, angle, (173, 165, 153), PAPER, 1,
        )
    blanket_y = hd * 0.16
    draw.line(
        (*_local_to_px(px, py, -hw * 0.86, blanket_y, angle),
         *_local_to_px(px, py, hw * 0.86, blanket_y, angle)),
        fill=(180, 168, 151), width=2,
    )
    for offset in (-0.52, 0, 0.52):
        x = hw * offset
        draw.line(
            (*_local_to_px(px, py, x, hd * 0.40, angle),
             *_local_to_px(px, py, x, hd * 0.82, angle)),
            fill=(216, 208, 195),
            width=1,
        )


def _draw_sofa(draw, px, py, hw, hd, angle, color, fill):
    pts = _rounded_rect_points(px, py, hw * 2, hd * 2, 12, angle)
    draw.polygon(pts, fill=fill, outline=color, width=3)
    arm_w = min(max(hw * 0.20, 7), hw * 0.38)
    back_h = min(max(hd * 0.28, 8), hd * 0.42)
    for side in (-1, 1):
        _draw_rounded_rect(
            draw,
            *_local_to_px(px, py, side * (hw - arm_w / 2), 0, angle),
            arm_w, hd * 1.72, 7, angle, color, (164, 143, 119), 2,
        )
    _draw_rounded_rect(
        draw, *_local_to_px(px, py, 0, -hd + back_h / 2, angle),
        hw * 1.45, back_h, 6, angle, color, (176, 157, 134), 2,
    )
    cushion_y = back_h * 0.35
    cushion_count = 3 if hw >= 48 else 2
    cushion_gap = 3
    cushion_total_w = max(8, hw * 2 - arm_w * 2 - cushion_gap * (cushion_count + 1))
    cushion_w = cushion_total_w / cushion_count
    for seat in range(cushion_count):
        cushion_x = (
            -cushion_total_w / 2
            + cushion_w / 2
            + seat * (cushion_w + cushion_gap)
        )
        _draw_rounded_rect(
            draw,
            *_local_to_px(px, py, cushion_x, cushion_y, angle),
            cushion_w, hd * 0.60, 7, angle, color, (215, 200, 180), 2,
        )
        seam_x = cushion_x + cushion_w * 0.36
        draw.line(
            (*_local_to_px(px, py, seam_x, cushion_y - hd * 0.20, angle),
             *_local_to_px(px, py, seam_x, cushion_y + hd * 0.20, angle)),
            fill=(186, 168, 145),
            width=1,
        )


def _draw_chair(draw, px, py, hw, hd, angle, color, fill):
    back_h = max(7, hd * 0.62)
    back_y = -hd * 0.66
    seat_y = hd * 0.04
    _draw_rounded_rect(
        draw, *_local_to_px(px, py, 0, back_y, angle),
        hw * 1.70, back_h, 7, angle, color, fill, 2,
    )
    _draw_rounded_rect(
        draw, *_local_to_px(px, py, 0, seat_y, angle),
        hw * 1.62, hd * 0.70, 8, angle, color, (224, 211, 188), 2,
    )
    inset_w, inset_h = hw * 1.25, back_h * 0.56
    _draw_rounded_rect(
        draw, *_local_to_px(px, py, 0, back_y, angle),
        inset_w, inset_h, 4, angle,
        (156, 130, 101), (219, 203, 178), 1,
    )
    leg = max(2, min(hw, hd) * 0.05)
    for x in (-hw * 0.72, hw * 0.72):
        for y in (-hd * 0.70, hd * 0.70):
            lx, ly = _local_to_px(px, py, x, y, angle)
            draw.ellipse(
                (lx - leg, ly - leg, lx + leg, ly + leg),
                fill=color,
            )


def _draw_table(draw, px, py, hw, hd, angle, color, fill):
    pts = _rounded_rect_points(px, py, hw * 2, hd * 2, 8, angle)
    draw.polygon(pts, fill=fill, outline=color, width=3)
    inset = max(4, min(hw, hd) * 0.10)
    _draw_rounded_rect(
        draw, px, py, hw * 2 - inset * 2, hd * 2 - inset * 2,
        5, angle, (161, 125, 83), fill, 1,
    )
    for offset in (-0.35, 0.10, 0.48):
        draw.line(
            (*_local_to_px(px, py, -hw * 0.72, hd * offset, angle),
             *_local_to_px(px, py, hw * 0.72, hd * (offset + 0.08), angle)),
            fill=(208, 177, 137),
            width=1,
        )


def _draw_desk(draw, px, py, hw, hd, angle, color, fill):
    pts = _rounded_rect_points(px, py, hw * 2, hd * 2, 7, angle)
    draw.polygon(pts, fill=fill, outline=color, width=3)
    _draw_rounded_rect(
        draw, *_local_to_px(px, py, 0, -hd * 0.45, angle),
        hw * 0.72, max(8, hd * 0.24), 3, angle, color, (169, 132, 91), 1,
    )
    for side in (-1, 1):
        hx, hy = _local_to_px(px, py, side * hw * 0.16, -hd * 0.45, angle)
        draw.ellipse((hx - 1.5, hy - 1.5, hx + 1.5, hy + 1.5), fill=LIGHT_DETAIL)
    draw.line(
        (*_local_to_px(px, py, -hw * 0.42, hd * 0.55, angle),
         *_local_to_px(px, py, hw * 0.42, hd * 0.55, angle)),
        fill=(151, 119, 84), width=2,
    )


def _draw_cabinet(draw, px, py, hw, hd, angle, color, fill):
    pts = _rounded_rect_points(px, py, hw * 2, hd * 2, 6, angle)
    draw.polygon(pts, fill=fill, outline=color, width=3)
    panel_w = hw * 0.78
    panel_h = hd * 1.55
    for side in (-1, 1):
        _draw_rounded_rect(
            draw, *_local_to_px(px, py, side * hw * 0.50, 0, angle),
            panel_w, panel_h, 3, angle, (153, 117, 78), (198, 164, 125), 1,
        )
        hx, hy = _local_to_px(px, py, side * hw * 0.10, 0, angle)
        draw.ellipse((hx - 2, hy - 2, hx + 2, hy + 2), fill=(94, 77, 59))


def _draw_tv(draw, px, py, hw, hd, angle, color, fill):
    _draw_rounded_rect(
        draw, px, py, hw * 2, max(hd * 2, 8), 4, angle, color, fill, 3,
    )
    _draw_rounded_rect(
        draw, px, py, hw * 1.62, max(hd * 1.18, 4), 2, angle,
        (95, 106, 120), (61, 77, 97), 1,
    )
    _draw_rounded_rect(
        draw, *_local_to_px(px, py, 0, hd * 0.78, angle),
        max(8, hw * 0.36), max(3, hd * 0.14), 2, angle,
        color, (89, 98, 109), 1,
    )


def _draw_generic(draw, px, py, hw, hd, angle, color, fill):
    pts = _rounded_rect_points(px, py, hw * 2, hd * 2, 8, angle)
    draw.polygon(pts, fill=fill, outline=color, width=3)
    cx, cy = _local_to_px(px, py, 0, 0, angle)
    radius = max(3, min(hw, hd) * 0.16)
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        outline=color,
        width=2,
    )


# ---------------------------------------------------------------------------
# Markers and legend
# ---------------------------------------------------------------------------

def _draw_numbered_markers(draw, placed, plan):
    """Supplementary numbered markers placed next to furniture."""
    used = []
    inner = (
        plan["inner_x0"], plan["inner_y0"],
        plan["inner_x0"] + plan["inner_w"], plan["inner_y0"] + plan["inner_h"],
    )

    def available(x, y, marker_index):
        if not (inner[0] <= x <= inner[2] and inner[1] <= y <= inner[3]):
            return False
        if any(
            ((x - ux) ** 2 + (y - uy) ** 2) ** 0.5 < MARKER_MIN_DIST
            for ux, uy in used
        ):
            return False
        for other_index, other, ox, oy in placed:
            if other_index == marker_index:
                continue
            other_footprint = other.footprint or {}
            other_radius = max(
                _sym_px(
                    float(other_footprint.get("width", 0.18) or 0.18),
                    plan["inner_w"],
                ),
                _sym_px(
                    float(other_footprint.get("depth", 0.18) or 0.18),
                    plan["inner_h"],
                ),
            )
            if ((x - ox) ** 2 + (y - oy) ** 2) ** 0.5 < other_radius + MARKER_R + 4:
                return False
        return True

    for index, _obj, fpx, fpy in placed:
        best = None
        footprint = _obj.footprint or {}
        half_w = _sym_px(
            float(footprint.get("width", 0.18) or 0.18), plan["inner_w"]
        )
        half_h = _sym_px(
            float(footprint.get("depth", 0.18) or 0.18), plan["inner_h"]
        )
        first_ring = int((max(half_w, half_h) + MARKER_R + 6) / 5.0)
        for ring in range(first_ring, 40):
            radius = ring * 5.0
            if radius == 0:
                candidates = [(fpx, fpy)]
            else:
                candidates = []
                steps = max(8, int(radius))
                for k in range(steps):
                    ang = 2 * math.pi * k / steps
                    candidates.append(
                        (fpx + radius * math.cos(ang), fpy + radius * math.sin(ang))
                    )
            for mpx, mpy in candidates:
                if not available(mpx, mpy, index):
                    continue
                best = (mpx, mpy)
                break
            if best is not None:
                break
        if best is None:
            for ring in range(first_ring - 1, -1, -1):
                radius = ring * 5.0
                steps = max(8, int(radius)) if radius else 1
                for k in range(steps):
                    ang = 2 * math.pi * k / steps if radius else 0
                    candidate = (
                        fpx + radius * math.cos(ang),
                        fpy + radius * math.sin(ang),
                    )
                    if available(candidate[0], candidate[1], index):
                        best = candidate
                        break
                if best is not None:
                    break
        if best is None:
            mpx, mpy = fpx, fpy
        else:
            mpx, mpy = best
        used.append((mpx, mpy))
        draw.ellipse(
            (mpx - MARKER_R, mpy - MARKER_R, mpx + MARKER_R, mpy + MARKER_R),
            fill=(255, 255, 255),
            outline=INK,
            width=2,
        )
        draw.text(
            (mpx - 7, mpy - 9),
            str(index),
            font=_font(16, bold=True),
            fill=INK,
        )


def _chip(draw, x, y, w, h, fill, color, text, font):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=7, fill=fill, outline=color, width=2)
    tw = draw.textlength(text, font=font)
    draw.text((x + (w - tw) / 2, y + (h - 17) / 2), text, font=font, fill=INK)


def _draw_legend(draw, hypothesis, placed):
    x = COL_X
    y = 200
    draw.text((x, y), "Object legend", font=_font(26, bold=True), fill=INK)
    y += 46

    for index, obj, _px, _py in placed:
        color = STATUS_COLORS[obj.evidence_status]
        label = obj.label or obj.type or "object"
        conf = f"  {round(obj.confidence * 100)}%" if obj.confidence is not None else ""
        draw.ellipse((x, y - 5, x + 26, y + 21), fill=color, outline=INK, width=2)
        draw.text((x + 7, y - 3), str(index), font=_font(17, bold=True), fill=INK)
        lw = draw.textlength(label + conf, font=_font(19))
        draw.text((x + 42, y + 1), label + conf, font=_font(19), fill=INK)
        chip_w = 125
        status_text = obj.evidence_status.value.title()
        chip_x = max(x + 46 + lw + 16, COL_X + COL_W - chip_w - 10)
        fillc = STATUS_FILLS[obj.evidence_status]
        draw.rounded_rectangle(
            (chip_x, y + 1, chip_x + chip_w, y + 21), radius=6,
            fill=fillc, outline=color, width=2,
        )
        draw.text((chip_x + 9, y + 3), status_text, font=_font(15), fill=INK)
        y += 42

    unplaced = [o for o in hypothesis.objects if o.position is None]
    if unplaced:
        y += 6
        draw.text((x, y), "Unplaced — position not estimated", font=_font(18, bold=True), fill=MUTED)
        y += 30
        for obj in unplaced:
            label = obj.label or obj.type or "object"
            draw.text((x + 16, y + 1), "• " + label, font=_font(18), fill=MUTED)
            draw.text((x + 16, y + 20),
                      "(" + (obj.uncertainty_reason or "no floor position") + ")",
                      font=_font(14), fill=MUTED)
            y += 36

    y += 14
    draw.text((x, y), "Evidence status", font=_font(20, bold=True), fill=INK)
    y += 36
    key_items = [
        (EvidenceStatus.OBSERVED, "Observed — visible in the photo"),
        (EvidenceStatus.INFERRED, "Inferred — estimated from evidence"),
        (EvidenceStatus.GENERATED, "Generated — from an AI-generated view"),
        (EvidenceStatus.UNCERTAIN, "Uncertain — not enough evidence"),
    ]
    for status, desc in key_items:
        color = STATUS_COLORS[status]
        fillc = STATUS_FILLS[status]
        draw.rounded_rectangle((x, y + 4, x + 22, y + 24), radius=5,
                               fill=fillc, outline=color, width=2)
        draw.text((x + 32, y + 2), desc, font=_font(17), fill=INK)
        draw.text((x + 32, y + 18), status.value.title(), font=_font(14, bold=True), fill=color)
        y += 40
