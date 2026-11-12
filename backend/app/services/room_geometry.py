"""Room geometry: connected wall segments, corners, and openings.

Builds a coherent room perimeter from the estimated visible floor
polygon. Each polygon edge becomes one wall segment; polygon vertices
become room corners. This is an *approximation of the visible room*:
walls outside the photograph (behind the camera, occluded) are simply
absent, and the geometry is flagged uncertain rather than fabricated.

Openings (door/window candidates from feature extraction) are projected
into plan space and snapped to the nearest wall segment; the wall side
and an approximate width are recorded. Door/window existence itself is
never confirmed — every opening is UNCERTAIN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from app.schemas.spatial import (
    EvidenceStatus,
    Opening,
    PlanPoint,
    RoomGeometry,
    WallSegment,
)
from app.services.floorplane import FloorPlaneResult


@dataclass(frozen=True)
class WallCandidate:
    start: tuple[float, float]  # plan coords
    end: tuple[float, float]
    length: float


def _segment(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return (dx * dx + dy * dy) ** 0.5, math.degrees(math.atan2(dy, dx)) % 180.0


def build_room_geometry(
    floorplane: FloorPlaneResult,
    openings: list[Opening],
    default_thickness: float = 0.04,
) -> RoomGeometry:
    """Construct room geometry from the estimated floor plane.

    Walls are the edges of the visible floor polygon. Openings are
    snapped to the wall whose centreline passes nearest their projected
    position.
    """
    polygon = floorplane.floor_polygon
    walls: list[WallSegment] = []
    corners: list[PlanPoint] = []
    perimeter: list[PlanPoint] = []
    rectangular_assumption = False
    uncertainty_reason: Optional[str] = None
    status = EvidenceStatus.UNCERTAIN

    if polygon and len(polygon) >= 3:
        pts = [PlanPoint(x=round(p[0], 4), y=round(p[1], 4)) for p in polygon]
        # Close the ring explicitly.
        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        perimeter = pts
        corners = [PlanPoint(x=round(p[0], 4), y=round(p[1], 4)) for p in polygon]

        if len(polygon) == 4:
            rectangular_assumption = _looks_rectangular(polygon)

        n = len(polygon)
        for i in range(n):
            a = polygon[i]
            b = polygon[(i + 1) % n]
            length, _ = _segment(a, b)
            if length < 1e-6:
                continue
            walls.append(
                WallSegment(
                    id=f"wall-{i:03d}",
                    start=PlanPoint(x=round(a[0], 4), y=round(a[1], 4)),
                    end=PlanPoint(x=round(b[0], 4), y=round(b[1], 4)),
                    thickness=default_thickness,
                    wall_type="uncertain",
                    evidence_status=EvidenceStatus.INFERRED,
                    uncertainty_reason=(
                        "Wall segment inferred from the visible floor-wall "
                        "intersection; hidden walls are not represented."
                    ),
                )
            )
        status = EvidenceStatus.INFERRED
        uncertainty_reason = (
            "Perimeter is the projected visible floor region; walls behind "
            "the camera or fully occluded are absent and not fabricated."
        )
    else:
        uncertainty_reason = (
            "No reliable floor plane could be estimated; room geometry "
            "is unavailable."
        )

    # Snap openings onto walls.
    for opening in openings:
        if opening.position is None:
            continue
        target = (opening.position.x, opening.position.y)
        best_wall = None
        best_dist = float("inf")
        for wall in walls:
            d = _point_segment_dist(
                target,
                (wall.start.x, wall.start.y),
                (wall.end.x, wall.end.y),
            )
            if d < best_dist:
                best_dist = d
                best_wall = wall
        if best_wall is None or best_dist > 0.25:
            # Too far from any wall: keep the opening un-snapped.
            continue
        opening.wall_id = best_wall.id
        opening.side = _wall_side(best_wall)
        if opening.width is None:
            w_len, _ = _segment(
                (best_wall.start.x, best_wall.start.y),
                (best_wall.end.x, best_wall.end.y),
            )
            opening.width = round(min(0.10, max(0.02, w_len * 0.12)), 4)
        opening.position = PlanPoint(
            x=round(opening.position.x, 4), y=round(opening.position.y, 4)
        )

    return RoomGeometry(
        perimeter=perimeter,
        corners=corners,
        walls=walls,
        doors=[o for o in openings if o.kind == "door"],
        windows=[o for o in openings if o.kind == "window"],
        evidence_status=status,
        uncertainty_reason=uncertainty_reason,
        rectangular_assumption=rectangular_assumption,
    )


def _looks_rectangular(polygon: list[tuple[float, float]]) -> bool:
    """Weak rectangularity cue from opposite edges (assumption only)."""
    if len(polygon) != 4:
        return False
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = polygon

    def vec(a, b):
        return (b[0] - a[0], b[1] - a[1])

    def norm(u):
        return (u[0] ** 2 + u[1] ** 2) ** 0.5

    e0 = vec((x0, y0), (x1, y1))
    e1 = vec((x1, y1), (x2, y2))
    e2 = vec((x2, y2), (x3, y3))
    e3 = vec((x3, y3), (x0, y0))

    n0, n1, n2, n3 = norm(e0), norm(e1), norm(e2), norm(e3)
    if min(n0, n1, n2, n3) < 1e-6:
        return False
    cross01 = abs(e0[0] * e2[1] - e0[1] * e2[0]) / (n0 * n2)
    cross12 = abs(e1[0] * e3[1] - e1[1] * e3[0]) / (n1 * n3)
    len_ok = abs(n0 - n2) / max(n0, n2) < 0.35 and abs(n1 - n3) / max(n1, n3) < 0.35
    return cross01 < 0.35 and cross12 < 0.35 and len_ok


def _wall_side(wall: WallSegment) -> str:
    """Compass side of a wall segment in plan space (y is down)."""
    dx = wall.end.x - wall.start.x
    dy = wall.end.y - wall.start.y
    ang = math.degrees(math.atan2(dy, dx)) % 180.0
    if ang < 20.0 or ang > 160.0:
        return "south" if dy > 0 else "north"
    return "east" if dx > 0 else "west"


def _point_segment_dist(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    fx, fy = ax + t * dx, ay + t * dy
    return ((px - fx) ** 2 + (py - fy) ** 2) ** 0.5
