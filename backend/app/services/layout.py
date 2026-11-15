import math


def poly_aabb(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def point_in_poly(x, y, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def rect(cx, cy, w, d):
    return (cx - w / 2, cy - d / 2, cx + w / 2, cy + d / 2)


def rect_overlap(a, b):
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    return (ox, oy) if ox > 0.005 and oy > 0.005 else (0.0, 0.0)


def rect_inside_poly(r, poly):
    return all(
        point_in_poly(x, y, poly)
        for x, y in ((r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3]))
    )


def fit_footprint(cx, cy, w, d, poly):
    """Return (cx, cy, w, d, issues).

    Minimal translation into the projected floor polygon:
    scale a footprint larger than 85% of the room span (keeping the
    aspect ratio), clamp the centre into the AABB with a small wall
    clearance, then nudge the centre toward the polygon centroid until
    the full rect is strictly inside.
    """
    issues = []
    minx, miny, maxx, maxy = poly_aabb(poly)
    spanx = max(maxx - minx, 1e-9)
    spany = max(maxy - miny, 1e-9)
    inset = 0.008  # keep footprints clear of the wall line
    if w > 0.85 * spanx or d > 0.85 * spany:
        scale = min(0.85 * spanx / w, 0.85 * spany / d)
        w *= scale
        d *= scale
        issues.append("footprint scaled to fit the projected floor region")
    if maxx - minx > w + 2 * inset:
        cx = min(max(cx, minx + w / 2 + inset), maxx - w / 2 - inset)
    else:
        cx = (minx + maxx) / 2
    if maxy - miny > d + 2 * inset:
        cy = min(max(cy, miny + d / 2 + inset), maxy - d / 2 - inset)
    else:
        cy = (miny + maxy) / 2
    if not rect_inside_poly(rect(cx, cy, w, d), poly):
        gx = sum(p[0] for p in poly) / len(poly)
        gy = sum(p[1] for p in poly) / len(poly)
        for _ in range(40):
            if rect_inside_poly(rect(cx, cy, w, d), poly):
                break
            tx = 0.5 * (gx - cx)
            ty = 0.5 * (gy - cy)
            if abs(tx) < 1e-6 and abs(ty) < 1e-6:
                break
            cx += tx
            cy += ty
        if not rect_inside_poly(rect(cx, cy, w, d), poly):
            issues.append("footprint partly outside projected floor region")
    return cx, cy, w, d, issues


def resolve_overlap(r_new, placed, poly):
    """Make the smallest in-room correction that clears existing footprints."""
    cx = (r_new[0] + r_new[2]) / 2
    cy = (r_new[1] + r_new[3]) / 2
    width = r_new[2] - r_new[0]
    depth = r_new[3] - r_new[1]
    candidates = []
    clearance = 0.005

    for existing in placed:
        ox, oy = rect_overlap(r_new, existing)
        if ox <= 0 and oy <= 0:
            continue
        if ox > 0:
            candidates.extend(
                ((cx + ox + clearance, cy), (cx - ox - clearance, cy))
            )
        if oy > 0:
            candidates.extend(
                ((cx, cy + oy + clearance), (cx, cy - oy - clearance))
            )

    candidates.sort(
        key=lambda point: (
            (point[0] - cx) ** 2 + (point[1] - cy) ** 2,
            point[1],
            point[0],
        )
    )
    for nx, ny in candidates:
        candidate = rect(nx, ny, width, depth)
        if not rect_inside_poly(candidate, poly):
            continue
        if any(
            rect_overlap(candidate, existing)[0] > 0
            and rect_overlap(candidate, existing)[1] > 0
            for existing in placed
        ):
            continue
        return nx, ny, None

    if candidates:
        return cx, cy, "overlaps another object; position approximate"
    return cx, cy, None


def rect_vs_walls(r, walls, radius: float = 0.001):
    """Return wall ids whose centreline is touched/crossed by the rect.

    A footprint is invalid only when it genuinely intersects a wall
    centreline (within ``radius``). Mere adjacency to a perimeter wall
    (e.g. a bed against a wall) is not a conflict.
    """
    issues = []
    rx0, ry0, rx1, ry1 = r
    corners = [(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)]
    for wall in walls:
        ax, ay = wall.start.x, wall.start.y
        bx, by = wall.end.x, wall.end.y
        conflict = False
        # Any rect corner within radius of the segment.
        for cx, cy in corners:
            if _pt_segment_dist(cx, cy, ax, ay, bx, by) <= radius:
                conflict = True
                break
        # Any segment endpoint inside the rect.
        if not conflict:
            if (rx0 <= ax <= rx1 and ry0 <= ay <= ry1) or (
                rx0 <= bx <= rx1 and ry0 <= by <= ry1
            ):
                conflict = True
        # Segment crosses a rect edge.
        if not conflict:
            edges = [
                ((rx0, ry0), (rx1, ry0)),
                ((rx1, ry0), (rx1, ry1)),
                ((rx1, ry1), (rx0, ry1)),
                ((rx0, ry1), (rx0, ry0)),
            ]
            for p, q in edges:
                if _segments_pass_within((ax, ay), (bx, by), p, q, radius):
                    conflict = True
                    break
        if conflict:
            issues.append(wall.id)
    return issues


def _pt_segment_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    fx, fy = ax + t * dx, ay + t * dy
    return ((px - fx) ** 2 + (py - fy) ** 2) ** 0.5


def _segments_pass_within(a, b, c, d, radius):
    """True if segment ab comes within ``radius`` of segment cd."""
    # Check endpoint-to-segment distances both ways.
    if (
        _pt_segment_dist(*a, *c, *d) <= radius
        or _pt_segment_dist(*b, *c, *d) <= radius
        or _pt_segment_dist(*c, *a, *b) <= radius
        or _pt_segment_dist(*d, *a, *b) <= radius
    ):
        return True
    # Proper segment intersection (zero distance crossing).
    return _segments_intersect(a, b, c, d)


def _segments_intersect(a, b, c, d):
    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]

    def sub(p, q):
        return (p[0] - q[0], p[1] - q[1])

    r = sub(b, a)
    s = sub(d, c)
    den = cross(r, s)
    if abs(den) < 1e-12:
        return False
    qp = sub(c, a)
    t = cross(qp, s) / den
    u = cross(qp, r) / den
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


def validate_object_placement(r, poly, walls, placed):
    """Return list of human-readable layout issues for a placement."""
    issues = []
    if not rect_inside_poly(r, poly):
        issues.append("footprint outside the estimated room perimeter")
    for p in placed:
        o = rect_overlap(r, p)
        if o[0] > 0 and o[1] > 0:
            issues.append("footprint overlaps another object")
            break
    return issues
