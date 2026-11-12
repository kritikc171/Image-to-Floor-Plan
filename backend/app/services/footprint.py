"""Approximate per-class footprint priors for top-down symbols.

These are *approximate* class priors (typical furniture footprints in
normalized plan units 0-1), never measured dimensions. The prior width
is scaled by the projected bbox base width when that evidence is
reliable; otherwise the prior is used as-is and every footprint is
labelled approximate.
"""

PRIORS = {
    "dining table": (0.34, 0.34),
    "couch": (0.55, 0.22),
    "sofa": (0.55, 0.22),
    "wardrobe": (0.45, 0.14),
    "cabinet": (0.45, 0.14),
    "counter": (0.50, 0.16),
    "table": (0.30, 0.30),
    "chair": (0.14, 0.14),
    "desk": (0.42, 0.18),
    "bed": (0.42, 0.52),
    "tv": (0.42, 0.06),
}

DEFAULT = (0.18, 0.18)

_BASIS = (
    "approximate class prior scaled by the projected bbox base width; "
    "not a measured dimension"
)


def prior_for(label: str) -> tuple[float, float]:
    """Return the (width, depth) prior for a class, longest key first."""
    text = (label or "").lower()
    for key in sorted(PRIORS, key=len, reverse=True):
        if key in text:
            return PRIORS[key]
    return DEFAULT


def footprint_for(label: str, ground_width: float | None) -> dict:
    """Return an approximate footprint dict for a detected object."""
    pw, pd = prior_for(label)
    if ground_width is not None and ground_width > 0:
        w = pw * min(max(ground_width / pw, 0.6), 1.8)
    else:
        w = pw
    return {
        "width": round(w, 4),
        "depth": round(pd, 4),
        "approximate": True,
        "basis": _BASIS,
    }
