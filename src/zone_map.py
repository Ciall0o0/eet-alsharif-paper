"""Zone label mapping for auxiliary destination prediction (shared by train/runner)."""

HEIGHT_10F = [(1, 3), (4, 7), (8, 10)]
HEIGHT_20F = [(1, 7), (8, 13), (14, 20)]
FUNCTIONAL_20F = [(1, 3), (4, 10), (11, 15), (16, 20)]


def zone_label(floor: int, mode: str = "height", n_floors: int = 10) -> int:
    """Map a 1-based destination floor to its zone id.

    - height:      equal-height bands (10F: 1-3/4-7/8-10; 20F: 1-7/8-13/14-20).
    - functional:  mixed-use function bands (Siikonen 2024 EJOR):
                   1-3 retail / 4-10 office / 11-15 hotel / 16-20 apartment.
    """
    if mode == "functional":
        bounds = FUNCTIONAL_20F
    elif n_floors >= 20:
        bounds = HEIGHT_20F
    else:
        bounds = HEIGHT_10F
    for i, (lo, hi) in enumerate(bounds):
        if lo <= floor <= hi:
            return i
    return len(bounds) - 1  # clamp top floor
