"""Al-Sharif Origin-Destination matrix builder for elevator traffic simulation.

Based on Al-Sharif, L. "New Concepts in Lift Traffic Analysis: The Inverse
S-P (I-S-P) Method." Building Services Engineering Research and Technology,
2020/2022.

The OD matrix captures three traffic components:
  - Incoming (alpha %): passengers from the entrance floor to other floors.
  - Outgoing (beta %):  passengers from other floors to the entrance floor.
  - Interfloor (gamma %): passengers travelling between non-entrance floors.

Population weights distribute traffic proportionally to floor populations.
"""

import numpy as np

# ── Traffic modes ─────────────────────────────────────────────────────────
# Each entry: (alpha %, beta %, gamma %) — incoming / outgoing / interfloor.
# All three sum to 100 for each mode.

TRAFFIC_MODES: dict[str, tuple[float, float, float]] = {
    "up_peak":    (80, 10, 10),
    "down_peak":  (10, 80, 10),
    "lunch_peak": (35, 35, 30),
    "interfloor": (25, 25, 50),
    "off_peak":   (15, 15, 70),
}

# Arrival rates in passengers per minute for each mode.  Used by the Poisson
# generator to sample inter-arrival times.

ARRIVAL_RATES: dict[str, float] = {
    "up_peak":    4.0,
    "down_peak":  4.0,
    "lunch_peak": 2.5,
    "interfloor": 1.5,
    "off_peak":   0.5,
}


def build_od_matrix(
    n_floors: int = 10,
    entrance_floor: int = 1,
    alpha: float = 80.0,
    beta: float = 10.0,
    gamma: float = 10.0,
    floor_population: np.ndarray | None = None,
) -> np.ndarray:
    """Build a normalised (n_floors+1, n_floors+1) OD matrix.

    Row/col 0 is always 0 (unused — floors are 1-indexed).
    The sum of **all** entries equals 1.0.

    Parameters
    ----------
    n_floors : int
        Total number of floors (1-indexed).
    entrance_floor : int
        Main entrance floor (1-indexed, typically 1).
    alpha : float
        Incoming traffic percentage  (entrance → other floors).
    beta : float
        Outgoing traffic percentage  (other floors → entrance).
    gamma : float
        Interfloor traffic percentage (non-entrance ↔ non-entrance).
    floor_population : np.ndarray | None
        Shape (n_floors+1,).  Index 0 is unused.  Uniform if *None*.

    Returns
    -------
    np.ndarray
        OD probability matrix of shape (n_floors+1, n_floors+1), float32.
    """
    if floor_population is None:
        floor_population = np.ones(n_floors + 1, dtype=np.float32)
        floor_population[0] = 0.0
    else:
        floor_population = np.asarray(floor_population, dtype=np.float32)
        if floor_population.shape[0] != n_floors + 1:
            raise ValueError(
                f"floor_population must have shape ({n_floors + 1},), "
                f"got {floor_population.shape}"
            )

    total = alpha + beta + gamma
    if total <= 0:
        raise ValueError("alpha + beta + gamma must be > 0")
    a = alpha / total
    b = beta / total
    g = gamma / total

    size = n_floors + 1
    od = np.zeros((size, size), dtype=np.float32)
    ent = entrance_floor

    # Non-entrance floors
    other = [f for f in range(1, n_floors + 1) if f != ent]

    # --- Incoming: entrance → other (weighted by destination population) ---
    if a > 0 and other:
        dw = np.array([floor_population[j] for j in other], dtype=np.float32)
        denom = float(dw.sum())
        if denom > 0:
            for j_idx, j in enumerate(other):
                od[ent, j] = a * (float(dw[j_idx]) / denom)

    # --- Outgoing: other → entrance (weighted by source population) --------
    if b > 0 and other:
        sw = np.array([floor_population[i] for i in other], dtype=np.float32)
        denom = float(sw.sum())
        if denom > 0:
            for i_idx, i in enumerate(other):
                od[i, ent] = b * (float(sw[i_idx]) / denom)

    # --- Interfloor: other → other (source × dest population) ---------------
    if g > 0 and len(other) >= 2:
        iw = np.zeros((size, size), dtype=np.float32)
        for i in other:
            for j in other:
                if i != j:
                    iw[i, j] = float(floor_population[i] * floor_population[j])
        denom = float(iw.sum())
        if denom > 0:
            od += g * (iw / denom)

    # Normalise (handles floating-point drift)
    s = float(od.sum())
    if s > 0:
        od /= s

    return od


def sample_passenger(
    od: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[int, int]:
    """Sample one (origin, dest) pair from an OD probability matrix.

    Parameters
    ----------
    od : np.ndarray
        OD matrix, shape (N+1, N+1), 1-indexed, sum = 1.0.
    rng : np.random.Generator | None
        If *None*, uses ``numpy.random.default_rng()``.

    Returns
    -------
    tuple[int, int]
        ``(origin_floor, dest_floor)``, both 1-indexed.
    """
    if rng is None:
        rng = np.random.default_rng()
    flat = od.ravel()
    idx = int(rng.choice(len(flat), p=flat))
    origin = idx // od.shape[1]
    dest = idx % od.shape[1]
    return origin, dest
