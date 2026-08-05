"""Passenger profile sampling for elevator traffic simulation.

Each passenger gets a :class:`PassengerProfile` with behavioural and physical
attributes sampled from realistic distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class RoutingType(str, Enum):
    """Passenger routing category — affects boarding / alighting behaviour."""

    STANDARD = "standard"
    WHEELCHAIR = "wheelchair"
    BAGGAGE = "baggage"
    VIP = "vip"
    STAFF = "staff"


# ── Sampling weights ──────────────────────────────────────────────────────
_ROUTING_WEIGHTS: dict[RoutingType, float] = {
    RoutingType.STANDARD: 0.70,
    RoutingType.WHEELCHAIR: 0.05,
    RoutingType.BAGGAGE: 0.10,
    RoutingType.VIP: 0.05,
    RoutingType.STAFF: 0.10,
}

# Lognormal: ln(μ) ≈ 4.55, σ ≈ 0.575  →  ~30–300 s span
_PATIENCE_MU: float = 4.55
_PATIENCE_SIGMA: float = 0.575

# Gamma(shape=16, scale=5)  →  mean=80 kg, std≈20 kg,  ~50–120 kg span
_MASS_SHAPE: float = 16.0
_MASS_SCALE: float = 5.0

# Group-size categorical distribution
_GROUP_SIZE_ITEMS: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_GROUP_SIZE_WEIGHTS: list[float] = [0.156667, 0.13, 0.163333, 0.233333,
                                   0.143333, 0.11, 0.026667, 0.023333,
                                   0.01, 0.003334]  # real-data calibration

# Arrival jitter: small N(0, 2) seconds around scheduled arrival
_ARRIVAL_JITTER_SIGMA: float = 2.0


@dataclass(slots=True)
class PassengerProfile:
    """Physical and behavioural attributes of a single passenger.

    Attributes
    ----------
    patience_seconds : float
        How long (s) the passenger waits before abandoning.  Sampled from
        a lognormal with ~30–300 s span.
    mass_kg : float
        Passenger body mass in kg.  Sampled from a gamma distribution
        spanning roughly 50–120 kg.
    group_size : int
        Number of people travelling together.  1 = 90 %, 2 = 7 %,
        3–5 = 3 % combined.
    routing_type : RoutingType
        Passenger category affecting boarding/alighting behaviour.
    arrival_jitter : float
        Normally-distributed jitter (± s) around the Poisson event time.
    """

    patience_seconds: float
    mass_kg: float
    group_size: int
    routing_type: RoutingType
    arrival_jitter: float


def sample_profile(
    rng: np.random.Generator | None = None,
) -> PassengerProfile:
    """Sample one :class:`PassengerProfile` from the configured distributions.

    Parameters
    ----------
    rng : np.random.Generator | None
        Optional seeded generator.  Falls back to ``default_rng()``.

    Returns
    -------
    PassengerProfile
    """
    if rng is None:
        rng = np.random.default_rng()

    patience = float(rng.lognormal(_PATIENCE_MU, _PATIENCE_SIGMA))
    # Clip to a reasonable floor / ceiling
    patience = max(10.0, min(patience, 600.0))

    mass = float(rng.gamma(_MASS_SHAPE, _MASS_SCALE))
    mass = max(20.0, min(mass, 200.0))

    group_size = int(rng.choice(_GROUP_SIZE_ITEMS, p=_GROUP_SIZE_WEIGHTS))

    routing_labels = list(_ROUTING_WEIGHTS.keys())
    routing_probs = list(_ROUTING_WEIGHTS.values())
    routing = routing_labels[int(rng.choice(len(routing_labels), p=routing_probs))]

    jitter = float(rng.normal(0.0, _ARRIVAL_JITTER_SIGMA))

    return PassengerProfile(
        patience_seconds=patience,
        mass_kg=mass,
        group_size=group_size,
        routing_type=routing,
        arrival_jitter=jitter,
    )
