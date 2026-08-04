"""Poisson-process traffic generator driven by an Al-Sharif OD matrix.

Produces (N, 10) float32 event arrays compatible with ``ElevatorEnv``::

    [origin, dest, event_time, patience_s, mass_kg, group_size, arrival_jitter, routing_type, floor_delta, 10.0]
"""

from __future__ import annotations

import numpy as np

from .od_matrix import ARRIVAL_RATES, TRAFFIC_MODES, build_od_matrix, sample_passenger
from .passenger_profile import PassengerProfile, RoutingType, sample_profile

# ── Daily schedule ────────────────────────────────────────────────────────
# (start_minute, end_minute, mode_name) — 480 min = 8 h working day.
# Modes are keys in *both* TRAFFIC_MODES and ARRIVAL_RATES.

DEFAULT_DAILY_SCHEDULE: list[tuple[int, int, str]] = [
    (0, 90, "up_peak"),       # 07:00 – 08:30
    (90, 150, "interfloor"),  # 08:30 – 09:30
    (150, 210, "lunch_peak"), # 09:30 – 10:30
    (210, 390, "interfloor"), # 10:30 – 13:30
    (390, 480, "down_peak"),  # 13:30 – 15:00
]

# Full 12 h working day (07:00 - 19:00) including off_peak. Training and
# validation BOTH use this so the model learns real mode transitions
# (up_peak -> interfloor -> lunch -> ... -> off_peak) like a real day.
DAILY_SCHEDULE_12H: list[tuple[int, int, str]] = [
    (0, 90, "up_peak"),       # 07:00 - 08:30
    (90, 150, "interfloor"),  # 08:30 - 09:30
    (150, 210, "lunch_peak"), # 09:30 - 10:30
    (210, 390, "interfloor"), # 10:30 - 13:30
    (390, 480, "down_peak"),  # 13:30 - 15:00
    (480, 600, "off_peak"),   # 15:00 - 17:00  (was missing entirely)
    (600, 720, "interfloor"), # 17:00 - 19:00
]


class TrafficGenerator:
    """Generate passenger event sequences for elevator simulation.

    Spatial distribution uses an Al-Sharif OD matrix; temporal distribution
    uses a Poisson (exponential inter-arrival) process.

    Parameters
    ----------
    n_floors : int
        Total floors (1-indexed, ≤ 10 for ElevatorEnv).
    entrance_floor : int
        Main entrance floor index (1-indexed).
    floor_population : np.ndarray | None
        Shape ``(n_floors+1,)``.  Index 0 unused.  Uniform if *None*.
    schedule : list[tuple[int,int,str]] | None
        Daily schedule.  Defaults to ``DEFAULT_DAILY_SCHEDULE``.
    seed : int | None
        Seed for the internal random generator.
    """

    def __init__(
        self,
        n_floors: int = 10,
        entrance_floor: int = 1,
        floor_population: np.ndarray | None = None,
        schedule: list[tuple[int, int, str]] | None = None,
        arrival_rates: dict[str, float] | None = None,
        max_events: int = 100,
        seed: int | None = None,
    ):
        self.n_floors = n_floors
        self.entrance_floor = entrance_floor
        self.floor_population = floor_population
        self.schedule = schedule or DEFAULT_DAILY_SCHEDULE
        self.arrival_rates = arrival_rates or ARRIVAL_RATES
        self._rng = np.random.default_rng(seed)

        # Pre-compute OD matrices for every mode
        self._od_cache: dict[str, np.ndarray] = {}
        for mode_name, (alpha, beta, gamma) in TRAFFIC_MODES.items():
            self._od_cache[mode_name] = build_od_matrix(
                n_floors=n_floors,
                entrance_floor=entrance_floor,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                floor_population=floor_population,
            )

    # -- public API ---------------------------------------------------------

    def generate_episode(
        self,
        duration_seconds: float = 3600.0,
        max_events: int = 500,
    ) -> np.ndarray:
        """Generate one episode by walking the daily schedule.

        For each time block a Poisson process with the block's arrival rate
        injects passengers, up to *max_events* total.

        Parameters
        ----------
        duration_seconds : float
            Episode horizon in seconds.
        max_events : int
            Hard cap on the number of events.

        Returns
        -------
        np.ndarray
            Event array ``(M, 10)`` float32, ``M ≤ max_events``.
        """
        events: list[tuple[int, int, float, PassengerProfile]] = []

        for block_start, block_end, mode_name in self.schedule:
            if len(events) >= max_events:
                break

            t_start = block_start * 60.0
            t_end = block_end * 60.0
            rate_sec = self.arrival_rates[mode_name] / 60.0  # pax / sec
            od = self._od_cache[mode_name]

            t = t_start
            while t < t_end and len(events) < max_events:
                inter = self._rng.exponential(1.0 / rate_sec) if rate_sec > 0 else float("inf")
                t += inter
                if t >= t_end:
                    break
                origin, dest = sample_passenger(od, self._rng)
                if origin == dest or origin < 1 or dest < 1:
                    continue
                profile = sample_profile(self._rng)
                events.append((origin, dest, t + profile.arrival_jitter, profile))

        return self._pack(events, max_events)

    def generate_episode_multi_segment(
        self,
        n_segments: int,
        seed_shift: int = 0,
        schedule: list[tuple[int, int, str]] | None = None,
        max_events: int = 500,
    ) -> list[np.ndarray]:
        """Generate *n_segments* independent episodes (multi-env training).

        Each segment runs the (optionally overridden) daily schedule with a
        distinct seed. Passing ``schedule=DAILY_SCHEDULE_12H`` trains on the
        full 12 h working day incl. off_peak so train/val/deploy
        distributions agree.

        Parameters
        ----------
        n_segments : int
            Number of parallel environments.
        seed_shift : int
            Offset added to each segment's seed for determinism control.
        schedule : list[tuple[int,int,str]] | None
            Schedule override; defaults to ``self.schedule`` (8 h).
        max_events : int
            Per-episode event cap.

        Returns
        -------
        list[np.ndarray]
            One event array per segment.
        """
        base = int(self._rng.integers(0, 2**31 - 1))
        return [
            self._new_with_seed(base + seed_shift + i,
                                schedule=schedule or self.schedule)
            .generate_episode(max_events=max_events)
            for i in range(n_segments)
        ]

    def generate_validation_set(
        self,
        n_per_mode: int = 5,
        seed: int = 9999,
        arrival_rates: dict[str, float] | None = None,
        max_events: int = 100,
    ) -> dict[str, list[np.ndarray]]:
        """Generate a fixed-seed validation set covering every traffic mode.

        Each mode runs as a single-block 8 h schedule so every episode
        is purely that mode.

        Parameters
        ----------
        n_per_mode : int
            Episodes per mode (default 5).
        seed : int
            Deterministic seed.

        Returns
        -------
        dict[str, list[np.ndarray]]
            ``{mode_name: [events_array, …]}``, five keys total.
        """
        result: dict[str, list[np.ndarray]] = {}
        for mode_name in TRAFFIC_MODES:
            episodes: list[np.ndarray] = []
            for i in range(n_per_mode):
                gen = self._new_with_seed(seed + i, schedule=[(0, 480, mode_name)])
                episodes.append(gen.generate_episode(duration_seconds=900.0, max_events=max_events))
            result[mode_name] = episodes
        return result

    # -- helpers ------------------------------------------------------------

    def _new_with_seed(self, seed: int, **overrides) -> "TrafficGenerator":
        """Return a deep-equivalent copy with a different seed and optional
        kwarg overrides."""
        kwargs = {
            "n_floors": self.n_floors,
            "entrance_floor": self.entrance_floor,
            "floor_population": self.floor_population,
            "schedule": self.schedule,
            "seed": seed,
            "arrival_rates": self.arrival_rates,
        }
        kwargs.update(overrides)
        return TrafficGenerator(**kwargs)

    @staticmethod
    def _pack(
        events: list[tuple[int, int, float, PassengerProfile]],
        max_events: int,
    ) -> np.ndarray:
        """Pack ``(origin, dest, time)`` tuples into a ``(N, 10)`` float32
        array with floor-delta sentinel and column-9 constant."""

        n = min(len(events), max_events)
        arr = np.zeros((n, 10), dtype=np.float32)

        if n == 0:
            return arr

        events_sorted = sorted(events[:n], key=lambda e: e[2])

        prev_time = 0.0
        for idx, (origin, dest, evt_time, profile) in enumerate(events_sorted):
            arr[idx, 0] = float(origin)
            arr[idx, 1] = float(dest)
            arr[idx, 2] = float(evt_time)
            # Profile columns
            arr[idx, 3] = float(profile.patience_seconds)
            arr[idx, 4] = float(profile.mass_kg)
            arr[idx, 5] = float(profile.group_size)
            arr[idx, 6] = float(profile.arrival_jitter)
            arr[idx, 7] = float(RoutingType._member_names_.index(profile.routing_type.name))
            # floor_delta: -1 sentinel for first event; time gap thereafter
            arr[idx, 8] = -1.0 if idx == 0 else float(evt_time - prev_time)
            arr[idx, 9] = 10.0
            prev_time = evt_time

        return arr
