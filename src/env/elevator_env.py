"""Elevator group-control simulation environment (Gymnasium)."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .metrics import compute_episode_metrics

MAX_FLOOR = 10


class RewardNormalizer:
    """Running reward normalizer using Welford's online algorithm."""

    def __init__(self, clip_range: float = 5.0):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-8
        self.clip_range = clip_range

    def update(self, reward: float) -> float:
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.var += (delta * delta2 - self.var) / self.count
        std = max(self.var ** 0.5, 1e-8)
        normalized = (reward - self.mean) / std
        return float(np.clip(normalized, -self.clip_range, self.clip_range))

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict):
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]


@dataclass(slots=True)
class Elevator:
    id: int
    current_floor: float = 1.0
    target_floor: float = 1.0
    direction: int = 0          # -1=down, 0=idle, 1=up
    state: str = "idle"         # idle | moving | doors_open | doors_close
    load_kg: float = 0.0
    max_load: float = 900.0
    door_timer: float = 0.0
    car_calls: set = field(default_factory=set)
    assigned_passengers: list = field(default_factory=list)
    door_open_time: float = 2.0
    door_close_time: float = 2.5

    # Jerk-limited S-curve parameters
    rated_speed: float = 1.5        # m/s
    acceleration: float = 1.0       # m/s²
    jerk: float = 1.5               # m/s³
    floor_height: float = 3.0       # m
    max_floor: int = 10             # building height (floors)
    boarding_time_per_pax: float = 0.5   # s per boarding passenger
    alighting_time_per_pax: float = 0.4  # s per alighting passenger

    # Runtime state for S-curve trips
    trip_start_floor: float = 1.0
    trip_total_time: float = 0.0

    # Feature-ablation switch (propagated from ElevatorEnv config)
    obs_car_calls_dist: bool = True

    @property
    def load_ratio(self) -> float:
        return min(self.load_kg / self.max_load, 1.0) if self.max_load > 0 else 0.0

    @property
    def is_moving(self) -> bool:
        return self.state == "moving"

    @property
    def is_door_open(self) -> bool:
        return self.state == "doors_open"

    def reset(self):
        self.current_floor = 1.0
        self.target_floor = 1.0
        self.direction = 0
        self.state = "idle"
        self.load_kg = 0.0
        self.door_timer = 0.0
        self.car_calls.clear()
        self.assigned_passengers.clear()

    def step(self, dt: float) -> list:
        """Advance elevator state by dt seconds. Returns list of delivered passengers."""
        delivered = []
        if self.state == "moving":
            self._step_moving(dt, delivered)
        elif self.state == "doors_open":
            delivered.extend(self._step_doors_open(dt))
        elif self.state == "doors_close":
            self._step_doors_close(dt)
        return delivered

    def _step_moving(self, dt: float, delivered: list):
        # S-curve kinematics using precomputed trip_total_time
        self.trip_total_time = max(self.trip_total_time, 1e-9)
        frac = min(1.0, dt / self.trip_total_time)
        dist = abs(self.target_floor - self.trip_start_floor)
        self.current_floor += self.direction * frac * dist
        self.current_floor = max(0.5, min(self.max_floor + 0.5, self.current_floor))

        if (self.direction == 1 and self.current_floor >= self.target_floor) or \
           (self.direction == -1 and self.current_floor <= self.target_floor):
            self.current_floor = round(self.target_floor)
            self.direction = 0
            self.state = "doors_open"
            self.door_timer = self.door_open_time
            delivered.extend(self._disembark_passengers())

    def _step_doors_open(self, dt: float) -> list:
        # BUGFIX (2026-08-01): disembark MUST happen while doors are open.
        # Previously disembark only ran in _step_moving at the arrival instant,
        # before boarding, so boarded passengers could never leave the car ->
        # n_alighted stayed > 0 -> dwell reset door_timer forever -> deadlock.
        self.door_timer -= dt
        delivered = self._disembark_passengers()
        n_new_boarded = self._board_passengers()
        if self.door_timer <= 0:
            # Recompute dwell based on passengers that alighted/boarded THIS door
            # cycle (previously counted ALL boarded passengers with pickup==cur,
            # which never changes -> door_timer reset forever).
            dwell = self.compute_dwell_time(n_new_boarded, len(delivered))
            if dwell > self.door_open_time:
                self.door_timer = dwell - self.door_open_time
                return delivered
            if self.car_calls:
                target = self._pick_service_target()
                if abs(target - self.current_floor) < 0.1:
                    # Nearest call is this floor: keep doors open to board/
                    # disembark rather than entering a zero-distance move loop.
                    self.door_timer = max(self.door_timer, self.door_open_time)
                    return delivered
                self.target_floor = float(target)
                self.direction = 1 if target > self.current_floor else -1
                self.trip_start_floor = self.current_floor
                self.trip_total_time = self.travel_time_for_distance(target - self.current_floor)
                self.state = "doors_close"
                self.door_timer = self.door_close_time
            else:
                self.state = "idle"
        return delivered

    def _step_doors_close(self, dt: float):
        self.door_timer -= dt
        if self.door_timer <= 0:
            self.state = "moving"
            self.direction = 1 if self.target_floor > self.current_floor else -1

    def assign_call(self, pickup_floor: int, dest_floor: int, passenger_id: int):
        """Assign a passenger call to this elevator."""
        p = {"id": passenger_id, "pickup": pickup_floor, "dest": dest_floor,
             "arrive_time": None, "boarded": False}
        self.assigned_passengers.append(p)
        self._recompute_calls()  # keep car_calls in sync with the roster
        if self.state == "idle":
            self._select_next_target()

    def _select_next_target(self):
        if not self.car_calls:
            return
        current = self.current_floor
        target = self._pick_service_target()
        self.target_floor = float(target)
        if self.state == "idle":
            if abs(self.target_floor - current) < 0.1:
                self.state = "doors_open"
                self.door_timer = self.door_open_time
            else:
                self.direction = 1 if self.target_floor > current else -1
                # BUGFIX (2026-08-02): init trip params here too — they were
                # stale from the previous trip, so dist=|target-stale_start|
                # was wrong and an idle-wakeup to the stale start floor moved
                # 0 distance forever (12H eval stuck at 18/1338 delivered).
                self.trip_start_floor = current
                self.trip_total_time = self.travel_time_for_distance(target - current)
                self.state = "moving"

    def _pick_service_target(self) -> int:
        """Pick the next car-call floor to serve.

        Same-direction calls are preferred (the car keeps its heading and
        serves floors along the way); otherwise fall back to the nearest
        floor. Guards against the full-set deadlock where the nearest call is
        the current floor (zero-distance move loops forever).
        """
        calls = sorted(self.car_calls)
        current = self.current_floor
        if self.direction > 0:
            ahead = [f for f in calls if f > current + 0.1]
        elif self.direction < 0:
            ahead = [f for f in calls if f < current - 0.1]
        else:
            ahead = []
        if ahead:
            return ahead[0] if self.direction > 0 else ahead[-1]
        return min(calls, key=lambda f: abs(f - current))

    def _recompute_calls(self):
        """Rebuild car_calls as the set of floors this elevator still
        needs to visit: pickup floors of unboarded passengers plus
        dest floors of boarded ones. Deriving it from the live
        roster (instead of a one-shot add/discard set) guarantees a
        boarded passenger's dest can never be silently dropped, which
        is what used to wedge episodes with orphaned passengers."""
        calls = set()
        for p in self.assigned_passengers:
            if not p["boarded"]:
                calls.add(p["pickup"])
            else:
                calls.add(p["dest"])
        self.car_calls = calls

    def _board_passengers(self) -> int:
        """Board passengers at current floor. Returns count newly boarded."""
        current = round(self.current_floor)
        n = 0
        for p in self.assigned_passengers:
            if not p["boarded"] and p["pickup"] == current:
                if self.load_kg + 75.0 > self.max_load:
                    continue  # capacity exceeded: passenger stays waiting
                p["boarded"] = True
                p["arrive_time"] = None  # will be set on delivery
                self.load_kg += 75.0  # average passenger weight
                n += 1
        # After boarding, this passenger's pickup is satisfied and their
        # dest becomes the active outstanding visit floor.
        self._recompute_calls()
        return n

    def _disembark_passengers(self) -> list:
        current = round(self.current_floor)
        delivered = []
        remaining = []
        for p in self.assigned_passengers:
            if p["boarded"] and p["dest"] == current:
                p["arrive_time"] = current
                self.load_kg = max(0, self.load_kg - 75.0)
                delivered.append(p)
            else:
                remaining.append(p)
        self.assigned_passengers = remaining
        self._recompute_calls()  # outstanding floors from remaining roster
        return delivered

    def travel_time_for_distance(self, dist_floors: float) -> float:
        import math
        d = abs(dist_floors) * self.floor_height
        if d < 1e-9:
            return 0.0
        a, j, v = self.acceleration, self.jerk, self.rated_speed
        t_j = a / j
        d_jerk = (j * t_j**3) / 6
        v_after_jerk = 0.5 * a * t_j
        d_reach_v = d_jerk + (v**2 - v_after_jerk**2) / (2 * a) + d_jerk
        if d < 2 * d_reach_v:
            v_peak = math.sqrt(a * d / 2)
            t_a = v_peak / a
            return 2 * t_a + 2 * t_j
        d_cruise = d - 2 * d_reach_v
        t_cruise = d_cruise / v
        t_a = v / a
        return 2 * (t_j + t_a) + t_cruise

    def compute_dwell_time(self, n_boarding: int, n_alighting: int) -> float:
        return (self.door_open_time + n_boarding * self.boarding_time_per_pax
                + n_alighting * self.alighting_time_per_pax)

    def to_vector(self) -> np.ndarray:
        """Encode elevator state as fixed-length vector."""
        extra = self.max_floor if getattr(self, "obs_car_calls_dist", True) else 0
        out = np.empty(self.max_floor + 3 + 8 + extra, dtype=np.float32)
        self._write_to_buffer(out, 0)
        return out

    def _write_to_buffer(self, out: np.ndarray, offset: int,
                         dist_to_oldest: float = 0.0) -> int:
        """Write elevator state into out[offset:] and return new offset."""
        f = int(round(self.current_floor))
        out[offset:offset + self.max_floor] = 0.0
        if 1 <= f <= self.max_floor:
            out[offset + f - 1] = 1.0
        offset += self.max_floor

        out[offset:offset + 3] = 0.0
        out[offset + self.direction + 1] = 1.0
        offset += 3

        # Inline property lookups (hot path: called per-env per-step)
        lr = min(self.load_kg / self.max_load, 1.0)
        out[offset] = lr; offset += 1
        out[offset] = 1.0 if self.state == "moving" else 0.0; offset += 1
        out[offset] = 1.0 if self.state == "doors_open" else 0.0; offset += 1
        out[offset] = 1.0 if self.car_calls else 0.0; offset += 1
        out[offset] = min(len(self.car_calls) / 8.0, 1.0); offset += 1  # workload visibility
        # car_calls floor one-hot: destination info of boarded passengers (observable)
        # (switchable for feature-ablation: obs_car_calls_dist=false -> 92-dim, no dist)
        if self.obs_car_calls_dist:
            out[offset:offset + self.max_floor] = 0.0
            for cf in self.car_calls:
                if 1 <= cf <= self.max_floor:
                    out[offset + cf - 1] = 1.0
            offset += self.max_floor
        max_pax = int(self.max_load / 75.0)
        out[offset] = min(len(self.assigned_passengers) / max(max_pax, 1), 1.0); offset += 1
        out[offset] = self.target_floor / self.max_floor; offset += 1
        out[offset] = dist_to_oldest; offset += 1

        return offset


class ElevatorEnv(gym.Env):
    """Multi-elevator group-control environment driven by .eet event sequences."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = config or {}
        self.num_elevators = cfg.get("num_elevators", 3)
        self.max_floor = int(cfg.get("num_floors", MAX_FLOOR))
        self.zone_mode = str(cfg.get("zone_mode", "height"))
        # Feature ablation switch: car_calls floor one-hot (122-dim) vs without (92-dim)
        self.obs_car_calls_dist = bool(config.get("obs_car_calls_dist", True))
        self.state_dim = self._compute_state_dim(
            self.num_elevators, self.max_floor, self.obs_car_calls_dist)
        self.rated_speed = cfg.get("rated_speed", 1.5)
        self.acceleration = cfg.get("acceleration", 1.0)
        self.jerk = cfg.get("jerk", 1.5)
        self.floor_height = cfg.get("floor_height", 3.0)
        self.boarding_time_per_pax = cfg.get("boarding_time_per_pax", 0.5)
        self.alighting_time_per_pax = cfg.get("alighting_time_per_pax", 0.4)
        self.long_wait_threshold = cfg.get("long_wait_threshold", 30.0)
        self.door_open_time = cfg.get("door_open_time", 1.5)
        self.door_close_time = cfg.get("door_close_time", 2.0)
        self.long_wait_threshold = cfg.get("long_wait_threshold", 60.0)
        self.max_dt = cfg.get("max_dt", 5.0)  # cap inter-event dt to prevent extreme per-step penalties
        self.max_total_time = cfg.get("max_total_time", 3600.0)
        self.max_load_kg = float(cfg.get("max_load_kg", 900.0))
        self.max_episode_steps = int(cfg.get("max_episode_steps", 0))

        self.observation_space = spaces.Box(
            low=-1.0, high=10.0, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space: spaces.Discrete = spaces.Discrete(self.num_elevators)

        # Pre-allocated buffer for observation construction
        self._obs_buffer = np.empty(self.state_dim, dtype=np.float32)

        self.elevators: list[Elevator] = []
        self.traffic_generator = None  # set early for _get_obs()
        self._init_elevators()

        # Episode state
        self.events: np.ndarray | None = None
        self.event_idx: int = 0
        self.elapsed: float = 0.0
        self.passenger_id_counter: int = 0
        self.pending_calls: deque = deque()
        self._last_dest_floor = -1
        self.floors_up_calls: set[int] = set()
        self.floors_down_calls: set[int] = set()
        self.completed_passengers: list[dict] = []

        # Sanity: computed state dim must match actual encoding size
        test_obs = self._get_obs()
        assert test_obs.shape[0] == self.state_dim, \
            f"State dim computed={self.state_dim} != actual={test_obs.shape[0]}"

        # Stats
        self.total_empty_floors: float = 0.0
        self.total_loaded_floors: float = 0.0
        self.start_stop_count: int = 0
        self.elevator_active_time: float = 0.0

        # Reward config
        self.r_passenger = cfg.get("passenger_delivered", 2.0)
        self.r_wait_sec = cfg.get("wait_time_per_sec", -0.05)
        self.r_empty_floor = cfg.get("empty_distance_per_floor", -0.1)
        self.r_start_stop = cfg.get("energy_per_start_stop", -0.05)
        self.r_idle_sec = cfg.get("idle_penalty_per_sec", 0.0)

        # Assignment-level shaping: immediate reward at decision time
        # These fire at dt=0 when the agent assigns a call to an elevator,
        # giving the policy gradient a real signal about assignment quality.
        self.r_proximity = cfg.get("assignment_proximity", -0.05)
        self.r_dir_align = cfg.get("assignment_direction_align", 0.02)
        self.r_load_balance = cfg.get("assignment_load_balance", -0.03)
        self.r_estimated_wait = cfg.get("assignment_estimated_wait", -0.01)
        self.r_correct = cfg.get("assignment_correct", 0.3)  # positive reward for best-choice assignment
        # Global reward scale (e.g. 0.01 for 12h schedules whose raw
        # returns reach -200k; keeps value targets small and stable).
        self.reward_scale = float(cfg.get("reward_scale", 1.0))

        # Traffic generator (optional on-the-fly event generation)
        traffic_cfg = cfg.get("traffic", {})
        self.traffic_generator = None
        self.traffic_config = traffic_cfg
        if traffic_cfg.get("enabled", False):
            from src.traffic.generator import TrafficGenerator
            self.traffic_generator = TrafficGenerator(
                n_floors=traffic_cfg.get("n_floors", self.max_floor),
                entrance_floor=traffic_cfg.get("entrance_floor", 1),
                seed=traffic_cfg.get("traffic_seed", 42),
            )
        
        # Reward normalization (shared across episodes, passed from outside)
        normalize = cfg.get("normalize", False)
        clip_range = cfg.get("clip_range", 5.0)
        if normalize:
            self.reward_normalizer = RewardNormalizer(clip_range=clip_range)
        else:
            self.reward_normalizer = None

        self._event_wait_buffers: dict[int, float] = {}  # passenger_id → arrival_time
        self._passenger_arrival_times: dict[int, float] = {}  # passenger_id → time entered pending_calls

        # Event-level features for observation (computed per injected event)
        self._last_event_time: float | None = None

    def _init_elevators(self):
        self.elevators = [
            Elevator(
                id=i,
                door_open_time=self.door_open_time,
                door_close_time=self.door_close_time,
                rated_speed=self.rated_speed if hasattr(self, "rated_speed") else 1.5,
                acceleration=self.acceleration if hasattr(self, "acceleration") else 1.0,
                jerk=self.jerk if hasattr(self, "jerk") else 1.5,
                floor_height=self.floor_height if hasattr(self, "floor_height") else 3.0,
                boarding_time_per_pax=self.boarding_time_per_pax if hasattr(self, "boarding_time_per_pax") else 0.5,
                alighting_time_per_pax=self.alighting_time_per_pax if hasattr(self, "alighting_time_per_pax") else 0.4,
                max_floor=self.max_floor,
                max_load=self.max_load_kg if hasattr(self, "max_load_kg") else 900.0,
                obs_car_calls_dist=self.obs_car_calls_dist,
            )
            for i in range(self.num_elevators)
        ]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step_count = 0
        self._ep_cap = 0

        for el in self.elevators:
            el.reset()

        options = options or {}
        self.events = options.get("events", None)
        self.event_idx = 0
        self.passenger_id_counter = 0
        self.pending_calls.clear()
        self.floors_up_calls.clear()
        self.floors_down_calls.clear()
        self.completed_passengers.clear()
        self._event_wait_buffers.clear()
        self._last_event_time = None

        self.total_empty_floors = 0.0
        self.total_loaded_floors = 0.0
        self.start_stop_count = 0
        self.elevator_active_time = 0.0

        # Normalize event times: shift to start at 0, scale to max 3600s
        if self.events is None and self.traffic_generator is not None:
            ep_dur = self.traffic_config.get("episode_duration_seconds", 1800)
            max_ev = self.traffic_config.get("max_events_per_episode", 400)
            self.events = self.traffic_generator.generate_episode(
                duration_seconds=ep_dur, max_events=max_ev,
            )

        if self.events is not None and len(self.events) > 0:
            # Copy to avoid mutating original data
            events = np.array(self.events, copy=True)
            raw_times = events[:, 2]
            t_min = float(raw_times.min())
            t_max = float(raw_times.max())
            if t_max - t_min > 1e-6:
                scale = (self.max_total_time * 0.9) / (t_max - t_min)
                events[:, 2] = (raw_times - t_min) * scale
            else:
                events[:, 2] = 0.0
            self.events = events
            self._time_offset = t_min
            self.elapsed = 0.0
            self._advance_to_next_events()
        else:
            self._time_offset = 0.0
            self.elapsed = 0.0

        n_ev = len(self.events) if self.events is not None else 0
        # Step cap for full-day (12 h) schedules: time advance (max_total_time
        # / max_dt) plus per-event service steps, with generous headroom.
        time_steps = int(self.max_total_time / max(self.max_dt, 1.0)) + 2000
        self._ep_cap = max(self.max_episode_steps, n_ev * 15 + time_steps, 2000)
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        """Process one scheduling decision.

        action: elevator index (0..num_elevators-1) to assign to the oldest pending call.
        If no pending calls, action is ignored and time advances to next event.
        """
        self._step_count += 1
        reward = 0.0
        # Diagnostic reward breakdown (entropy-pinning investigation)
        self._r_assign = 0.0
        self._r_deliver = 0.0
        self._r_wait_idle = 0.0

        # --- Assign pending call ---
        if self.pending_calls and 0 <= action < self.num_elevators:
            call = self.pending_calls.popleft()
            self._last_dest_floor = call["dest"]
            elevator = self.elevators[action]
            elevator.assign_call(call["floor"], call["dest"], call["passenger_id"])
            # Wait clock starts from passenger's true arrival, not assignment time
            self._event_wait_buffers[call["passenger_id"]] = call["arrival_time"]
            if call["direction"] == 1:
                self.floors_up_calls.discard(call["floor"])
            else:
                self.floors_down_calls.discard(call["floor"])

            # --- Immediate assignment reward (dt=0 shaping) ---
            # This is the ONLY reward signal at decision time. Without it,
            # the policy gradient is pure noise because dt=0 produces zero reward
            # for all standard components. The magnitudes are calibrated to be
            # comparable to r_passenger (0.3) so the signal isn't drowned out.
            pickup = call["floor"]

            # 1. Proximity: penalize choosing a far-away elevator
            #    Scale: 5 floors away → r_proximity * 5 = -0.25 (comparable to r_passenger=0.3)
            dist_to_pickup = abs(elevator.current_floor - pickup)
            self._r_assign += self.r_proximity * dist_to_pickup
            reward += self.r_proximity * dist_to_pickup

            # 2. Direction alignment: reward choosing an elevator already heading
            #    toward the pickup floor (or idle). Penalize choosing one going away.
            if elevator.direction == 0:
                # Idle elevator: neutral, small bonus for being available
                self._r_assign += self.r_dir_align * 0.5
                reward += self.r_dir_align * 0.5
            elif elevator.direction == call["direction"]:
                # Moving in the same direction as the call: good alignment
                self._r_assign += self.r_dir_align * 1.0
                reward += self.r_dir_align * 1.0
            else:
                # Moving opposite direction: bad, must reverse
                self._r_assign += self.r_dir_align * -1.0
                reward += self.r_dir_align * -1.0

            # 3. Load balancing: penalize assigning to already-busy elevators
            #    when others might be free
            n_assigned = len(elevator.assigned_passengers)
            self._r_assign += self.r_load_balance * n_assigned
            reward += self.r_load_balance * n_assigned

            # 4. Estimated wait: penalize assignments that will take long to serve
            est_wait = self._estimate_pickup_time(call)
            reward += self.r_estimated_wait * est_wait
            # 4b. Positive reward for CORRECT assignment: chosen elevator is the
            #     globally nearest AND already heading the call's direction.
            #     Directly widens the value gap between actions so the policy
            #     can leave the uniform-random plateau (entropy pinned ~0.9-1.0).
            nearest = min(range(self.num_elevators),
                         key=lambda k: abs(self.elevators[k].current_floor - pickup))
            if action == nearest and elevator.direction == call["direction"]:
                self._r_assign += self.r_correct
                reward += self.r_correct

        # --- Determine time delta ---
        if self.pending_calls:
            dt = 0.0  # still have calls to assign, don't advance time
        else:
            dt = min(self._time_to_next_event(), self.max_dt)

        # --- Advance simulation by dt ---
        if dt > 0:
            elevators = self.elevators
            n_el = self.num_elevators

            prev_positions = [0.0] * n_el
            prev_states = [""] * n_el
            for i in range(n_el):
                prev_positions[i] = elevators[i].current_floor
                prev_states[i] = elevators[i].state

            for i in range(n_el):
                delivered = elevators[i].step(dt)
                for p in delivered:
                    wait_time = self.elapsed + dt - self._event_wait_buffers.pop(p["id"], self.elapsed + dt)
                    self.completed_passengers.append({
                        **p,
                        "wait_time": wait_time,
                        "ride_time": elevators[i].travel_time_for_distance(p["dest"] - p["pickup"]),
                    })
                    self._r_deliver += self.r_passenger
                    reward += self.r_passenger

            self.elapsed += dt

            for i in range(n_el):
                el = elevators[i]
                if el.is_moving:
                    dist = abs(el.current_floor - prev_positions[i])
                    if el.load_ratio > 0.05:
                        self.total_loaded_floors += dist
                    else:
                        self.total_empty_floors += dist
                        self._r_wait_idle += self.r_empty_floor * dist
                        reward += self.r_empty_floor * dist
                if prev_states[i] != "moving" and el.state == "moving":
                    self.start_stop_count += 1
                    self._r_wait_idle += self.r_start_stop
                    reward += self.r_start_stop
                if el.state != "idle":
                    self.elevator_active_time += dt

            # Penalize waiting passengers (from true arrival time)
            self._r_wait_idle += self.r_wait_sec * dt * len(self._event_wait_buffers)
            reward += self.r_wait_sec * dt * len(self._event_wait_buffers)

            # Penalize idle elevators
            for el in self.elevators:
                if el.state == "idle":
                    self._r_wait_idle += self.r_idle_sec * dt
            reward += self.r_idle_sec * dt

        # --- Inject new events ---
        if self.events is not None:
            self._advance_to_next_events()

        done = self._is_done()
        info = self._get_info()
        info["reward_breakdown"] = {
            "assign": self._r_assign,
            "deliver": self._r_deliver,
            "wait_idle": self._r_wait_idle,
        }

        if self.reward_normalizer is not None:
            reward = self.reward_normalizer.update(reward)

        if self.reward_scale != 1.0:
            reward = reward * self.reward_scale

        return self._get_obs(), reward, done, False, info

    def _time_to_next_event(self) -> float:
        """Compute dt to advance: next event time or next elevator arrival."""
        candidates = []

        # Next event in sequence
        if self.events is not None and self.event_idx < len(self.events):
            next_et = float(self.events[self.event_idx, 2])
            gap = next_et - self.elapsed
            if gap > 0:
                candidates.append(gap)

        # Next elevator state change
        for el in self.elevators:
            if el.is_moving:
                remaining = el.travel_time_for_distance(el.target_floor - el.current_floor)
                candidates.append(max(0.01, remaining))
            elif el.state == "doors_open":
                candidates.append(el.door_timer + 0.01)
            elif el.state == "doors_close":
                candidates.append(el.door_timer + 0.01)

        return min(candidates) if candidates else 0.5  # fallback

    def _advance_to_next_events(self):
        """Inject pending calls from event sequence based on event_time."""
        if self.events is None:
            return
        while self.event_idx < len(self.events):
            ev = self.events[self.event_idx]
            et = float(ev[2])  # event_time
            if et > self.elapsed + 1e-6:
                break
            src = max(1, min(self.max_floor, int(ev[0])))
            dst = max(1, min(self.max_floor, int(ev[1])))

            # Compute time_delta from consecutive normalized event times
            if self._last_event_time is not None:
                time_delta = et - self._last_event_time
            else:
                time_delta = 0.0
            self._last_event_time = et

            # Read floor_delta from col 8, replace -1 sentinel with 0
            fd_raw = float(ev[8])
            if fd_raw < -0.5:  # -1.0 sentinel for first event
                fd_raw = 0.0

            if src != dst:
                direction = 1 if dst > src else -1
                pid = self.passenger_id_counter
                self.passenger_id_counter += 1
                self.pending_calls.append({
                    "floor": src, "dest": dst, "direction": direction,
                    "passenger_id": pid,
                    "arrival_time": self.elapsed,  # record true arrival time
                    "time_delta": time_delta,
                    "floor_delta": fd_raw,
                })
                if direction == 1:
                    self.floors_up_calls.add(src)
                else:
                    self.floors_down_calls.add(src)
            self.event_idx += 1

    def _estimate_pickup_time(self, call: dict) -> float:
        """Estimate seconds until an elevator reaches the pickup floor."""
        pickup = call["floor"]
        best_time = float("inf")
        for el in self.elevators:
            dist = abs(el.current_floor - pickup)
            travel_time = el.travel_time_for_distance(dist)
            if el.state == "moving":
                remaining = el.travel_time_for_distance(el.target_floor - el.current_floor)
                travel_time += remaining + self.door_open_time + self.door_close_time
            elif el.state in ("doors_open", "doors_close"):
                if dist < 0.1:
                    # Already at pickup floor, just wait for current door op
                    travel_time = el.door_timer
                else:
                    travel_time += el.door_timer + self.door_close_time
            best_time = min(best_time, travel_time)
        return best_time

    @staticmethod
    def _log_normalize(x: float, max_val: float) -> float:
        """Signed log1p normalization, output in approximately [-1, 1]."""
        return float(np.sign(x) * np.log1p(abs(x)) / np.log1p(max_val))

    def _get_obs(self) -> np.ndarray:
        oldest_floor = self.pending_calls[0]["floor"] if self.pending_calls else None

        offset = 0
        for el in self.elevators:
            if oldest_floor is not None:
                dist_oldest = abs(el.current_floor - oldest_floor) / self.max_floor
            else:
                dist_oldest = 0.0
            offset = el._write_to_buffer(self._obs_buffer, offset,
                                         dist_to_oldest=dist_oldest)

        # up calls + down calls one-hot (zeroed together — adjacent blocks)
        self._obs_buffer[offset:offset + self.max_floor * 2] = 0.0
        for f in self.floors_up_calls:
            if 1 <= f <= self.max_floor:
                self._obs_buffer[offset + f - 1] = 1.0
        for f in self.floors_down_calls:
            if 1 <= f <= self.max_floor:
                self._obs_buffer[offset + self.max_floor + f - 1] = 1.0
        offset += self.max_floor * 2

        # A2: destination blocks REMOVED (no-destination model). Hall-call
        # dispatch cannot know a passenger's destination (only revealed on a
        # car call after boarding), so the policy must not train on it.

        # global features

        max_time = max(self.max_total_time, 1.0)
        self._obs_buffer[offset] = min(self.elapsed / max_time, 1.0); offset += 1
        self._obs_buffer[offset] = min(len(self.pending_calls) / 30.0, 1.0); offset += 1

        # Event-level features from oldest pending call (log-scale normalization)
        if self.pending_calls:
            oldest = self.pending_calls[0]
            td = oldest.get("time_delta", 0.0)
            self._obs_buffer[offset] = self._log_normalize(td, 300.0); offset += 1
            fd = oldest.get("floor_delta", 0.0)
            self._obs_buffer[offset] = self._log_normalize(fd, self.max_floor); offset += 1
        else:
            self._obs_buffer[offset] = 0.0; offset += 1
            self._obs_buffer[offset] = 0.0; offset += 1

        # --- New temporal/aggregate features ---

        # Feature: event progress (how far through the event sequence)
        if self.events is None and self.traffic_generator is not None:
            ep_dur = self.traffic_config.get("episode_duration_seconds", 1800)
            max_ev = self.traffic_config.get("max_events_per_episode", 400)
            self.events = self.traffic_generator.generate_episode(
                duration_seconds=ep_dur, max_events=max_ev,
            )

        if self.events is not None and len(self.events) > 0:
            self._obs_buffer[offset] = self.event_idx / len(self.events)
        else:
            self._obs_buffer[offset] = 0.0
        offset += 1

        # Features: aggregate time_delta stats across ALL pending calls
        if self.pending_calls:
            all_td = [c.get("time_delta", 0.0) for c in self.pending_calls]
            mean_td = float(np.mean(all_td))
            max_td = float(np.max(all_td))
            self._obs_buffer[offset] = self._log_normalize(mean_td, 300.0)
            offset += 1
            self._obs_buffer[offset] = self._log_normalize(max_td, 300.0)
            offset += 1
        else:
            self._obs_buffer[offset] = 0.0; offset += 1
            self._obs_buffer[offset] = 0.0; offset += 1

        # Feature: aggregate floor_delta across all pending calls
        if self.pending_calls:
            all_fd = [c.get("floor_delta", 0.0) for c in self.pending_calls]
            mean_fd = float(np.mean(all_fd))
            self._obs_buffer[offset] = self._log_normalize(mean_fd, self.max_floor)
            offset += 1
        else:
            self._obs_buffer[offset] = 0.0; offset += 1

        # Feature: estimated wait time for oldest pending call
        if self.pending_calls:
            est_wait = self._estimate_pickup_time(self.pending_calls[0])
            self._obs_buffer[offset] = min(est_wait / 120.0, 1.0)
            offset += 1
        else:
            self._obs_buffer[offset] = 0.0; offset += 1

        # Safety: zero any unfilled trailing slots (defense against layout mismatch)
        if offset < len(self._obs_buffer):
            self._obs_buffer[offset:] = 0.0
        return self._obs_buffer

    def _get_info(self) -> dict:
        n_total = len(self.events) if self.events is not None else 0
        n_done = len(self.completed_passengers)
        next_zone = -1
        if self.events is not None and self.event_idx + 1 < len(self.events):
            nf = int(self.events[self.event_idx + 1][0])  # next event floor
            from src.zone_map import zone_label
            next_zone = zone_label(nf, mode=getattr(self, "zone_mode", "height"), n_floors=self.max_floor)
            ndf = int(self.events[self.event_idx + 1][1])  # next event DEST floor
            next_dest_zone = zone_label(ndf, mode=getattr(self, "zone_mode", "height"), n_floors=self.max_floor)
        else:
            next_dest_zone = -1
        return {
            "active_dest": self._last_dest_floor,
            "next_event_zone": next_zone,
            "next_event_dest_zone": next_dest_zone,
            "elapsed": self.elapsed,
            "pending_calls": len(self.pending_calls),
            "completed": n_done,
            "total_passengers": n_total,
            "delivered_passengers": n_done,
            "completion_rate": (n_done / n_total) if n_total > 0 else 0.0,
            "total_empty_floors": self.total_empty_floors,
            "total_loaded_floors": self.total_loaded_floors,
            "start_stop_count": self.start_stop_count,
            "elevator_active_time": self.elevator_active_time,
        }

    def _is_done(self) -> bool:
        if self.elapsed >= self.max_total_time:
            return True
        # Hard safety cap: never let a single episode run away (e.g. when the
        # simulation wedges with stuck passengers and dt collapses to the 0.5s
        # fallback). Bounded generously above realistic serve time.
        if self._ep_cap and self._step_count >= self._ep_cap:
            return True
        if self.events is not None and self.event_idx < len(self.events):
            return False
        if self.pending_calls:
            return False
        # No more events and no pending calls. End as soon as nothing is
        # actively moving/transitioning. This terminates idle-stalled episodes
        # immediately instead of looping at dt=0.5 until max_total_time.
        if any(el.is_moving or el.state in ("doors_open", "doors_close")
               for el in self.elevators):
            return False
        return True

    @staticmethod
    def _compute_state_dim(num_elevators: int, max_floor: int,
                            car_calls_dist: bool = True) -> int:
        # No-destination (A2) layout: elevators + up/down calls + global.
        per = max_floor + 3 + 8 + (max_floor if car_calls_dist else 0)
        return per * num_elevators + max_floor * 2 + 2 + 2 + 5

    @property
    def STATE_DIM(self) -> int:
        return self.state_dim

    def get_episode_metrics(self) -> dict:
        return compute_episode_metrics({
            "completed": self.completed_passengers,
            "total_time": self.elapsed,
            "empty_movement_floors": self.total_empty_floors,
            "loaded_movement_floors": self.total_loaded_floors,
            "start_stop_count": self.start_stop_count,
            "elevator_uptime": self.elevator_active_time,
            "elevator_idle_time": max(0, self.elapsed * self.num_elevators - self.elevator_active_time),
            "num_elevators": self.num_elevators,
        })


if __name__ == "__main__":
    # Quick smoke test with random data
    env = ElevatorEnv()
    fake_events = np.array([
        [1, 5, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 10],
        [3, 7, 2.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 10],
        [8, 2, 4.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 10],
        [5, 1, 6.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 10],
        [2, 9, 8.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 10],
    ], dtype=np.float32)

    obs, info = env.reset(options={"events": fake_events})
    print(f"State dim: {len(obs)}, Action space: {env.action_space.n}")
    print(f"Initial obs[:20]: {obs[:20]}")

    total_reward = 0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        if done:
            print(f"Done at step {step+1}, reward={total_reward:.2f}")
            break

    metrics = env.get_episode_metrics()
    print(f"\nEpisode metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
