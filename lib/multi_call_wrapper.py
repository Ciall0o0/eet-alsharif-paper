"""Multi-call injection wrapper for ElevatorEnv.

Models the real PLC polling behaviour: in one dispatch cycle several hall calls
can arrive together. We inject extra calls into the env's internal event stream
so they surface as multiple sequential `pending_calls` at dt=0, and we add a
small balance penalty for dumping a whole batch onto one car when idle cars exist.

This is a robust rewrite of the fragile BatchDispatchWrapper (which poked
env._obs_buffer at hardcoded 59/69 offsets and never updated the
floors_up_calls/down_calls sets, and mis-detected batches via elapsed==0).
"""
from __future__ import annotations

import numpy as np

from src.env.elevator_env import ElevatorEnv, MAX_FLOOR


class MultiCallInjectionWrapper:
    def __init__(self, env: ElevatorEnv, inject_prob: float = 0.35,
                 max_extra: int = 3, batch_penalty: float = -0.5):
        self.env = env
        self.action_space = env.action_space
        self.num_elevators = env.num_elevators
        self.STATE_DIM = env.STATE_DIM
        self.inject_prob = inject_prob
        self.max_extra = max_extra
        self.batch_penalty = batch_penalty
        self._batch_assign = np.zeros(env.num_elevators, dtype=np.int64)
        self._batch_active = False
        self._rng = np.random.default_rng()

    @property
    def _real_env(self):
        """Unwrap a masked/adapter template to reach the underlying ElevatorEnv."""
        env = self.env
        return getattr(env, "_env", env)

    def reset(self, **kwargs):
        self._batch_assign = np.zeros(self.num_elevators, dtype=np.int64)
        self._batch_active = False
        return self.env.reset(**kwargs)

    @property
    def pending_calls(self):
        return self._real_env.pending_calls

    def _inject_batch(self):
        """Inject 1..max_extra extra hall calls into the env's pending queue via public API."""
        n = int(self._rng.integers(1, self.max_extra + 1))
        pid_base = 100000  # avoid collision with the env's own counter
        for k in range(n):
            sf = int(self._rng.integers(1, MAX_FLOOR + 1))
            direction = 1 if self._rng.random() < 0.6 else -1
            dest = sf + direction
            dest = max(1, min(MAX_FLOOR, dest if dest != sf else sf + direction))
            self._real_env.pending_calls.append({
                "floor": sf, "dest": dest, "direction": direction,
                "passenger_id": pid_base + k, "arrival_time": self._real_env.elapsed,
            })
            if direction == 1:
                self._real_env.floors_up_calls.add(sf)
            else:
                self._real_env.floors_down_calls.add(sf)
        self._batch_active = True
        self._batch_assign = np.zeros(self.num_elevators, dtype=np.int64)

    def step(self, action: int):
        # Before acting, maybe inject a batch of extra calls (only if there is
        # headroom so we don't explode the pending queue).
        if (self._real_env.pending_calls
                and self._rng.random() < self.inject_prob
                and len(self._real_env.pending_calls) <= self.max_extra + 1):
            self._inject_batch()

        obs, reward, done, truncated, info = self.env.step(action)

        # Track assignments inside the current dt=0 batch.
        self._batch_assign[action] += 1
        if self._batch_active and len(self._real_env.pending_calls) > 1 and self._real_env.elapsed == 0.0:
            # Penalize piling all calls onto one car while others sit idle.
            most = int(self._batch_assign.max())
            idle_others = int((most - self._batch_assign > 0).sum())
            if idle_others > 0:
                reward += self.batch_penalty
        if len(self._real_env.pending_calls) <= 1:
            self._batch_active = False
        return obs, reward, done, truncated, info

    def close(self):
        self.env.close()
