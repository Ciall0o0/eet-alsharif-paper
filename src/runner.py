"""Multi-env parallel runner with single hidden state (UNREAL-style).

Removed all pred_hidden management (dual-GRU is gone).
"""
from __future__ import annotations

import copy
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import torch

_PROJ_LIB = str(Path(__file__).resolve().parents[1] / "lib")
if _PROJ_LIB not in sys.path:
    sys.path.insert(0, _PROJ_LIB)

from lib.multi_call_wrapper import MultiCallInjectionWrapper


def _env_worker(env, conn, worker_id: int) -> None:
    """Run environment updates on a dedicated process."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    while True:
        cmd, data = conn.recv()
        if cmd == "reset":
            obs, info = env.reset(**data)
            conn.send((obs, info))
        elif cmd == "step":
            action = data
            obs, reward, done, truncated, info = env.step(action)
            conn.send((obs, reward, done, truncated, info))
        elif cmd == "close" or cmd is None:
            break


def _floor_to_zone(floor: int, mode: str = "height", n_floors: int = 10) -> int:
    """Map destination floor to zone label (delegates to src.zone_map)."""
    from src.zone_map import zone_label
    return zone_label(floor, mode=mode, n_floors=n_floors)


class MultiEnvRunner:
    """Manages N parallel envs with batched GPU policy inference.

    Supports both LSTM (hidden_is_tuple=True) and GRU (hidden_is_tuple=False).
    Stores one hidden state per env (NO separate pred_hidden).
    """

    def __init__(self, env_template, num_envs: int, device: torch.device,
                 augment: bool = False, seed: int = 12345,
                 inject_prob: float = 0.0,
                 hidden_is_tuple: bool = True):
        self.num_envs = num_envs
        self.device = device
        self.state_dim = env_template.STATE_DIM
        self.num_elevators = getattr(env_template, "num_elevators", 3)
        self.zone_mode = getattr(env_template, "zone_mode", "height")
        self.n_floors = int(getattr(env_template, "max_floor", 10))
        self.augment = augment
        self.hidden_is_tuple = hidden_is_tuple
        self.rng = np.random.default_rng(seed)

        self.envs = [
            MultiCallInjectionWrapper(copy.deepcopy(env_template), inject_prob=inject_prob)
            for _ in range(num_envs)
        ]

        self.obs: list = [None] * num_envs
        self.hidden: list = [None] * num_envs
        self.done: list[bool] = [True] * num_envs
        self.perms: list = [None] * num_envs

        self._active_count = 0
        self._active: list[int] = []

        self._obs_gpu = torch.empty(num_envs, self.state_dim, dtype=torch.float32, device=device)
        self._use_pinned = device.type == 'cuda'
        if self._use_pinned:
            self._obs_pinned = torch.empty(num_envs, self.state_dim, dtype=torch.float32, pin_memory=True)
        else:
            self._obs_pinned = None

        ctx = mp.get_context("fork")
        self.workers = []
        self.conns = []
        for i in range(num_envs):
            parent_conn, child_conn = ctx.Pipe()
            w = ctx.Process(target=_env_worker, args=(self.envs[i], child_conn, i), daemon=True)
            w.start()
            child_conn.close()
            self.workers.append(w)
            self.conns.append(parent_conn)
        self.envs = None

    def reset_env(self, i: int, events_trimmed, policy):
        if self.done[i]:
            self.done[i] = False
            self._active_count += 1
            self._active.append(i)
        self.conns[i].send(("reset", {"options": {"events": events_trimmed}}))
        obs, _ = self.conns[i].recv()
        self.obs[i] = obs
        self.hidden[i] = policy.get_initial_hidden(1, self.device)

    def _extract_dest_label(self, obs, info) -> int:
        d = info.get("active_dest", -1)
        return _floor_to_zone(d, mode=self.zone_mode, n_floors=self.n_floors) if d > 0 else -1

    def step_all(self, policy, buffer=None, deterministic=False) -> tuple[float, int, int]:
        active = self._active
        if not active:
            return 0.0, 0, 0

        if buffer is not None and buffer.is_ready(int(buffer.max_steps * 0.95)):
            still_active = []
            for i in active:
                if buffer.head[i] < (i + 1) * buffer.per_env:
                    still_active.append(i)
                else:
                    self.done[i] = True
            active = still_active
            if not active:
                self._active = []
                return 0.0, 0, 0

        # Build batched obs
        active_first = active[0]
        obs_list = [self.obs[i] for i in active]
        obs_stack = np.stack(obs_list).astype(np.float32, copy=False)
        n_active = len(active)
        if self._use_pinned:
            self._obs_pinned[:n_active].copy_(torch.from_numpy(obs_stack))
            self._obs_gpu[:n_active].copy_(self._obs_pinned[:n_active], non_blocking=True)
        else:
            self._obs_gpu[:n_active].copy_(torch.from_numpy(obs_stack))

        batched_hidden = self._batch_hidden(active)
        obs_seq = self._obs_gpu[:n_active].unsqueeze(1)

        with torch.no_grad():
            if buffer is None:
                actions, _, _, new_hidden, _ = policy.get_action(
                    obs_seq, hidden=batched_hidden, deterministic=deterministic)
            else:
                actions, log_probs, values, new_hidden, _ = policy.get_action(
                    obs_seq, hidden=batched_hidden, deterministic=deterministic)

        self._unbatch_hidden(active, new_hidden)
        action_ints: list[int] = actions.squeeze(-1).tolist()

        total_reward = 0.0
        total_steps = 0
        n_done = 0

        _obs = self.obs
        _done_flags = self.done
        _obs_gpu = self._obs_gpu

        # Dispatch actions to workers
        for j, i in enumerate(active):
            step_action = action_ints[j]
            if self.augment and self.perms[i] is not None:
                step_action = inv_action(self.perms[i], action_ints[j])
            self.conns[i].send(("step", step_action))

        results = {}
        for j, i in enumerate(active):
            results[i] = self.conns[i].recv()

        for j, i in enumerate(active):
            next_obs, reward, done, trunc, info = results[i]
            dest_label = self._extract_dest_label(next_obs, info)

            if buffer is not None:
                ptr_before = buffer.head[i]
                buffer.add_fast(i, _obs_gpu[j], action_ints[j], reward,
                               values[j], log_probs[j], done, dest_label,
                               event_zone=int(info.get("next_event_zone", -1)))
                if buffer.head[i] == ptr_before + 1:
                    if self.hidden_is_tuple:
                        pre_ah = batched_hidden[0][0][:, j:j + 1, :]
                        pre_ac = batched_hidden[0][1][:, j:j + 1, :]
                        pre_ch = batched_hidden[1][0][:, j:j + 1, :]
                        pre_cc = batched_hidden[1][1][:, j:j + 1, :]
                        buffer.set_hidden(i, ((pre_ah, pre_ac), (pre_ch, pre_cc)))
                    else:
                        pre_ah = batched_hidden[0][:, j:j + 1, :]
                        pre_ch = batched_hidden[1][:, j:j + 1, :]
                        buffer.set_hidden(i, ((pre_ah,), (pre_ch,)))

            _obs[i] = next_obs
            total_reward += reward
            total_steps += 1

            if done:
                _done_flags[i] = True
                n_done += 1

        self._active = [i for i in active if not _done_flags[i]]
        self._active_count = len(self._active)
        return total_reward, total_steps, n_done

    def _batch_hidden(self, active: list[int]):
        if self.hidden_is_tuple:
            actor_h = torch.cat([self.hidden[i][0][0] for i in active], dim=1)
            actor_c = torch.cat([self.hidden[i][0][1] for i in active], dim=1)
            critic_h = torch.cat([self.hidden[i][1][0] for i in active], dim=1)
            critic_c = torch.cat([self.hidden[i][1][1] for i in active], dim=1)
            return ((actor_h, actor_c), (critic_h, critic_c))
        else:
            actor_h = torch.cat([self.hidden[i][0] for i in active], dim=1)
            critic_h = torch.cat([self.hidden[i][1] for i in active], dim=1)
            return (actor_h, critic_h)

    def _unbatch_hidden(self, active: list[int], new_hidden):
        if self.hidden_is_tuple:
            (new_ah, new_ac), (new_ch, new_cc) = new_hidden
            for j, i in enumerate(active):
                self.hidden[i] = (
                    (new_ah[:, j:j + 1, :], new_ac[:, j:j + 1, :]),
                    (new_ch[:, j:j + 1, :], new_cc[:, j:j + 1, :]),
                )
        else:
            new_ah, new_ch = new_hidden
            for j, i in enumerate(active):
                self.hidden[i] = (
                    new_ah[:, j:j + 1, :],
                    new_ch[:, j:j + 1, :],
                )

    @property
    def all_done(self) -> bool:
        return self._active_count == 0

    def get_last_obs_per_env(self) -> list:
        return [self.obs[i] if not self.done[i] else None for i in range(self.num_envs)]

    def close(self):
        for i, c in enumerate(self.conns):
            try:
                c.send(("close", None))
            except Exception:
                pass
        for w in self.workers:
            w.join(timeout=2)
