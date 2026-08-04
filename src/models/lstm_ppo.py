"""PPO trainer with LSTM/GRU encoder, GAE advantage estimation, AMP, KL early stopping, and destination prediction.

UNREAL-style shared encoder: dest_head is attached to the actor_encoder output,
no separate pred_encoder. Single hidden state, single optimizer step.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from .policy import LSTMActorCritic
from .gru_ppo import GRUActorCritic
from ..utils import get_device


class RunningRewardNormalizer:
    """Batch-level reward normalizer using Welford's online algorithm on GPU."""

    def __init__(self, clip_range: float = 5.0, device=None):
        self.device = device or torch.device("cpu")
        self.mean = torch.tensor(0.0, device=self.device)
        self.var = torch.tensor(1.0, device=self.device)
        self.count = 1e-8
        self.clip_range = clip_range

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        if rewards.numel() < 2:
            return rewards
        batch_mean = rewards.mean()
        batch_var = rewards.var(correction=1)
        batch_count = float(rewards.numel())
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        new_var = m2 / total
        self.mean = new_mean
        self.var = new_var
        self.count = total
        std = torch.clamp(self.var.sqrt(), min=1e-8)
        normalized = (rewards - self.mean) / std
        return torch.clamp(normalized, -self.clip_range, self.clip_range)


class RolloutBuffer:
    """Stores trajectories on GPU partitioned by env for correct per-trajectory GAE.

    UNREAL: stores a single hidden state per step (no pred_hidden).
    """

    def __init__(self, max_steps: int, state_dim: int, num_envs: int = 1,
                 rnn_layers: int = 2, model_type: str = "lstm",
                 num_dest_classes: int = 10, device=None):
        self.max_steps = max_steps
        self.num_envs = num_envs
        self.per_env = max_steps // num_envs
        self.policy_rnn_layers = rnn_layers
        self.model_type = model_type
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs = torch.zeros(max_steps, state_dim, device=device)
        self.actions = torch.zeros(max_steps, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(max_steps, device=device)
        self.values = torch.zeros(max_steps, device=device)
        self.log_probs = torch.zeros(max_steps, device=device)
        self.dones = torch.zeros(max_steps, device=device)
        self.dest_labels = torch.full((max_steps,), -1, dtype=torch.int64, device=device)
        self.event_zones = torch.full((max_steps,), -1, dtype=torch.int64, device=device)
        self.head = [i * self.per_env for i in range(num_envs)]
        self._size = 0
        # Per-step hidden states: single (dispatch_hidden,) — no pred_hidden
        self.hiddens = [[] for _ in range(num_envs)]

    def add_fast(self, env_id: int, obs_gpu: torch.Tensor, action: int, reward: float,
                 value: torch.Tensor, log_prob: torch.Tensor, done: bool,
                 dest_label: int = -1, event_zone: int = -1):
        ptr = self.head[env_id]
        end = (env_id + 1) * self.per_env
        if ptr >= end:
            return
        self.obs[ptr] = obs_gpu
        self.actions[ptr] = action
        self.rewards[ptr] = reward
        self.values[ptr] = value
        self.log_probs[ptr] = log_prob
        self.dones[ptr] = float(done)
        self.dest_labels[ptr] = dest_label
        self.event_zones[ptr] = event_zone
        self.head[env_id] = ptr + 1
        self._size += 1

    def set_hidden(self, env_id: int, hidden):
        """Store the PRE-step hidden for the most recently added transition.

        For LSTM: hidden is ((actor_h, actor_c), (critic_h, critic_c)) with each tensor
                  shaped (L, 1, H). Flattened to (4, L, H).
        For GRU:  hidden is ((actor_h,), (critic_h,)) with each tensor shaped (L, 1, H).
                  Flattened to (2, L, H).
        """
        ptr = self.head[env_id] - 1
        if ptr < env_id * self.per_env:
            return
        if self.model_type == "lstm":
            (ah, ac), (ch, cc) = hidden
            flat = torch.stack([
                ah.detach().reshape(self.policy_rnn_layers, -1),
                ac.detach().reshape(self.policy_rnn_layers, -1),
                ch.detach().reshape(self.policy_rnn_layers, -1),
                cc.detach().reshape(self.policy_rnn_layers, -1),
            ])
        else:
            (ah,), (ch,) = hidden
            flat = torch.stack([
                ah.detach().reshape(self.policy_rnn_layers, -1),
                ch.detach().reshape(self.policy_rnn_layers, -1),
            ])
        idx = ptr - env_id * self.per_env
        if len(self.hiddens[env_id]) <= idx:
            self.hiddens[env_id].append(flat)
        else:
            self.hiddens[env_id][idx] = flat

    def get_hidden_slice(self, env_id: int):
        """Return stacked pre-step hiddens for this env's slice.
        LSTM: (4, n, L, H), GRU: (2, n, L, H)
        """
        start, end = self.get_env_slice(env_id)
        n = end - start
        if n == 0 or len(self.hiddens[env_id]) < n:
            return None
        return torch.stack(self.hiddens[env_id][:n], dim=1)

    def clear(self):
        for i in range(self.num_envs):
            self.head[i] = i * self.per_env
            self.hiddens[i] = []
        self._size = 0

    def get_env_slice(self, env_id: int) -> tuple[int, int]:
        start = env_id * self.per_env
        end = self.head[env_id]
        return start, end

    def env_size(self, env_id: int) -> int:
        return self.head[env_id] - env_id * self.per_env

    def size(self) -> int:
        return self._size

    def is_ready(self, min_total: int) -> bool:
        return self._size >= min_total


class UNREALReplayBuffer:
    """Lightweight replay buffer for UNREAL auxiliary tasks.
    
    Stores (obs, reward) tuples. Supports skewed sampling (50/50 zero/nonzero
    reward) for reward prediction and uniform sampling for value replay.
    
    Args:
        capacity: Maximum number of transitions to store
        state_dim: Observation dimension
        device: torch device
    """
    
    def __init__(self, capacity: int = 4096, state_dim: int = 91, device=None):
        self.capacity = capacity
        self.device = device or torch.device("cpu")
        self.obs = torch.zeros(capacity, state_dim, device=self.device)
        self.rewards = torch.zeros(capacity, device=self.device)
        self._size = 0
        self._ptr = 0
        self._nonzero_idx: list[int] = []
        self._zero_idx: list[int] = []
    
    def add(self, obs: torch.Tensor, reward: float):
        """Add one transition. Maintains separate indices for skewed sampling."""
        idx = self._ptr
        
        # Remove old entry from its list before overwriting
        if self._size >= self.capacity:
            old_reward = self.rewards[idx].item()
            if old_reward != 0:
                self._nonzero_idx = [i for i in self._nonzero_idx if i != idx]
            else:
                self._zero_idx = [i for i in self._zero_idx if i != idx]
        
        self.obs[idx] = obs.detach()
        self.rewards[idx] = reward
        
        if reward != 0:
            self._nonzero_idx.append(idx)
        else:
            self._zero_idx.append(idx)
        
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
    
    @property
    def size(self) -> int:
        return self._size
    
    def sample_reward_pred(self, batch_size: int):
        """Skewed sampling: 50% zero-reward, 50% non-zero-reward.
        
        Returns:
            obs: (B, state_dim) — single-frame observations
            labels: (B,) — 0=negative, 1=zero, 2=positive
        """
        if self._size < batch_size:
            return None, None
        
        half = batch_size // 2
        nonzero_n = min(half, len(self._nonzero_idx))
        zero_n = batch_size - nonzero_n
        
        indices = []
        if nonzero_n > 0 and self._nonzero_idx:
            idx_tensor = torch.tensor(self._nonzero_idx, device=self.device)
            chosen = idx_tensor[torch.randint(0, len(idx_tensor), (nonzero_n,), device=self.device)]
            indices.append(chosen)
        if zero_n > 0 and self._zero_idx:
            idx_tensor = torch.tensor(self._zero_idx, device=self.device)
            chosen = idx_tensor[torch.randint(0, len(idx_tensor), (zero_n,), device=self.device)]
            indices.append(chosen)
        
        if not indices:
            return None, None
        
        indices = torch.cat(indices)
        obs = self.obs[indices]  # (B, state_dim)
        rewards = self.rewards[indices]  # (B,)
        
        # Classify: 0=negative, 1=zero, 2=positive
        labels = torch.where(rewards > 0, torch.tensor(2, device=self.device),
                            torch.where(rewards < 0, torch.tensor(0, device=self.device),
                                       torch.tensor(1, device=self.device)))
        return obs, labels.long()
    
    def sample_value_replay(self, batch_size: int):
        """Uniform sampling for value replay.
        
        Returns:
            obs: (B, state_dim)
        """
        if self._size < batch_size:
            return None
        indices = torch.randint(0, self._size, (batch_size,), device=self.device)
        return self.obs[indices]


class PPOTrainer:
    """PPO trainer with UNREAL shared encoder, GAE, AMP, KL early stopping, and optional destination prediction."""

    def __init__(
        self,
        state_dim: int = 73,
        action_dim: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        rollout_steps: int = 2048,
        seq_len: int = 32,
        burn_in_steps: int = 0,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.0,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        gru_dropout: float = 0.0,
        model_type: str = "lstm",
        num_dest_classes: int = 10,
        actor_hidden: int = 64,
        critic_hidden: int = 64,
        weight_decay: float = 0.0,
        num_envs: int = 1,
        compile_policy: bool = False,
        device: str | torch.device = "cuda",
        activation: str = "relu",
        use_amp: bool = False,
        kl_target: float = 0.01,
        kl_early_stop: bool = True,
        normalize_advantage: bool = True,
        normalize_rewards: bool = False,
        actor_dropout: float = 0.0,
        critic_dropout: float = 0.0,
        use_layer_norm: bool = False,
        aux_prediction: bool = False,
        aux_lambda: float = 1.0,
        dest_head_on: bool = True,
        event_head_on: bool = True,
        reward_change_on: bool = True,
        reward_pred_weight: float = 0.5,
        value_replay_weight: float = 0.5,
        replay_capacity: int = 4096,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_type = model_type
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.seq_len = seq_len
        self.burn_in_steps = burn_in_steps
        self.num_envs = num_envs
        self.kl_target = kl_target
        self.kl_early_stop = kl_early_stop
        self.normalize_advantage = normalize_advantage
        self.normalize_rewards = normalize_rewards
        self.use_amp = use_amp
        self.aux_prediction = aux_prediction
        self.dest_head_on = dest_head_on
        self.event_head_on = event_head_on
        self.reward_change_on = reward_change_on
        self.aux_lambda = aux_lambda
        self.reward_pred_weight = reward_pred_weight
        self.value_replay_weight = value_replay_weight
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type == "gru_shared":
            from .gru_ppo import GRUSharedActorCritic
            self.policy = GRUSharedActorCritic(
                state_dim=state_dim, action_dim=action_dim,
                gru_hidden=gru_hidden, gru_layers=gru_layers,
                gru_dropout=gru_dropout,
                actor_hidden=actor_hidden, critic_hidden=critic_hidden,
                activation=activation,
                actor_dropout=actor_dropout, critic_dropout=critic_dropout,
                use_layer_norm=use_layer_norm,
                aux_prediction=aux_prediction,
                num_dest_classes=num_dest_classes,
                dest_head_on=dest_head_on,
                event_head_on=event_head_on,
                reward_change_on=reward_change_on,
            ).to(self.device)
            rnn_layers = gru_layers
        elif model_type == "gru":
            self.policy = GRUActorCritic(
                state_dim=state_dim, action_dim=action_dim,
                gru_hidden=gru_hidden, gru_layers=gru_layers,
                gru_dropout=gru_dropout,
                actor_hidden=actor_hidden, critic_hidden=critic_hidden,
                activation=activation,
                actor_dropout=actor_dropout, critic_dropout=critic_dropout,
                use_layer_norm=use_layer_norm,
                aux_prediction=aux_prediction,
                num_dest_classes=num_dest_classes,
            ).to(self.device)
            rnn_layers = gru_layers
        else:
            self.policy = LSTMActorCritic(
                state_dim=state_dim, action_dim=action_dim,
                lstm_hidden=lstm_hidden, lstm_layers=lstm_layers,
                lstm_dropout=lstm_dropout,
                actor_hidden=actor_hidden, critic_hidden=critic_hidden,
                activation=activation,
                actor_dropout=actor_dropout, critic_dropout=critic_dropout,
                use_layer_norm=use_layer_norm,
                aux_prediction=aux_prediction,
                num_dest_classes=num_dest_classes,
            ).to(self.device)
            rnn_layers = lstm_layers

        if compile_policy and hasattr(torch, 'compile'):
            try:
                self.policy = torch.compile(self.policy)
            except Exception:
                pass

        use_fused = self.device.type == 'cuda'
        # UNREAL: single shared optimizer for ALL params (actor + critic + dest_head)
        actor_params = []
        critic_params = []
        for n, p in self.policy.named_parameters():
            if "actor" in n:
                actor_params.append(p)
            elif "critic" in n:
                critic_params.append(p)
            else:
                actor_params.append(p)
                critic_params.append(p)
        self.actor_optimizer = optim.Adam(actor_params, lr=lr, weight_decay=weight_decay, fused=use_fused)
        self.critic_optimizer = optim.Adam(critic_params, lr=lr, weight_decay=weight_decay, fused=use_fused)
        self.optimizer = self.actor_optimizer

        # init_scale=1024 (was 65536 default): shared-encoder double backward +
        # Huber gradient accumulation overflowed fp16 after env reward fix.
        self.scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, growth_factor=1.5) \
            if (use_amp and self.device.type == "cuda") else None
        self.buffer = RolloutBuffer(rollout_steps, state_dim, num_envs=num_envs,
                                    rnn_layers=rnn_layers, model_type=model_type,
                                    device=self.device)
        self.reward_normalizer = RunningRewardNormalizer(clip_range=5.0, device=self.device)
        self.replay_buffer = UNREALReplayBuffer(
            capacity=replay_capacity, state_dim=state_dim, device=self.device,
        ) if aux_prediction else None
        self.stats: dict = {}

    def set_entropy_coef(self, coef: float):
        self.entropy_coef = coef

    @property
    def hidden_is_tuple(self) -> bool:
        """True for LSTM (hidden = (h,c) tuple), False for GRU (single tensor)."""
        return self.model_type == "lstm"

    @classmethod
    def from_config(cls, state_dim: int, action_dim: int,
                    cfg: dict, device: str | torch.device = "cuda") -> "PPOTrainer":
        ppo_cfg = cfg.get("ppo", {})
        model_cfg = cfg.get("model", {})
        training_cfg = cfg.get("training", {})
        aux_cfg = cfg.get("aux_prediction", {})
        num_envs = training_cfg.get("num_envs", 1)
        rollout_steps = ppo_cfg.get("rollout_steps", 2048)
        if rollout_steps % num_envs != 0:
            rollout_steps = (rollout_steps // num_envs) * num_envs

        model_type = model_cfg.get("type", "lstm")

        return cls(
            state_dim=state_dim, action_dim=action_dim,
            aux_prediction=aux_cfg.get("enabled", False),
            lr=ppo_cfg.get("learning_rate", 3e-4),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_epsilon=ppo_cfg.get("clip_epsilon", 0.2),
            value_loss_coef=ppo_cfg.get("value_loss_coef", 0.5),
            entropy_coef=ppo_cfg.get("entropy_coef_start", 0.05),
            max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
            ppo_epochs=ppo_cfg.get("ppo_epochs", 10),
            batch_size=ppo_cfg.get("batch_size", 64),
            rollout_steps=rollout_steps,
            seq_len=ppo_cfg.get("seq_len", 32),
            burn_in_steps=ppo_cfg.get("burn_in_steps", 0),
            lstm_hidden=model_cfg.get("lstm_hidden", 128),
            lstm_layers=model_cfg.get("lstm_layers", 2),
            lstm_dropout=model_cfg.get("lstm_dropout", 0.0),
            gru_hidden=model_cfg.get("gru_hidden", 128),
            gru_layers=model_cfg.get("gru_layers", 2),
            gru_dropout=model_cfg.get("gru_dropout", 0.0),
            model_type=model_type,
            actor_hidden=model_cfg.get("actor_hidden", 64),
            critic_hidden=model_cfg.get("critic_hidden", 64),
            weight_decay=ppo_cfg.get("weight_decay", 0.0),
            num_envs=num_envs,
            compile_policy=ppo_cfg.get("compile_policy", False),
            device=device,
            activation=model_cfg.get("activation", "relu"),
            use_amp=ppo_cfg.get("use_amp", False),
            kl_target=ppo_cfg.get("kl_target", 0.01),
            kl_early_stop=ppo_cfg.get("kl_early_stop", True),
            normalize_advantage=ppo_cfg.get("normalize_advantage", True),
            normalize_rewards=ppo_cfg.get("normalize_rewards", False),
            actor_dropout=model_cfg.get("actor_dropout", 0.0),
            critic_dropout=model_cfg.get("critic_dropout", 0.0),
            use_layer_norm=model_cfg.get("use_layer_norm", False),
            aux_lambda=aux_cfg.get("lambda", 0.1),
            num_dest_classes=aux_cfg.get("num_classes", 10),
            dest_head_on=aux_cfg.get("dest_head_on", True),
            event_head_on=aux_cfg.get("event_head_on", True),
            reward_change_on=aux_cfg.get("reward_change_on", True),
            reward_pred_weight=aux_cfg.get("reward_pred_lambda", 0.5),
            value_replay_weight=aux_cfg.get("value_replay_lambda", 0.5),
            replay_capacity=aux_cfg.get("replay_capacity", 4096),
        )

    @staticmethod
    @torch.jit.script
    def _compute_gae_jit(rewards: torch.Tensor, values: torch.Tensor,
                         dones: torch.Tensor,
                         gamma: float, gae_lambda: float,
                         last_value: float) -> tuple[torch.Tensor, torch.Tensor]:
        T = rewards.size(0)
        next_values = torch.empty_like(values)
        next_values[:T - 1] = values[1:]
        next_values[T - 1] = last_value
        next_values = next_values * (1.0 - dones)
        deltas = rewards + gamma * next_values - values
        advantages = torch.zeros(T, device=rewards.device)
        gae = 0.0
        discount = gamma * gae_lambda
        for t in range(T - 1, -1, -1):
            gae = deltas[t] + discount * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                    dones: torch.Tensor,
                    last_value: torch.Tensor | float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        lv = float(last_value) if not isinstance(last_value, float) else last_value
        return self._compute_gae_jit(rewards, values, dones,
                                     self.gamma, self.gae_lambda, lv)

    def update(self, last_obs_per_env: list | None = None):
        total_steps = self.buffer.size()
        if total_steps < self.batch_size:
            self.buffer.clear()
            return {}

        all_advantages = []
        all_returns = []
        all_obs_segs = []
        all_act_segs = []
        all_old_lp_segs = []
        all_old_v_segs = []
        all_hid_segs = []
        all_dest_segs = []
        all_event_segs = []

        last_obs_indices = []
        last_obs_list = []
        for i in range(self.num_envs):
            if last_obs_per_env and last_obs_per_env[i] is not None:
                last_obs_indices.append(i)
                last_obs_list.append(last_obs_per_env[i])

        last_values = [torch.tensor(0.0, device=self.device) for _ in range(self.num_envs)]
        if last_obs_list:
            with torch.no_grad():
                obs_stack = np.stack(last_obs_list)
                o_batch = torch.as_tensor(obs_stack, dtype=torch.float32, device=self.device)
                o_batch = o_batch.unsqueeze(1)
                # UNREAL: forward returns 4 values: logits, values, hidden, dest_logits
                if self.model_type == "gru_shared":
                    _, v_batch, _, _, _, _ = self.policy.forward(o_batch)
                else:
                    _, v_batch, _, _ = self.policy.forward(o_batch)
                for j, i in enumerate(last_obs_indices):
                    last_values[i] = v_batch[j].squeeze()

        burn_in = self.burn_in_steps
        total_window = burn_in + self.seq_len

        for i in range(self.num_envs):
            start, end = self.buffer.get_env_slice(i)
            n = end - start
            if n < total_window:
                continue

            raw_rewards = self.buffer.rewards[start:end]
            rewards = self.reward_normalizer.normalize(raw_rewards) if self.normalize_rewards else raw_rewards
            values = self.buffer.values[start:end]
            dones = self.buffer.dones[start:end]

            adv, ret = self.compute_gae(rewards, values, dones,
                                        last_values[i])

            usable = ((n - burn_in) // self.seq_len) * self.seq_len
            n_seg = usable // self.seq_len

            obs_windows = []
            act_windows = []
            old_lp_windows = []
            old_v_windows = []
            adv_windows = []
            ret_windows = []
            hid_windows = []
            dest_windows = []
            event_windows = []
            env_hid = self.buffer.get_hidden_slice(i)
            for s in range(n_seg):
                seg_start = start + s * self.seq_len
                obs_windows.append(self.buffer.obs[seg_start:seg_start + total_window])
                act_start = seg_start + burn_in
                act_windows.append(self.buffer.actions[act_start:act_start + self.seq_len])
                old_lp_windows.append(self.buffer.log_probs[act_start:act_start + self.seq_len])
                local_start = seg_start - start + burn_in
                old_v_windows.append(values[local_start:local_start + self.seq_len])
                adv_windows.append(adv[local_start:local_start + self.seq_len])
                ret_windows.append(ret[local_start:local_start + self.seq_len])
                dest_windows.append(self.buffer.dest_labels[act_start:act_start + self.seq_len])
                event_windows.append(self.buffer.event_zones[act_start:act_start + self.seq_len])
                if env_hid is not None:
                    hid_windows.append(env_hid[:, seg_start - start])
                else:
                    hid_windows.append(None)

            all_obs_segs.append(torch.stack(obs_windows))
            all_act_segs.append(torch.stack(act_windows))
            all_old_lp_segs.append(torch.stack(old_lp_windows))
            all_old_v_segs.append(torch.stack(old_v_windows))
            all_advantages.append(torch.stack(adv_windows))
            all_returns.append(torch.stack(ret_windows))
            all_dest_segs.append(torch.stack(dest_windows))
            all_event_segs.append(torch.stack(event_windows))
            env_hidden_segs = [h for h in hid_windows if h is not None]
            all_hid_segs.append(torch.stack(env_hidden_segs)) if env_hidden_segs else None

        if not all_obs_segs:
            self.buffer.clear()
            return {}

        obs_segs = torch.cat(all_obs_segs)
        actions_segs = torch.cat(all_act_segs)
        old_lp_segs = torch.cat(all_old_lp_segs)
        old_v_segs = torch.cat(all_old_v_segs)
        adv_segs = torch.cat(all_advantages)
        ret_segs = torch.cat(all_returns)
        dest_segs = torch.cat(all_dest_segs)
        event_segs = torch.cat(all_event_segs)
        hidden_segs = torch.cat(all_hid_segs) if all_hid_segs else None

        with torch.no_grad():
            raw_adv_mean = adv_segs.mean().item()
            raw_adv_std = adv_segs.std().item()
            raw_adv_frac_pos = (adv_segs > 0).float().mean().item()

        if self.normalize_advantage:
            adv_std, adv_mean = torch.std_mean(adv_segs, correction=1)
            adv_segs = (adv_segs - adv_mean) / (adv_std + 1e-8)

        with torch.no_grad():
            explained_var = 1.0 - (ret_segs - old_v_segs).var() / (ret_segs.var() + 1e-8)
            value_pred_error = torch.abs(old_v_segs - ret_segs).mean()

        total_policy_loss = torch.tensor(0.0, device=self.device)
        total_value_loss = torch.tensor(0.0, device=self.device)
        total_entropy = torch.tensor(0.0, device=self.device)
        total_clip_frac = torch.tensor(0.0, device=self.device)
        total_actor_grad_norm = torch.tensor(0.0, device=self.device)
        total_critic_grad_norm = torch.tensor(0.0, device=self.device)
        total_dest_loss = torch.tensor(0.0, device=self.device)
        total_event_acc = 0.0
        total_event_n = 0
        n_updates = 0

        n_segments = obs_segs.size(0)
        segments_per_mb = max(1, self.batch_size // self.seq_len)

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n_segments, device=self.device)
            epoch_kl_sum = torch.tensor(0.0, device=self.device)
            epoch_kl_count = torch.tensor(0.0, device=self.device)

            for start in range(0, n_segments, segments_per_mb):
                batch_idx = perm[start:start + segments_per_mb]

                obs_b = obs_segs[batch_idx]
                act_b = actions_segs[batch_idx]
                old_lp_b = old_lp_segs[batch_idx]
                old_v_b = old_v_segs[batch_idx]
                adv_b = adv_segs[batch_idx]
                ret_b = ret_segs[batch_idx]
                dest_b = dest_segs[batch_idx]
                event_b = event_segs[batch_idx] if event_segs is not None else None

                # UNREAL: single hidden state (no pred_hidden)
                window_hidden = None
                if hidden_segs is not None:
                    hid_b = hidden_segs[batch_idx]
                    if self.model_type == "lstm":
                        # (B, 4, L, H) -> (4, L, B, H)
                        hid_b = hid_b.permute(1, 2, 0, 3).contiguous()
                        window_hidden = ((hid_b[0], hid_b[1]), (hid_b[2], hid_b[3]))
                    else:
                        # GRU: (B, 2, L, H) -> (2, L, B, H)
                        hid_b = hid_b.permute(1, 2, 0, 3).contiguous()
                        window_hidden = (hid_b[0], hid_b[1])

                use_amp_step = self.scaler is not None
                with torch.amp.autocast('cuda', enabled=use_amp_step):
                    # UNREAL: forward returns 4 values: logits, values, hidden, dest_logits
                    if self.model_type == "gru_shared":
                        full_logits, full_values, _, full_dest_logits, full_event_logits, full_rc_logits = self.policy.forward(obs_b, hidden=window_hidden)
                    else:
                        full_logits, full_values, _, full_dest_logits = self.policy.forward(obs_b, hidden=window_hidden)
                        full_event_logits = full_rc_logits = None

                    if burn_in > 0:
                        action_logits = full_logits[:, burn_in:]
                        new_v = full_values[:, burn_in:].squeeze(-1)
                        dest_logits = full_dest_logits[:, burn_in:] if full_dest_logits is not None else None
                        event_logits = full_event_logits[:, burn_in:] if full_event_logits is not None else None
                        rc_logits = full_rc_logits[:, burn_in:] if full_rc_logits is not None else None
                    else:
                        action_logits = full_logits
                        new_v = full_values.squeeze(-1)
                        dest_logits = full_dest_logits
                        event_logits = full_event_logits
                        rc_logits = full_rc_logits

                    # Dispatch PPO loss
                    dist = Categorical(logits=action_logits)
                    new_lp = dist.log_prob(act_b)
                    ent = dist.entropy()

                    ratio = torch.exp(new_lp - old_lp_b)
                    surr1 = ratio * adv_b
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_b
                    policy_loss = -torch.min(surr1, surr2).mean()

                    v_clipped = old_v_b + torch.clamp(
                        new_v - old_v_b, -self.clip_epsilon, self.clip_epsilon)
                    vl_unclipped = nn.functional.huber_loss(new_v, ret_b, delta=10.0)
                    vl_clipped = nn.functional.huber_loss(v_clipped, ret_b, delta=10.0)
                    value_loss = torch.max(vl_unclipped, vl_clipped)

                    ent_loss = ent.mean()

                    actor_loss = policy_loss - self.entropy_coef * ent_loss
                    critic_loss = self.value_loss_coef * value_loss

                    # UNREAL: destination prediction loss from shared encoder
                    if self.aux_prediction and dest_logits is not None:
                        valid_mask = (dest_b >= 0) & (dest_b < dest_logits.size(-1))
                        if valid_mask.any():
                            logits_flat = dest_logits.reshape(-1, dest_logits.size(-1))
                            labels_flat = dest_b.reshape(-1)
                            dest_loss_raw = nn.functional.cross_entropy(
                                logits_flat, labels_flat, ignore_index=-1,
                            )
                            dest_loss_val = self.aux_lambda * dest_loss_raw
                        else:
                            dest_loss_val = torch.tensor(0.0, device=self.device)
                        total_dest_loss += dest_loss_val.detach()
                    else:
                        dest_loss_val = torch.tensor(0.0, device=self.device)

                    # Event prediction aux: next-event zone (temporally valid, observable)
                    if self.aux_prediction and event_logits is not None and event_b is not None:
                        ev_mask = (event_b >= 0) & (event_b < 3)
                        if ev_mask.any():
                            ev_logits_flat = event_logits.reshape(-1, event_logits.size(-1))
                            ev_labels_flat = event_b.reshape(-1)
                            ev_loss_raw = nn.functional.cross_entropy(
                                ev_logits_flat, ev_labels_flat, ignore_index=-1,
                            )
                            ev_loss_val = self.aux_lambda * ev_loss_raw
                            with torch.no_grad():
                                ev_pred = ev_logits_flat.argmax(-1)
                                ok = (ev_pred == ev_labels_flat) & (ev_labels_flat >= 0)
                                total_event_acc += ok.float().sum().item()
                                total_event_n += (ev_labels_flat >= 0).sum().item()
                        else:
                            ev_loss_val = torch.tensor(0.0, device=self.device)
                        total_dest_loss += ev_loss_val.detach()
                        dest_loss_val = dest_loss_val + ev_loss_val
                    else:
                        ev_loss_val = torch.tensor(0.0, device=self.device)

                # UNREAL: single backward step — ALL gradients flow through shared encoder
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # NaN guard: detect NaN in loss BEFORE backward
                loss_check = (actor_loss + dest_loss_val)
                if torch.isnan(loss_check) or torch.isinf(loss_check):
                    dbg = {
                        "actor_loss": actor_loss.detach().item() if actor_loss.numel() else float("nan"),
                        "dest_loss": dest_loss_val.detach().item() if dest_loss_val.numel() else float("nan"),
                        "policy_loss": policy_loss.detach().item() if policy_loss.numel() else float("nan"),
                        "adv_min": adv_b.min().item(), "adv_max": adv_b.max().item(),
                        "adv_nan": bool(torch.isnan(adv_b).any()),
                        "ret_nan": bool(torch.isnan(ret_b).any()),
                        "obs_nan": bool(torch.isnan(obs_b).any()),
                        "dest_b_min": dest_b.min().item(), "dest_b_max": dest_b.max().item(),
                        "dest_b_neg": int((dest_b < 0).sum()),
                        "ratio_nan": bool(torch.isnan(ratio).any()),
                    }
                    print(f"\n[NAN IN LOSS] epoch update {n_updates} — skipping | {dbg}", flush=True)
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    continue

                # Combine all losses: PPO + entropy + dest_loss (all from shared encoder)
                if self.scaler is not None:
                    self.scaler.scale(loss_check).backward(retain_graph=True)
                    self.scaler.unscale_(self.actor_optimizer)
                    # NaN grad guard: skip step if any gradient is NaN/Inf
                    actor_params = [p for n, p in self.policy.named_parameters() if "actor" in n]
                    actor_grad_nan = any(
                        p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                        for p in actor_params
                    )
                    if not actor_grad_nan:
                        actor_grad_norm = nn.utils.clip_grad_norm_(actor_params, self.max_grad_norm)
                        self.scaler.step(self.actor_optimizer)
                        total_actor_grad_norm += actor_grad_norm
                    else:
                        print(f"\n[NAN IN ACTOR GRAD] epoch update {n_updates} — skipping", flush=True)
                        self.actor_optimizer.zero_grad()

                    self.scaler.scale(critic_loss).backward(retain_graph=True)
                    self.scaler.unscale_(self.critic_optimizer)
                    critic_params = [p for n, p in self.policy.named_parameters() if "critic" in n]
                    critic_grad_nan = any(
                        p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                        for p in critic_params
                    )
                    if not critic_grad_nan:
                        critic_grad_norm = nn.utils.clip_grad_norm_(critic_params, self.max_grad_norm)
                        self.scaler.step(self.critic_optimizer)
                        total_critic_grad_norm += critic_grad_norm
                    else:
                        dbg = {
                            "critic_loss": critic_loss.detach().item() if critic_loss.numel() else float("nan"),
                            "new_v_nan": bool(torch.isnan(new_v).any()),
                            "new_v_min": new_v.min().item(), "new_v_max": new_v.max().item(),
                            "ret_nan": bool(torch.isnan(ret_b).any()),
                            "ret_min": ret_b.min().item(), "ret_max": ret_b.max().item(),
                            "ret_inf": bool(torch.isinf(ret_b).any()),
                            "value_grad_nan": sum(1 for p in critic_params if p.grad is not None and torch.isnan(p.grad).any()),
                            "value_grad_inf": sum(1 for p in critic_params if p.grad is not None and torch.isinf(p.grad).any()),
                            "scale": self.scaler.get_scale() if self.scaler is not None else None,
                        }
                        print(f"\n[NAN IN CRITIC GRAD] epoch update {n_updates} — skipping | {dbg}", flush=True)
                        self.critic_optimizer.zero_grad()
                else:
                    (actor_loss + dest_loss_val).backward(retain_graph=True)
                    actor_params = [p for n, p in self.policy.named_parameters() if "actor" in n]
                    actor_grad_nan = any(
                        p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                        for p in actor_params
                    )
                    if not actor_grad_nan:
                        actor_grad_norm = nn.utils.clip_grad_norm_(actor_params, self.max_grad_norm)
                        self.actor_optimizer.step()
                        total_actor_grad_norm += actor_grad_norm
                    else:
                        print(f"\n[NAN IN ACTOR GRAD] epoch update {n_updates} — skipping", flush=True)
                        self.actor_optimizer.zero_grad()

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward(retain_graph=True)
                    critic_params = [p for n, p in self.policy.named_parameters() if "critic" in n]
                    critic_grad_nan = any(
                        p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                        for p in critic_params
                    )
                    if not critic_grad_nan:
                        critic_grad_norm = nn.utils.clip_grad_norm_(critic_params, self.max_grad_norm)
                        self.critic_optimizer.step()
                        total_critic_grad_norm += critic_grad_norm
                    else:
                        print(f"\n[NAN IN CRITIC GRAD] epoch update {n_updates} — skipping", flush=True)
                        self.critic_optimizer.zero_grad()

                if self.scaler is not None:
                    self.scaler.update()
                    if actor_grad_nan or critic_grad_nan:
                        # NaN detected: halve AMP post-update to break overflow cycle
                        scale = self.scaler.get_scale()
                        self.scaler._scale = torch.full_like(
                            self.scaler._scale, max(scale * 0.5, 1.0))

                total_policy_loss += policy_loss.detach()
                total_value_loss += value_loss.detach()
                total_entropy += ent_loss.detach()
                n_updates += 1

                # NaN guard: abort epoch if any parameter became NaN
                any_nan = False
                for name, p in self.policy.named_parameters():
                    if torch.isnan(p).any():
                        any_nan = True
                        break
                if any_nan:
                    print(f"\n[NAN DETECTED] epoch update {n_updates} — skipping remaining updates", flush=True)
                    break

                with torch.no_grad():
                    clip_frac_val = ((ratio.detach() - 1.0).abs() > self.clip_epsilon).float()
                    total_clip_frac += clip_frac_val.mean()

                epoch_kl_sum += (old_lp_b - new_lp.detach()).sum()
                epoch_kl_count += old_lp_b.numel()

            if epoch_kl_count > 0:
                approx_kl = epoch_kl_sum / epoch_kl_count
                if approx_kl > self.kl_target * 1.5:
                    break

        # ── UNREAL Auxiliary Tasks ──────────────────────────────────
        # Only run if model is healthy (no NaN in parameters)
        any_nan = any(torch.isnan(p).any() for p in self.policy.parameters())
        if not any_nan:
            # Reward Prediction: predict next-step reward class from replay
            if (self.aux_prediction and self.reward_pred_weight > 0
                    and self.replay_buffer is not None and self.replay_buffer.size >= 32):
                rp_obs, rp_labels = self.replay_buffer.sample_reward_pred(
                    batch_size=min(64, self.replay_buffer.size))
                if rp_obs is not None:
                    with torch.amp.autocast('cuda', enabled=use_amp_step):
                        rp_in = rp_obs.unsqueeze(1)  # (B, 1, state_dim)
                        rp_gru_out, _ = self.policy.actor_encoder(rp_in, 
                            self.policy._init_hidden(rp_obs.size(0), self.device))
                        rp_logits = self.policy.reward_pred_head(rp_gru_out[:, -1, :])  # (B, 3)
                        rp_loss = nn.functional.cross_entropy(rp_logits, rp_labels)
                    
                    self.actor_optimizer.zero_grad()
                    rp_total = self.reward_pred_weight * rp_loss
                    rp_total.backward()
                    actor_params_rp = [p for n, p in self.policy.named_parameters() if "actor" in n]
                    rp_grad_nan = any(
                        p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                        for p in actor_params_rp
                    )
                    if not rp_grad_nan:
                        nn.utils.clip_grad_norm_(actor_params_rp, self.max_grad_norm)
                        self.actor_optimizer.step()
                    else:
                        print('[NAN IN RP GRAD] skipping reward_pred step', flush=True)
                        self.actor_optimizer.zero_grad()
                    if not hasattr(self, '_unreal_stats'): self._unreal_stats = {}
                    self._unreal_stats["reward_pred_loss"] = rp_loss.detach().item()
                    self._unreal_stats["reward_pred_acc"] = (rp_logits.argmax(-1) == rp_labels).float().mean().item()

            # Reward-change-direction aux: sign(next_reward - reward) from shared rep
            if (self.model_type == "gru_shared" and self.aux_prediction
                    and getattr(self.policy, "reward_change_on", False)
                    and self.reward_pred_weight > 0 and self.buffer._size >= 64):
                with torch.no_grad():
                    rew = self.buffer.rewards[:self.buffer._size]
                    diff = rew[1:] - rew[:-1]
                    eps = 1e-9
                    rc_lab = torch.where(diff > eps, torch.tensor(2, device=self.device),
                                         torch.where(diff < -eps, torch.tensor(0, device=self.device),
                                                     torch.tensor(1, device=self.device)))
                    # keep one hidden-state step: use last seq of obs
                    obs_slice = self.buffer.obs[:self.buffer._size]
                if rc_lab.numel() >= 32:
                    rc_in = obs_slice[:-1].unsqueeze(1)
                    with torch.amp.autocast('cuda', enabled=use_amp_step):
                        rc_out, _ = self.policy.encoder(rc_in, self.policy._init_hidden(rc_in.size(0), self.device))
                        rc_logits = self.policy.reward_change_head(rc_out[:, -1, :])
                        rc_loss = nn.functional.cross_entropy(rc_logits, rc_lab)
                    self.actor_optimizer.zero_grad()
                    rc_total = self.reward_pred_weight * rc_loss
                    rc_total.backward()
                    rc_params = [p for n, p in self.policy.named_parameters() if "reward_change" in n]
                    rc_nan = any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()) for p in rc_params)
                    if not rc_nan:
                        nn.utils.clip_grad_norm_(rc_params, self.max_grad_norm)
                        self.actor_optimizer.step()
                    else:
                        self.actor_optimizer.zero_grad()
                    if not hasattr(self, '_unreal_stats'): self._unreal_stats = {}
                    self._unreal_stats["reward_change_loss"] = rc_loss.detach().item()
                    self._unreal_stats["reward_change_acc"] = (rc_logits.argmax(-1) == rc_lab).float().mean().item()

            # Value Replay: extra value regression on random buffer samples
            if (self.aux_prediction and self.value_replay_weight > 0
                    and self.replay_buffer is not None and self.replay_buffer.size >= 32):
                vr_obs = self.replay_buffer.sample_value_replay(
                    batch_size=min(64, self.replay_buffer.size))
                if vr_obs is not None:
                    vr_in = vr_obs.unsqueeze(1)  # (B, 1, state_dim)
                    with torch.no_grad():
                        vr_gru_out, _ = self.policy.critic_encoder(vr_in,
                            self.policy._init_hidden(vr_obs.size(0), self.device))
                        vr_target = self.policy.critic(vr_gru_out[:, -1, :]).squeeze(-1)
                        vr_target = torch.clamp(vr_target, -100.0, 100.0)
                    # Forward again with grad
                    vr_gru_out_grad, _ = self.policy.critic_encoder(vr_in,
                        self.policy._init_hidden(vr_obs.size(0), self.device))
                    vr_pred = self.policy.critic(vr_gru_out_grad[:, -1, :]).squeeze(-1)
                    vr_pred = torch.clamp(vr_pred, -100.0, 100.0)
                    vr_loss = nn.functional.mse_loss(vr_pred, vr_target)
                    
                    self.critic_optimizer.zero_grad()
                    vr_total = self.value_replay_weight * vr_loss
                    vr_total.backward()
                    critic_params_vr = [p for n, p in self.policy.named_parameters() if "critic" in n]
                    vr_grad_nan = any(
                        p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                        for p in critic_params_vr
                    )
                    if not vr_grad_nan:
                        nn.utils.clip_grad_norm_(critic_params_vr, self.max_grad_norm)
                        self.critic_optimizer.step()
                    else:
                        print('[NAN IN VR GRAD] skipping value_replay step', flush=True)
                        self.critic_optimizer.zero_grad()
                    if not hasattr(self, '_unreal_stats'): self._unreal_stats = {}
                    self._unreal_stats["value_replay_loss"] = vr_loss.detach().item()
        # ──────────────────────────────────────────────────────────────

        self.buffer.clear()

        last_approx_kl = 0.0
        if epoch_kl_count > 0:
            last_approx_kl = (epoch_kl_sum / epoch_kl_count).item()

        self.stats = {
            "policy_loss": (total_policy_loss / max(n_updates, 1)).item(),
            "value_loss": (total_value_loss / max(n_updates, 1)).item(),
            "entropy": (total_entropy / max(n_updates, 1)).item(),
            "n_updates": n_updates,
            "actor_grad_norm": (total_actor_grad_norm / max(n_updates, 1)).item(),
            "critic_grad_norm": (total_critic_grad_norm / max(n_updates, 1)).item(),
            "approx_kl": last_approx_kl,
            "clip_frac": (total_clip_frac / max(n_updates, 1)).item(),
            "explained_var": explained_var.item(),
            "value_pred_error": value_pred_error.item(),
            "adv_mean": raw_adv_mean,
            "adv_std": raw_adv_std,
            "adv_frac_pos": raw_adv_frac_pos,
        }
        if self.aux_prediction:
            self.stats["dest_loss"] = (total_dest_loss / max(n_updates, 1)).item()
            if total_event_n > 0:
                self.stats["event_acc"] = total_event_acc / total_event_n
            if hasattr(self, '_unreal_stats') and self._unreal_stats:
                self.stats.update(self._unreal_stats)
        return self.stats

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_type": self.model_type,
            "aux_prediction": self.aux_prediction,
            "policy_state": self.policy.state_dict(),
            "actor_optimizer_state": self.actor_optimizer.state_dict(),
            "critic_optimizer_state": self.critic_optimizer.state_dict(),
            "stats": self.stats,
            "reward_normalizer": {
                "mean": self.reward_normalizer.mean.item(),
                "var": self.reward_normalizer.var.item(),
                "count": self.reward_normalizer.count,
            },
        }
        if self.scaler is not None:
            data["scaler_state"] = self.scaler.state_dict()
        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None:
            data["scheduler_state"] = scheduler.state_dict()
        data["epoch"] = int(getattr(self, "current_epoch", 0))
        data["entropy_coef"] = float(getattr(self, "entropy_coef", 0.0))
        torch.save(data, path)

    def load(self, path: str, load_optimizer: bool = True):
        ckpt = torch.load(path, map_location=self.device)

        ckpt_model_type = ckpt.get("model_type", "lstm")
        if ckpt_model_type != self.model_type:
            print(f"[load] WARNING: checkpoint model_type={ckpt_model_type} != current={self.model_type}")

        raw_sd = ckpt["policy_state"]
        target_compiled = any(k.startswith("_orig_mod.") for k in self.policy.state_dict())
        ckpt_compiled = any(k.startswith("_orig_mod.") for k in raw_sd)
        if ckpt_compiled and not target_compiled:
            raw_sd = {k[len("_orig_mod."):]: v for k, v in raw_sd.items()}
        elif target_compiled and not ckpt_compiled:
            raw_sd = {f"_orig_mod.{k}": v for k, v in raw_sd.items()}
        missing, unexpected = self.policy.load_state_dict(raw_sd, strict=False)
        if missing or unexpected:
            print(f"[load] missing={missing}, unexpected={unexpected}")
        if load_optimizer:
            if "actor_optimizer_state" in ckpt:
                try:
                    self.actor_optimizer.load_state_dict(ckpt["actor_optimizer_state"])
                except (ValueError, RuntimeError):
                    pass
                try:
                    self.critic_optimizer.load_state_dict(ckpt["critic_optimizer_state"])
                except (ValueError, RuntimeError):
                    pass
            elif "optimizer_state" in ckpt:
                try:
                    self.actor_optimizer.load_state_dict(ckpt["optimizer_state"])
                except (ValueError, RuntimeError):
                    pass

        self.stats = ckpt.get("stats", {})
        if "reward_normalizer" in ckpt:
            rn = ckpt["reward_normalizer"]
            self.reward_normalizer.mean = torch.tensor(rn["mean"], device=self.device)
            self.reward_normalizer.var = torch.tensor(rn["var"], device=self.device)
            self.reward_normalizer.count = rn["count"]
        if self.scaler is not None and "scaler_state" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state"])
        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None and "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        self.current_epoch = int(ckpt.get("epoch", 0))
        self.entropy_coef = float(ckpt.get("entropy_coef",
                                           getattr(self, "entropy_coef", 0.0)))
        if self.current_epoch:
            print(f"[load] resumed at epoch {self.current_epoch}, "
                  f"entropy_coef={self.entropy_coef:.5f}")
        return self.current_epoch
