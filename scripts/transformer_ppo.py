"""Causal Transformer encoder actor-critic for BC (architecture ablation vs GRU).

Matches the GRUSharedActorCritic training contract: forward(obs_seq, hidden) ->
(action_logits[B,T,A], values, hidden, None, None, None). Each position attends only
to past positions (causal mask), so evaluation with a sliding window is consistent
with training on 64-step chunks.
"""
import math
import torch
import torch.nn as nn


class TFEncoderActorCritic(nn.Module):
    def __init__(self, state_dim: int = 128, action_dim: int = 3,
                 d_model: int = 256, nhead: int = 8, n_layers: int = 3,
                 dim_feedforward: int = 512, dropout: float = 0.1,
                 max_seq: int = 64, actor_hidden: int = 64):
        super().__init__()
        self.proj = nn.Linear(state_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_seq, d_model))
        nn.init.normal_(self.pos, 0.0, 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="relu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.actor = nn.Sequential(
            nn.Linear(d_model, actor_hidden), nn.ReLU(), nn.Linear(actor_hidden, action_dim))
        self.critic = nn.Sequential(
            nn.Linear(d_model, actor_hidden), nn.ReLU(), nn.Linear(actor_hidden, 1))

    def _causal_mask(self, T, device):
        return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

    def forward(self, obs_seq, hidden=None):
        """obs_seq [B, T, D] -> (logits[B,T,A], values[B,T,1], None, None, None, None)"""
        B, T, D = obs_seq.shape
        x = self.proj(obs_seq) + self.pos[:, :T]
        mask = self._causal_mask(T, obs_seq.device)
        h = self.encoder(x, mask=mask)  # [B, T, d_model]
        logits = self.actor(h)
        values = self.critic(h)
        return logits, values, None, None, None, None

    def get_action(self, obs_seq, hidden, deterministic=True):
        """obs_seq [B, T, D] (already a window ending at the decision step)."""
        with torch.no_grad():
            logits, values, *_ = self.forward(obs_seq, hidden)
            logits_t = logits[:, -1]
            if deterministic:
                a = logits_t.argmax(-1)
            else:
                a = torch.distributions.Categorical(logits=logits_t).sample()
        return a, None, None, None, values[:, -1]

    def get_initial_hidden(self, batch_size, dev):
        return None
