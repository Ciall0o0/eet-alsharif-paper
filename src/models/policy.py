"""LSTM Actor-Critic network with UNREAL-style destination prediction.

  obs -> actor_encoder (LSTM) -> actor -> PPO loss
                               -> dest_head -> CE loss (grad flows back!)
  obs -> critic_encoder (LSTM) -> critic -> value loss
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _get_activation(name: str):
    return {"relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}.get(name, nn.ReLU())


class LSTMActorCritic(nn.Module):
    """LSTM-based Actor-Critic with UNREAL-style destination prediction."""

    def __init__(
        self,
        state_dim: int = 89,
        action_dim: int = 3,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.0,
        actor_hidden: int = 64,
        critic_hidden: int = 64,
        activation: str = "relu",
        actor_dropout: float = 0.0,
        critic_dropout: float = 0.0,
        use_layer_norm: bool = False,
        aux_prediction: bool = False,
        num_dest_classes: int = 10,
    ):
        super().__init__()
        self.shared_dim = state_dim
        dropout = lstm_dropout if lstm_layers > 1 else 0.0

        self.actor_encoder = nn.LSTM(
            input_size=self.shared_dim, hidden_size=lstm_hidden,
            num_layers=lstm_layers, dropout=dropout, batch_first=True,
        )
        self.critic_encoder = nn.LSTM(
            input_size=self.shared_dim, hidden_size=lstm_hidden,
            num_layers=lstm_layers, dropout=dropout, batch_first=True,
        )
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.actor_layer_norm = nn.LayerNorm(lstm_hidden)
            self.critic_layer_norm = nn.LayerNorm(lstm_hidden)

        act_fn = _get_activation(activation)
        actor_layers = [nn.Linear(lstm_hidden, actor_hidden), act_fn]
        if actor_dropout > 0:
            actor_layers.append(nn.Dropout(p=actor_dropout))
        actor_layers.append(nn.Linear(actor_hidden, action_dim))
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = [nn.Linear(lstm_hidden, critic_hidden), act_fn]
        if critic_dropout > 0:
            critic_layers.append(nn.Dropout(p=critic_dropout))
        critic_layers.append(nn.Linear(critic_hidden, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.aux_prediction = aux_prediction
        if aux_prediction:
            self.dest_head = nn.Sequential(
                nn.Linear(lstm_hidden + 2, lstm_hidden // 2),
                nn.ReLU(),
                nn.Linear(lstm_hidden // 2, num_dest_classes),
            )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.actor_encoder.named_parameters():
            if "weight_ih" in name: nn.init.xavier_uniform_(param)
            elif "weight_hh" in name: nn.init.orthogonal_(param, gain=1.0)
        for name, param in self.critic_encoder.named_parameters():
            if "weight_ih" in name: nn.init.xavier_uniform_(param)
            elif "weight_hh" in name: nn.init.orthogonal_(param, gain=1.0)

    def _init_hidden(self, batch_size, device):
        h = torch.zeros(self.actor_encoder.num_layers, batch_size, self.actor_encoder.hidden_size, device=device)
        c = torch.zeros(self.actor_encoder.num_layers, batch_size, self.actor_encoder.hidden_size, device=device)
        return (h, c)

    def get_initial_hidden(self, batch_size, device):
        return (self._init_hidden(batch_size, device), self._init_hidden(batch_size, device))

    def forward(self, obs_seq, hidden=None, return_dest=True):
        batch_size = obs_seq.size(0)
        if hidden is None:
            actor_hidden = self._init_hidden(batch_size, obs_seq.device)
            critic_hidden = self._init_hidden(batch_size, obs_seq.device)
        else:
            actor_hidden, critic_hidden = hidden

        actor_out, actor_hidden = self.actor_encoder(obs_seq, actor_hidden)
        critic_out, critic_hidden = self.critic_encoder(obs_seq, critic_hidden)

        if self.use_layer_norm:
            actor_out = self.actor_layer_norm(actor_out)
            critic_out = self.critic_layer_norm(critic_out)

        dest_logits = None
        if return_dest and self.aux_prediction:
            dest_logits = self.dest_head(torch.cat([actor_out, obs_seq[:, :, -2:]], dim=-1))  # skip
            # UNREAL: CE gradient flows through dest_head -> actor_encoder

        action_logits = self.actor(actor_out)
        values = self.critic(critic_out)
        return action_logits, values, (actor_hidden, critic_hidden), dest_logits

    def get_action(self, obs_seq, hidden=None, deterministic=False):
        action_logits, values, hidden, dest_logits = self.forward(obs_seq, hidden)
        dist = Categorical(logits=action_logits)
        action = action_logits.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, values.squeeze(-1), hidden, dest_logits

    def evaluate_actions(self, obs_seq, actions, hidden=None):
        action_logits, values, hidden, dest_logits = self.forward(obs_seq, hidden)
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return action_logits, log_probs, values.squeeze(-1), entropy, hidden, dest_logits
