"""GRU Actor-Critic network with UNREAL-style destination prediction.

  obs → actor_encoder (GRU) → actor → PPO loss
                             → dest_head → CE loss (grad flows back!)
  obs → critic_encoder (GRU) → critic → value loss
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _get_activation(name: str):
    return {"relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}.get(name, nn.ReLU())


class GRUActorCritic(nn.Module):
    """GRU-based Actor-Critic with UNREAL-style destination prediction."""

    def __init__(
        self,
        state_dim: int = 89,
        action_dim: int = 3,
        gru_hidden: int = 256,
        gru_layers: int = 2,
        gru_dropout: float = 0.0,
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
        dropout = gru_dropout if gru_layers > 1 else 0.0

        self.actor_encoder = nn.GRU(
            input_size=self.shared_dim, hidden_size=gru_hidden,
            num_layers=gru_layers, dropout=dropout, batch_first=True,
        )
        self.critic_encoder = nn.GRU(
            input_size=self.shared_dim, hidden_size=gru_hidden,
            num_layers=gru_layers, dropout=dropout, batch_first=True,
        )
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.actor_layer_norm = nn.LayerNorm(gru_hidden)
            self.critic_layer_norm = nn.LayerNorm(gru_hidden)

        act_fn = _get_activation(activation)
        actor_layers = [nn.Linear(gru_hidden, actor_hidden), act_fn]
        if actor_dropout > 0:
            actor_layers.append(nn.Dropout(p=actor_dropout))
        actor_layers.append(nn.Linear(actor_hidden, action_dim))
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = [nn.Linear(gru_hidden, critic_hidden), act_fn]
        if critic_dropout > 0:
            critic_layers.append(nn.Dropout(p=critic_dropout))
        critic_layers.append(nn.Linear(critic_hidden, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.aux_prediction = aux_prediction
        if aux_prediction:
            self.dest_head = nn.Sequential(
                nn.Linear(gru_hidden + 2, gru_hidden // 2),
                nn.ReLU(),
                nn.Linear(gru_hidden // 2, num_dest_classes),
            )
            # UNREAL: Reward Prediction head (3-class: pos/zero/neg)
            self.reward_pred_head = nn.Sequential(
                nn.Linear(gru_hidden, gru_hidden // 2),
                nn.ReLU(),
                nn.Linear(gru_hidden // 2, 3),
            )
        self._init_weights()

    def _init_weights(self):
        # ── GRU encoders: gain=1.0 for sigmoid/tanh gates ──
        for encoder in [self.actor_encoder, self.critic_encoder]:
            for name, param in encoder.named_parameters():
                if "weight_ih" in name:
                    nn.init.orthogonal_(param, gain=1.0)  # GRU uses sigmoid/tanh, NOT ReLU
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif "bias" in name:
                    nn.init.constant_(param, 0)

        # ── Linear heads: orthogonal init with activation-aware gain ──
        for head in [self.actor, self.critic]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    # Last layer has no activation → gain=1.0; middle → gain=√2
                    gain = 2.0 ** 0.5 if m.out_features != head[-1].out_features else 1.0
                    nn.init.orthogonal_(m.weight, gain=gain)
                    nn.init.constant_(m.bias, 0)

        if self.aux_prediction:
            for head, last_out in [(self.dest_head, self.dest_head[-1].out_features),
                                    (self.reward_pred_head, 3)]:
                for m in head.modules():
                    if isinstance(m, nn.Linear):
                        gain = 2.0 ** 0.5 if m.out_features != last_out else 1.0
                        nn.init.orthogonal_(m.weight, gain=gain)
                        nn.init.constant_(m.bias, 0)

    def _init_hidden(self, batch_size, device):
        return torch.zeros(self.actor_encoder.num_layers, batch_size, self.actor_encoder.hidden_size, device=device)

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

    def predict_reward(self, gru_out):
        """UNREAL: Predict next-step reward class from GRU output.
        Args:
            gru_out: (B, T, gru_hidden) - actor GRU output at each timestep
        Returns:
            reward_logits: (B, T, 3) - positive/zero/negative class logits
        """
        return self.reward_pred_head(gru_out)

"""GRUSharedActorCritic: true UNREAL-style shared encoder + multi-task heads.
obs -> shared GRU -> actor / critic / reward_pred / dest_head (no skip)
Keeps (h, h) hidden tuple + 4/5-value returns to stay drop-in compatible
with lstm_ppo.py training loop and eval scripts."""


class GRUSharedActorCritic(nn.Module):
    """UNREAL-faithful v2: ONE GRU encoder, actor/critic/event/reward_change heads.
    dest_head optional (default on = old ckpt compat; new training turns off).
    event_head predicts the NEXT event's zone (temporally valid, observable).
    reward_change_head predicts sign of next-step reward delta (balanced classes)."""

    def __init__(
        self,
        state_dim: int = 122,
        action_dim: int = 3,
        gru_hidden: int = 256,
        gru_layers: int = 2,
        gru_dropout: float = 0.0,
        actor_hidden: int = 64,
        critic_hidden: int = 64,
        activation: str = "relu",
        actor_dropout: float = 0.0,
        critic_dropout: float = 0.0,
        use_layer_norm: bool = False,
        aux_prediction: bool = False,
        num_dest_classes: int = 10,
        dest_head_on: bool = True,
        event_head_on: bool = True,
        reward_change_on: bool = True,
    ):
        super().__init__()
        dropout = gru_dropout if gru_layers > 1 else 0.0

        self.encoder = nn.GRU(
            input_size=state_dim, hidden_size=gru_hidden,
            num_layers=gru_layers, dropout=dropout, batch_first=True,
        )
        self.actor_encoder = self.encoder
        self.critic_encoder = self.encoder

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(gru_hidden)

        act_fn = _get_activation(activation)
        actor_layers = [nn.Linear(gru_hidden, actor_hidden), act_fn]
        if actor_dropout > 0:
            actor_layers.append(nn.Dropout(p=actor_dropout))
        actor_layers.append(nn.Linear(actor_hidden, action_dim))
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = [nn.Linear(gru_hidden, critic_hidden), act_fn]
        if critic_dropout > 0:
            critic_layers.append(nn.Dropout(p=critic_dropout))
        critic_layers.append(nn.Linear(critic_hidden, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.aux_prediction = aux_prediction
        self.dest_head_on = dest_head_on
        self.event_head_on = event_head_on
        self.reward_change_on = reward_change_on
        if aux_prediction:
            if dest_head_on:
                self.dest_head = nn.Sequential(
                    nn.Linear(gru_hidden, gru_hidden // 2),
                    nn.ReLU(),
                    nn.Linear(gru_hidden // 2, num_dest_classes),
                )
            if event_head_on:
                self.event_head = nn.Sequential(
                    nn.Linear(gru_hidden, gru_hidden // 2),
                    nn.ReLU(),
                    nn.Linear(gru_hidden // 2, 3),  # zone classes
                )
            if reward_change_on:
                self.reward_change_head = nn.Sequential(
                    nn.Linear(gru_hidden, gru_hidden // 2),
                    nn.ReLU(),
                    nn.Linear(gru_hidden // 2, 3),  # worse / unchanged / better
                )
            # legacy reward_pred_head kept for old-ckpt load compat
            self.reward_pred_head = nn.Sequential(
                nn.Linear(gru_hidden, gru_hidden // 2),
                nn.ReLU(),
                nn.Linear(gru_hidden // 2, 3),
            )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.encoder.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                nn.init.constant_(param, 0)
        for head in [self.actor, self.critic]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    gain = 2.0 ** 0.5 if m.out_features != head[-1].out_features else 1.0
                    nn.init.orthogonal_(m.weight, gain=gain)
                    nn.init.constant_(m.bias, 0)
        if self.aux_prediction:
            heads = []
            if self.dest_head_on:
                heads.append((self.dest_head, self.dest_head[-1].out_features))
            if self.event_head_on:
                heads.append((self.event_head, 3))
            if self.reward_change_on:
                heads.append((self.reward_change_head, 3))
            heads.append((self.reward_pred_head, 3))
            for head, last_out in heads:
                for m in head.modules():
                    if isinstance(m, nn.Linear):
                        gain = 2.0 ** 0.5 if m.out_features != last_out else 1.0
                        nn.init.orthogonal_(m.weight, gain=gain)
                        nn.init.constant_(m.bias, 0)

    def _init_hidden(self, batch_size, device):
        return torch.zeros(self.encoder.num_layers, batch_size,
                           self.encoder.hidden_size, device=device)

    def get_initial_hidden(self, batch_size, device):
        h = self._init_hidden(batch_size, device)
        return (h, h)

    def forward(self, obs_seq, hidden=None, return_dest=True):
        batch_size = obs_seq.size(0)
        if hidden is None:
            actor_hidden = self._init_hidden(batch_size, obs_seq.device)
            critic_hidden = self._init_hidden(batch_size, obs_seq.device)
        else:
            actor_hidden, critic_hidden = hidden

        out, new_hidden = self.encoder(obs_seq, actor_hidden)
        if self.use_layer_norm:
            out = self.layer_norm(out)

        dest_logits = None
        event_logits = None
        rc_logits = None
        if self.aux_prediction:
            if self.dest_head_on and return_dest:
                dest_logits = self.dest_head(out)
            if self.event_head_on:
                event_logits = self.event_head(out)
            if self.reward_change_on:
                rc_logits = self.reward_change_head(out)

        action_logits = self.actor(out)
        values = self.critic(out)
        return action_logits, values, (new_hidden, new_hidden), dest_logits, event_logits, rc_logits

    def get_action(self, obs_seq, hidden=None, deterministic=False):
        action_logits, values, hidden, dest_logits, event_logits, rc_logits = self.forward(obs_seq, hidden)
        dist = Categorical(logits=action_logits)
        action = action_logits.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, values.squeeze(-1), hidden, dest_logits

    def evaluate_actions(self, obs_seq, actions, hidden=None):
        action_logits, values, hidden, dest_logits, event_logits, rc_logits = self.forward(obs_seq, hidden)
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return action_logits, log_probs, values.squeeze(-1), entropy, hidden, dest_logits
