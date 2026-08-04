"""
LSTM+PPO 训练脚本 — 无目的地信息, 纯外呼调度
包含: Optuna调参 + 早停 + 200 epochs默认
"""
from pathlib import Path

import sys, os, json, numpy as np, optuna
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eet"))

import torch
import torch.nn as nn
from src.env.elevator_env import ElevatorEnv
from src.data.dataset import load_raw_data
from src.utils import load_config, PROJ_ROOT

PROJ = Path(__file__).resolve().parents[1]
CKPT_DIR = PROJ / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 89  # 109 - 20(dest)


def mask_dest(obs):
    obs = obs.copy()
    obs[80:100] = 0.0
    return obs


class LSTMActorCritic(nn.Module):
    """LSTM+PPO策略网络 (89维输入, 3维动作)"""

    def __init__(self, lstm_hidden=256, lstm_layers=2, mlp_hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM, lstm_hidden, lstm_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(lstm_hidden)
        self.actor = nn.Sequential(
            nn.Linear(lstm_hidden, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, 3),
        )
        self.critic = nn.Sequential(
            nn.Linear(lstm_hidden, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, obs_seq, hidden=None):
        out, hidden = self.lstm(obs_seq, hidden)
        # out: (B, T, H)
        h = self.layer_norm(out)
        logits = self.actor(h)    # (B, T, 3)
        value = self.critic(h)     # (B, T, 1)
        return logits, value, hidden

    def get_action(self, obs_seq, hidden=None, deterministic=False):
        logits, value, hidden = self.forward(obs_seq, hidden)
        logits = logits[:, -1]  # (B, T, 3) → (B, 3) for single-step
        value = value[:, -1]    # (B, T, 1) → (B, 1)
        probs = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = probs.sample()
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        return action, log_prob, entropy, hidden

    def get_initial_hidden(self, batch_size, dev):
        h = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=dev)
        c = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=dev)
        return (h, c)


class PPOTrainer:
    def __init__(self, lr=5e-4, gamma=0.99, gae_lambda=0.95, clip=0.2,
                 ent_coef=0.05, vf_coef=0.5, lstm_hidden=256, mlp_hidden=64):
        self.policy = LSTMActorCritic(lstm_hidden, 2, mlp_hidden).to(device)
        self.actor_opt = torch.optim.Adam(self.policy.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.policy.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.device = device

    def compute_gae(self, rewards, values, dones):
        if len(rewards) == 0:
            return torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        """计算GAE优势估计"""
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = 0.0 if dones[t] else values[t]
            else:
                next_val = 0.0 if dones[t] else values[t + 1]
            delta = rewards[t] + self.gamma * next_val - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    def update(self, obs_seq, actions, old_log_probs, returns, advantages):
        """单次PPO更新"""
        logits, values, _ = self.policy(obs_seq)
        probs = torch.distributions.Categorical(logits=logits)
        log_probs = probs.log_prob(actions)
        entropy = probs.entropy().mean()

        ratio = torch.exp(log_probs - old_log_probs)
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        pg_loss1 = -ratio * adv
        pg_loss2 = -torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
        actor_loss = pg_loss1.max(pg_loss2).mean()
        critic_loss = nn.MSELoss()(values.squeeze(), returns)
        loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy

        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.actor_opt.step()
        self.critic_opt.step()
        return loss.item(), entropy.item()

    def save(self, path):
        torch.save({
            "policy": self.policy.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(ckpt["policy"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])


def collect_episode(env, policy, hidden, obs, max_steps=500):
    """收集一个完整episode的轨迹"""
    obs_list, act_list, rew_list, done_list = [], [], [], []
    hiddens = []
    obs = mask_dest(obs)
    done, steps = False, 0

    while not done and steps < max_steps:
        t = torch.from_numpy(obs[:INPUT_DIM]).float().to(device).unsqueeze(0).unsqueeze(0)
        action, log_prob, entropy, hidden = policy.get_action(t, hidden)
        action_int = int(action.item())
        n_obs, reward, done, _, _ = env.step(action_int)
        n_obs = mask_dest(n_obs)

        obs_list.append(obs[:INPUT_DIM])
        act_list.append(action_int)
        rew_list.append(reward)
        done_list.append(float(done))
        hiddens.append((hidden[0].cpu(), hidden[1].cpu()))

        obs = n_obs
        steps += 1

    return obs_list, act_list, rew_list, done_list, steps


def evaluate(trainer, val_indices, event_seqs, event_lens, max_steps=500):
    trainer.policy.eval()
    scores = []
    for idx in val_indices:
        seq = event_seqs[idx][:int(event_lens[idx])]
        env = ElevatorEnv()
        obs, _ = env.reset(options={"events": seq.copy()})
        obs = mask_dest(obs)
        hidden = trainer.policy.get_initial_hidden(1, device)
        done, steps, total = False, 0, 0.0
        while not done and steps < max_steps:
            t = torch.from_numpy(obs[:INPUT_DIM]).float().to(device).unsqueeze(0).unsqueeze(0)
            action, _, _, hidden = trainer.policy.get_action(t, hidden, deterministic=True)
            obs, r, done, _, _ = env.step(int(action.item()))
            obs = mask_dest(obs)
            total += r
            steps += 1
        env.close()
        scores.append(total)
    trainer.policy.train()
    return float(np.mean(scores))


def objective(trial):
    """Optuna: 30 episodes training, return val reward"""
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.90, 0.999),
        "lstm_hidden": 128,
        "mlp_hidden": trial.suggest_categorical("mlp_hidden", [64, 128]),
        "ent_coef": trial.suggest_float("ent_coef", 0.01, 0.15),
        "clip": trial.suggest_float("clip", 0.1, 0.3),
    }
    trainer = PPOTrainer(**params)

    data = load_raw_data()
    es = data["event_sequences"]["arr_0"]
    el = data["event_lengths"]["arr_0"]
    val_idx = np.arange(len(es))[-10:]
    best = -9999.0

    for ep in range(30):
        idx = np.random.choice(np.arange(len(es))[:-10])
        seq = es[idx][:int(el[idx])]
        env = ElevatorEnv()
        obs, _ = env.reset(options={"events": seq.copy()})
        hidden = trainer.policy.get_initial_hidden(1, device)

        obs_list, act_list, rew_list, done_list, _ = collect_episode(env, trainer.policy, hidden, obs)
        env.close()

        if len(obs_list) < 5:
            __import__('sys').stderr.write(f"  [DEBUG] ep {ep}: obs_list={len(obs_list)} steps={steps}\n")
            continue

        # 构造tensor
        obs_t = torch.from_numpy(np.array(obs_list)).float().to(device).unsqueeze(0)
        act_t = torch.tensor(act_list, device=device)
        rew_t = torch.tensor(rew_list, dtype=torch.float32, device=device)
        done_t = torch.tensor(done_list, dtype=torch.float32, device=device)

        # 获取价值和旧log_prob
        with torch.no_grad():
            logits, values, _ = trainer.policy(obs_t)
            old_probs = torch.distributions.Categorical(logits=logits)
            old_log = old_probs.log_prob(act_t)

        # GAE
        adv, ret = trainer.compute_gae(rew_t, values.squeeze(), done_t)
        loss, ent = trainer.update(obs_t, act_t, old_log, ret, adv)

        if ep % 10 == 9:
            vr = evaluate(trainer, val_idx, es, el)
            best = max(best, vr)
            trial.report(best, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best


def run_optuna(n_trials=20):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print(f"\nBest trial: {study.best_trial.value:.1f}")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
    with open(str(CKPT_DIR / "lstm_ppo_optuna_params.json"), "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
    return study.best_trial.params


def train(params=None, epochs=200):
    if params is None:
        params_path = CKPT_DIR / "lstm_ppo_optuna_params.json"
        if params_path.exists():
            with open(params_path) as f:
                params = json.load(f)
        else:
            print("No Optuna params, running search...")
            params = run_optuna()

    trainer = PPOTrainer(**params)
    data = load_raw_data()
    es = data["event_sequences"]["arr_0"]
    el = data["event_lengths"]["arr_0"]
    labels = np.squeeze(data["labels"]["arr_0"])

    import swanlab
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        np.arange(len(es)), test_size=0.15, random_state=42, stratify=labels)

    print(f"\n{'='*60}")
    print(f"LSTM+PPO Training (无目的地) — {epochs} epochs")
    print(f"Params: {params}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    print(f"{'='*60}")

    best_val = -9999.0
    patience = 0
    early_stop = 30
    actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.actor_opt, T_max=epochs, eta_min=params.get("lr", 5e-4) * 0.1)

    for ep in range(epochs):
        idx = np.random.choice(train_idx)
        seq = es[idx][:int(el[idx])]
        env = ElevatorEnv()
        obs, _ = env.reset(options={"events": seq.copy()})
        hidden = trainer.policy.get_initial_hidden(1, device)

        obs_list, act_list, rew_list, done_list, steps = collect_episode(env, trainer.policy, hidden, obs)
        env.close()

        if len(obs_list) < 5:
            continue

        obs_t = torch.from_numpy(np.array(obs_list)).float().to(device).unsqueeze(0)
        act_t = torch.tensor(act_list, device=device)
        rew_t = torch.tensor(rew_list, dtype=torch.float32, device=device)
        done_t = torch.tensor(done_list, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits, values, _ = trainer.policy(obs_t)
            old_probs = torch.distributions.Categorical(logits=logits)
            old_log = old_probs.log_prob(act_t)

        adv, ret = trainer.compute_gae(rew_t, values.squeeze(), done_t)
        loss, ent = trainer.update(obs_t, act_t, old_log, ret, adv)
        actor_scheduler.step()

        if ep % 10 == 9:
            vr = evaluate(trainer, val_idx, es, el)
            try:
                swanlab.log({
                    "val_reward": vr, "loss": loss, "entropy": ent, "steps": steps,
                    "actor_lr": trainer.actor_opt.param_groups[0]["lr"],
                }, step=ep+1)
            except Exception:
                pass
            print(f"  Ep {ep+1:3d}: steps={steps} val={vr:.1f} loss={loss:.3f} ent={ent:.3f}")

            if vr > best_val:
                best_val = vr
                patience = 0
                trainer.save(str(CKPT_DIR / "lstm_ppo_best.pt"))
                print(f"    → New best: {best_val:.1f}")
            else:
                patience += 1
                if patience >= early_stop:
                    print(f"  Early stop at ep {ep+1}")
                    break

    try:
        swanlab.finish()
    except Exception:
        pass
    trainer.save(str(CKPT_DIR / "lstm_ppo_final.pt"))
    print(f"\nDone! Best val: {best_val:.1f}")
    return best_val


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--optuna", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    if args.optuna:
        params = run_optuna()
        train(params, args.epochs)
    else:
        train(epochs=args.epochs)
