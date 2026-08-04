"""
DDQN 训练脚本 — 无目的地信息, 纯外呼调度
包含: Optuna调参 + 早停 + 200 epochs默认
"""
from pathlib import Path

import sys, os, json, numpy as np, optuna
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eet"))

import torch
from src.env.elevator_env import ElevatorEnv
from src.data.dataset import load_raw_data

PROJ = Path(__file__).resolve().parents[1]
CKPT_DIR = PROJ / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mask_dest(obs):
    """屏蔽obs[80:100]目的地信息"""
    obs = obs.copy()
    obs[80:100] = 0.0
    return obs


class DuelingDQN(torch.nn.Module):
    def __init__(self, state_dim=89, action_dim=3, hidden=256):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
        )
        self.value = torch.nn.Linear(hidden, 1)
        self.advantage = torch.nn.Linear(hidden, action_dim)

    def forward(self, x):
        f = self.features(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + a - a.mean(dim=-1, keepdim=True)


class ReplayBuffer:
    def __init__(self, cap=50000):
        self.buf = []
        self.pos = 0
        self.cap = cap

    def push(self, s, a, r, ns, d):
        if len(self.buf) < self.cap:
            self.buf.append(None)
        self.buf[self.pos] = (s, a, r, ns, d)
        self.pos = (self.pos + 1) % self.cap

    def sample(self, bs):
        import random
        batch = random.sample(self.buf, min(bs, len(self.buf)))
        return tuple(np.array(x) for x in zip(*batch))

    def __len__(self):
        return len(self.buf)


class DDQNAgent:
    def __init__(self, lr=3e-4, gamma=0.99, tau=0.05, hidden=256,
                 eps_start=1.0, eps_end=0.05, eps_decay=5000):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps = 0
        self.action_dim = 3

        self.policy = DuelingDQN(89, 3, hidden).to(device)
        self.target = DuelingDQN(89, 3, hidden).to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = ReplayBuffer()

    def act(self, obs, eval_mode=False):
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        t = torch.from_numpy(obs[:89]).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.policy(t).argmax(dim=-1).item())

    def update(self, bs=64):
        if len(self.memory) < bs:
            return 0.0
        s, a, r, ns, d = self.memory.sample(bs)
        s = torch.from_numpy(s[:, :89]).float().to(self.device)
        a = torch.from_numpy(a).long().unsqueeze(1).to(self.device)
        r = torch.from_numpy(r).float().unsqueeze(1).to(self.device)
        ns = torch.from_numpy(ns[:, :89]).float().to(self.device)
        d = torch.from_numpy(d).float().unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_a = self.policy(ns).argmax(dim=-1, keepdim=True)
            target_q = r + self.gamma * self.target(ns).gather(1, next_a) * (1 - d)

        loss = torch.nn.MSELoss()(self.policy(s).gather(1, a), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        for tp, pp in zip(self.target.parameters(), self.policy.parameters()):
            tp.data.copy_(self.tau * pp.data + (1 - self.tau) * tp.data)

        self.steps += 1
        self.epsilon = max(self.eps_end, self.epsilon * (1 - 1 / self.eps_decay))
        return loss.item()

    def save(self, path):
        torch.save({
            "policy": self.policy.state_dict(),
            "target": self.target.state_dict(),
            "opt": self.optimizer.state_dict(),
            "steps": self.steps,
            "eps": self.epsilon,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(ckpt["policy"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["opt"])
        self.steps = ckpt["steps"]
        self.epsilon = ckpt["eps"]


def evaluate(agent, val_indices, event_seqs, event_lens, max_steps=500):
    agent.policy.eval()
    scores = []
    for idx in val_indices:
        seq = event_seqs[idx][:int(event_lens[idx])]
        env = ElevatorEnv()
        obs, _ = env.reset(options={"events": seq.copy()})
        done = False
        total, steps = 0.0, 0
        while not done and steps < max_steps:
            action = agent.act(mask_dest(obs), eval_mode=True)
            obs, r, done, _, _ = env.step(action)
            total += r
            steps += 1
        env.close()
        scores.append(total)
    agent.policy.train()
    return float(np.mean(scores))


def objective(trial):
    """Optuna objective: 30 episodes of training, return val reward."""
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.90, 0.999),
        "tau": trial.suggest_float("tau", 0.005, 0.1, log=True),
        "hidden": trial.suggest_categorical("hidden", [128, 256]),
        "eps_decay": trial.suggest_int("eps_decay", 2000, 8000, step=500),
    }
    agent = DDQNAgent(**params)

    data = load_raw_data()
    es = data["event_sequences"]["arr_0"]
    el = data["event_lengths"]["arr_0"]
    train_idx = np.arange(len(es))
    val_idx = train_idx[-10:]

    best = -9999
    for ep in range(30):
        idx = np.random.choice(train_idx[:-10])
        seq = es[idx][:int(el[idx])]
        env = ElevatorEnv()
        obs, _ = env.reset(options={"events": seq.copy()})
        done, steps = False, 0
        while not done and steps < 500:
            action = agent.act(mask_dest(obs))
            n_obs, r, done, _, _ = env.step(action)
            n_obs = mask_dest(n_obs)
            agent.memory.push(obs[:89], action, r, n_obs[:89], done)
            agent.update(64)
            obs = n_obs
            steps += 1
        env.close()

        if ep % 10 == 9:
            vr = evaluate(agent, val_idx, es, el)
            best = max(best, vr)
            trial.report(best, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return best


def run_optuna(n_trials=30):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print(f"\nBest trial: {study.best_trial.value:.1f}")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
    with open(str(CKPT_DIR / "ddqn_optuna_params.json"), "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
    return study.best_trial.params


def train(params=None, epochs=200):
    if params is None:
        params_path = CKPT_DIR / "ddqn_optuna_params.json"
        if params_path.exists():
            with open(params_path) as f:
                params = json.load(f)
            print(f"Loaded Optuna params from {params_path}")
        else:
            print("No Optuna params found, running search first...")
            params = run_optuna()

    agent = DDQNAgent(**params)
    data = load_raw_data()
    es = data["event_sequences"]["arr_0"]
    el = data["event_lengths"]["arr_0"]
    labels = np.squeeze(data["labels"]["arr_0"])

    import swanlab
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        np.arange(len(es)), test_size=0.15, random_state=42, stratify=labels)

    print(f"\n{'='*60}")
    print(f"DDQN Training (无目的地) — {epochs} epochs")
    print(f"Params: {params}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    print(f"{'='*60}")

    best_val = -9999.0
    patience = 0
    early_stop = 30
    history = {"train": [], "val": []}

    for ep in range(epochs):
        idx = np.random.choice(train_idx)
        seq = es[idx][:int(el[idx])]
        env = ElevatorEnv()
        obs, _ = env.reset(options={"events": seq.copy()})
        done, steps, ep_r = False, 0, 0.0
        while not done and steps < 500:
            action = agent.act(mask_dest(obs))
            n_obs, r, done, _, _ = env.step(action)
            n_obs = mask_dest(n_obs)
            agent.memory.push(obs[:89], action, r, n_obs[:89], done)
            agent.update(64)
            obs = n_obs
            ep_r += r
            steps += 1
        env.close()
        history["train"].append(ep_r)

        if ep % 10 == 9:
            vr = evaluate(agent, val_idx, es, el)
            history["val"].append(vr)
            swanlab.log({"train_reward": ep_r, "val_reward": vr, "epsilon": agent.epsilon, "lr": agent.optimizer.param_groups[0]["lr"]}, step=ep+1)
            print(f"  Ep {ep+1:3d}: train={ep_r:.1f} val={vr:.1f} eps={agent.epsilon:.3f}")

            if vr > best_val:
                best_val = vr
                patience = 0
                agent.save(str(CKPT_DIR / "ddqn_best.pt"))
                print(f"    → New best: {best_val:.1f}")
            else:
                patience += 1
                if patience >= early_stop:
                    print(f"  Early stop at ep {ep+1}")
                    break

    swanlab.finish()
    agent.save(str(CKPT_DIR / "ddqn_final.pt"))
    print(f"\nDone! Best val: {best_val:.1f}, Final eps: {agent.epsilon:.3f}")
    return best_val


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--optuna", action="store_true", help="Run Optuna search first")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    if args.optuna:
        params = run_optuna()
        train(params, args.epochs)
    else:
        train(epochs=args.epochs)
