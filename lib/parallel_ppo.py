"""
并行PPO训练器 — 4环境并行, 手动GAE+PPO, 无依赖原始代码.
"""
from pathlib import Path
import sys, os, numpy as np, torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eet"))
from src.env.elevator_env import ElevatorEnv
from src.data.dataset import load_raw_data

PROJ = Path(__file__).resolve().parents[1]
CKPT_DIR = PROJ / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 89

def mask(obs):
    o = obs.copy(); o[80:100] = 0.0; return o[:89]


class LSTMActor(nn.Module):
    def __init__(self, input_dim=89, lstm_dim=256, action_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_dim, 2, batch_first=True)
        self.ln = nn.LayerNorm(lstm_dim)
        self.actor = nn.Sequential(nn.Linear(lstm_dim, 128), nn.ReLU(), nn.Linear(128, action_dim))
        self.critic = nn.Sequential(nn.Linear(lstm_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, obs_seq, hidden=None):
        out, hidden = self.lstm(obs_seq, hidden)
        h = self.ln(out)
        logits = self.actor(h)
        values = self.critic(h)
        return logits, values, hidden


class ParallelRunner:
    """N个并行环境, 每个独立推进"""
    def __init__(self, num_envs=4, max_steps=500):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.envs = [ElevatorEnv() for _ in range(num_envs)]
        self.obs = [None] * num_envs
        self.hidden = [None] * num_envs
        self.done = [True] * num_envs
        self.step_counts = [0] * num_envs
        self.total_steps = 0
        self.episode_rewards = [0.0] * num_envs
    
    def reset_env(self, i, events, policy):
        o, _ = self.envs[i].reset(options={"events": events.copy()})
        self.obs[i] = torch.from_numpy(mask(o)).float().to(device)
        self.hidden[i] = (torch.zeros(2, 1, 256, device=device),
                          torch.zeros(2, 1, 256, device=device))
        self.done[i] = False
        self.step_counts[i] = 0
        self.episode_rewards[i] = 0.0
    
    def step(self, i, policy):
        """推进单个环境一步"""
        with torch.no_grad():
            t = self.obs[i].unsqueeze(0).unsqueeze(0)
            logits, values, hidden = policy(t, self.hidden[i])
            probs = torch.distributions.Categorical(logits=logits[:, -1])
            action = probs.sample()
            log_prob = probs.log_prob(action)
            value = values[:, -1]
        
        o, r, d, _, _ = self.envs[i].step(int(action.item()))
        self.obs[i] = torch.from_numpy(mask(o)).float().to(device)
        self.hidden[i] = hidden
        self.done[i] = d
        self.step_counts[i] += 1
        self.total_steps += 1
        self.episode_rewards[i] += r
        
        # 达到最大步数则截断
        if self.step_counts[i] >= self.max_steps:
            self.done[i] = True
        
        return int(action.item()), r, d, log_prob, value.squeeze()
    
    def close(self):
        for e in self.envs:
            e.close()


def train(epochs=200, num_envs=2, load_augmented=True):
    rollout_steps = num_envs * 250  # 每环境250步
    import swanlab
    
    # 数据加载
    if load_augmented:
        p = PROJ / "augmented_dataset.npz"
        if p.exists():
            d = np.load(str(p)); es, el, labels = d["event_sequences"], d["event_lengths"], d["labels"]
        else:
            d = load_raw_data(); es, el = d["event_sequences"]["arr_0"], d["event_lengths"]["arr_0"]; labels = np.squeeze(d["labels"]["arr_0"])
    else:
        d = load_raw_data(); es, el = d["event_sequences"]["arr_0"], d["event_lengths"]["arr_0"]; labels = np.squeeze(d["labels"]["arr_0"])
    
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(np.arange(len(es)), test_size=0.15, random_state=42, stratify=labels)
    print(f"Data: {len(es)} episodes, {len(train_idx)} train, {len(val_idx)} val")
    
    # 模型
    policy = LSTMActor(INPUT_DIM, 256, 3).to(device)
    actor_opt = torch.optim.Adam(policy.actor.parameters(), lr=5e-4)
    critic_opt = torch.optim.Adam(policy.critic.parameters(), lr=5e-4)
    
    swanlab.init(project="elevator-ppo", config={"algo":"ParallelPPO","epochs":epochs,"aug":load_augmented},
                 name=f"ParallelPPO-aug{load_augmented}")
    
    runner = ParallelRunner(num_envs, max_steps=500)
    best_val = -9999.0
    patience = 0
    
    # 缓存：每个环境收集的轨迹
    class TrajBuffer:
        def __init__(self, cap=rollout_steps):
            self.states = []; self.actions = []; self.rewards = []
            self.dones = []; self.values = []; self.log_probs = []
            self.cap = cap
        def clear(self):
            self.states.clear(); self.actions.clear(); self.rewards.clear()
            self.dones.clear(); self.values.clear(); self.log_probs.clear()
        def ready(self): return len(self.states) >= self.cap
    
    buf = TrajBuffer()
    
    for epoch in range(epochs):
        # 为每个环境分配episode
        epoch_idx = np.random.permutation(train_idx)
        feed_ptr = 0
        for i in range(num_envs):
            if feed_ptr < len(epoch_idx):
                idx = epoch_idx[feed_ptr]; feed_ptr += 1
                runner.reset_env(i, es[idx][:int(el[idx])], policy)
        
        policy.eval()
        epoch_reward = 0.0
        epoch_steps = 0
        
        while epoch_steps < rollout_steps * 2:
            all_done = True
            for i in range(num_envs):
                if not runner.done[i]:
                    all_done = False
                    a, r, d, lp, v = runner.step(i, policy)
                    buf.states.append(runner.obs[i].cpu().numpy())
                    buf.actions.append(a); buf.rewards.append(r)
                    buf.dones.append(float(d)); buf.values.append(v.cpu().item())
                    buf.log_probs.append(lp.cpu().item())
                    epoch_reward += r; epoch_steps += 1
                    
                    if runner.done[i] and feed_ptr < len(epoch_idx):
                        idx = epoch_idx[feed_ptr]; feed_ptr += 1
                        runner.reset_env(i, es[idx][:int(el[idx])], policy)
                        if not runner.done[i]: all_done = False
            
            if all_done and epoch_steps < rollout_steps * 0.5: break
            
            if buf.ready():
                _do_ppo_update(policy, actor_opt, critic_opt, buf, epoch)
                buf.clear()
        
        if len(buf.states) > rollout_steps * 0.3:
            _do_ppo_update(policy, actor_opt, critic_opt, buf, epoch)
            buf.clear()
        
        if epoch % 10 == 9:
            vr = _evaluate(policy, val_idx, es, el)
            swanlab.log({"val": vr, "epoch_reward": epoch_reward}, step=epoch+1)
            print(f"  Ep {epoch+1}: reward={epoch_reward:.0f} val={vr:.1f}")
            if vr > best_val:
                best_val, patience = vr, 0
                torch.save(policy.state_dict(), str(CKPT_DIR / "parallel_ppo_best.pt"))
                print(f"    -> New best: {best_val:.1f}")
            else:
                patience += 1
                if patience >= 30:
                    print(f"  Early stop at ep {epoch+1}"); break
    
    swanlab.finish()
    runner.close()
    torch.save(policy.state_dict(), str(CKPT_DIR / "parallel_ppo_final.pt"))
    print(f"\nDone! Best val: {best_val:.1f}")


def _do_ppo_update(policy, actor_opt, critic_opt, buf, epoch, gamma=0.99, lam=0.95, clip=0.2, ppo_epochs=5):
    if len(buf.states) < 16: return
    policy.train()
    
    s = torch.from_numpy(np.array(buf.states)).float().to(device)
    a = torch.tensor(buf.actions, device=device)
    r = torch.tensor(buf.rewards, dtype=torch.float32, device=device)
    d = torch.tensor(buf.dones, dtype=torch.float32, device=device)
    v = torch.tensor(buf.values, dtype=torch.float32, device=device)
    old_lp = torch.tensor(buf.log_probs, dtype=torch.float32, device=device)
    
    # GAE
    adv = torch.zeros_like(r)
    last = 0.0
    for t in reversed(range(len(r))):
        nv = 0.0 if d[t] else (v[t+1] if t+1 < len(r) else 0.0)
        delta = r[t] + gamma * nv - v[t]
        last = delta + gamma * lam * (1 - d[t]) * last
        adv[t] = last
    ret = adv + v
    
    # 序列化输入 (LSTM需要序列维度)
    s_seq = s.unsqueeze(0)  # [1, T, 89]
    
    for _ in range(ppo_epochs):
        logits, vals, _ = policy(s_seq)
        logits, vals = logits.squeeze(0), vals.squeeze(0).squeeze(-1)
        probs = torch.distributions.Categorical(logits=logits)
        lp = probs.log_prob(a)
        ent = probs.entropy().mean()
        
        ratio = torch.exp(lp - old_lp)
        a_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg1 = -ratio * a_norm
        pg2 = -torch.clamp(ratio, 1-clip, 1+clip) * a_norm
        al = pg1.max(pg2).mean()
        cl = nn.MSELoss()(vals, ret)
        loss = al + 0.5 * cl - 0.01 * ent
        
        actor_opt.zero_grad(); critic_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        actor_opt.step(); critic_opt.step()
        
        kl = (old_lp - lp).mean().item()
        if kl > 0.02: break
    
    policy.eval()


def _evaluate(policy, val_idx, es, el, max_steps=500):
    policy.eval()
    scores = []
    for idx in val_idx:
        env = ElevatorEnv()
        o, _ = env.reset(options={"events": es[idx][:int(el[idx])].copy()})
        h = (torch.zeros(2, 1, 256, device=device), torch.zeros(2, 1, 256, device=device))
        done, steps, total = False, 0, 0.0
        while not done and steps < max_steps:
            t = torch.from_numpy(mask(o)).float().to(device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits, _, h = policy(t, h)
                a = logits[:, -1].argmax(dim=-1).item()
            o, r, done, _, _ = env.step(a)
            total += r; steps += 1
        env.close()
        scores.append(total)
    return float(np.mean(scores))


if __name__ == "__main__":
    import argparse, traceback
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--no-aug", action="store_true")
    args = parser.parse_args()
    try:
        train(epochs=args.epochs, load_augmented=not args.no_aug)
    except Exception as e:
        print(f"CRASH: {e}")
        traceback.print_exc()
        import sys; sys.exit(1)
