# -*- coding: utf-8 -*-
"""Compare BC vs pure-PPO action behavior:
1) Agreement on teacher (SD-ETA) trajectory states (same obs sequence, independent hiddens)
2) Own-rollout action distributions."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, torch
from bc_boost import build_model, gen_events, make_env, _eta
from src.models.gru_ppo import GRUSharedActorCritic

dev = "cuda"

def load(tag, ckpt, aux=False):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    sd = ck.get("policy_state", ck)
    n_in = sd["encoder.weight_ih_l0"].shape[1]
    if tag == "bc":
        m = build_model(n_in).to(dev)
    else:
        m = GRUSharedActorCritic(state_dim=n_in, action_dim=3, aux_prediction=aux,
                                 num_dest_classes=10, use_layer_norm=True,
                                 gru_hidden=256, gru_layers=2, gru_dropout=0.1,
                                 actor_hidden=64, critic_hidden=64).to(dev)
    m.load_state_dict(sd, strict=False); m.eval()
    return m

def decide(m, obs, h):
    ot = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        a, _, _, h, _ = m.get_action(ot, h, deterministic=True)
    return int(a.item()), h

# ---- 1. Teacher trajectory: record obs sequence + teacher actions ----
ev = gen_events(9999, 3.2)
env = make_env()
obs, _ = env.reset(options={"events": ev})
env._ep_cap = 300000
traj = []  # (obs, teacher_action)
done = False
while not done and len(traj) < 20000:
    if env.pending_calls:
        c = env.pending_calls[0]
        a = min(range(3), key=lambda k: _eta(env, k, c))
    else:
        a = 0
    traj.append((obs.copy(), a))
    obs, r, done, trunc, _ = env.step(a)
    if trunc: break
print(f"teacher trajectory: {len(traj)} decision states", flush=True)

# ---- 2. BC & PPO decide on the same states ----
bc = load("bc", "checkpoints/bc_boost_e20.pt")
ppo = load("ppo", "checkpoints/probe_8x_pponoaux_s42/ppo_elevator_best.pt", aux=False)
h_bc, h_ppo = bc.get_initial_hidden(1, dev), ppo.get_initial_hidden(1, dev)
a_bc, a_ppo, a_teacher = [], [], []
for obs_i, a_t in traj:
    ab, h_bc = decide(bc, obs_i, h_bc)
    ap, h_ppo = decide(ppo, obs_i, h_ppo)
    a_bc.append(ab); a_ppo.append(ap); a_teacher.append(a_t)
a_bc, a_ppo, a_teacher = map(np.array, (a_bc, a_ppo, a_teacher))

def match(x, y): return (x == y).mean()
print(f"BC   vs teacher match: {match(a_bc, a_teacher)*100:.1f}%", flush=True)
print(f"PPO  vs teacher match: {match(a_ppo, a_teacher)*100:.1f}%", flush=True)
print(f"BC   vs PPO   match:   {match(a_bc, a_ppo)*100:.1f}%", flush=True)

# disagreement pattern
dis = a_bc != a_ppo
print(f"\ndisagreement rate: {dis.mean()*100:.1f}% (n={dis.sum()})", flush=True)
print("confusion (BC row x PPO col):")
conf = np.zeros((3, 3), dtype=int)
for x, y in zip(a_bc, a_ppo):
    conf[x, y] += 1
print(conf, flush=True)
print("marginal BC:", np.bincount(a_bc, minlength=3), "PPO:", np.bincount(a_ppo, minlength=3), flush=True)

# ---- 3. Own-rollout action distributions ----
def own_dist(m, tag):
    ev2 = gen_events(10001, 3.2)
    env2 = make_env()
    obs2, _ = env2.reset(options={"events": ev2})
    env2._ep_cap = 300000
    h = m.get_initial_hidden(1, dev)
    acts = []
    done = False
    while not done and len(acts) < 20000:
        if env2.pending_calls:
            a, h = decide(m, obs2, h)
        else:
            a = 0
        obs2, r, done, trunc, _ = env2.step(a)
        acts.append(a)
        if trunc: break
    d = np.bincount(acts, minlength=3)
    print(f"{tag} own-rollout action dist (n={len(acts)}): {d} -> {np.round(d/len(acts), 3)}", flush=True)
    return d

own_dist(bc, "BC ")
own_dist(ppo, "PPO")
