# cmp_mix_sl.py — action comparison: bc_lb_norm vs sl1 vs mix_gate (e20), teachers LB/ETA/gate
import sys, os, warnings, json
warnings.filterwarnings("ignore")
os.environ["WEIGHT_MODE"] = "normal"
os.environ["TEACHER"] = "lb"
sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, torch
from bc_boost import build_model, gen_events, make_env, _eta, LB_LAMBDA

dev = "cuda" if torch.cuda.is_available() else "cpu"
GATE_TH = 0.6

def load_policy(path):
    ck = torch.load(path, map_location=dev, weights_only=False)
    sd = ck.get("policy_state", ck)
    n_in = sd["encoder.weight_ih_l0"].shape[1]
    m = build_model(n_in).to(dev)
    m.load_state_dict(sd, strict=False); m.eval()
    return m

def decide(m, obs, h):
    ot = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        a, _, _, h, _ = m.get_action(ot, h, deterministic=True)
    return int(a.item()), h

print("loading policies...", flush=True)
pols = {
    "bc_lb":  load_policy("checkpoints/bc_lb_norm_e20.pt"),
    "sl1":    load_policy("checkpoints/bc_sl1_e20.pt"),
    "mix_gate": load_policy("checkpoints/bc_mix_gate_e20.pt"),
}
print("ok", flush=True)

# shared trajectory: LB teacher rolls out; record obs + teacher actions (incl. gate teacher)
ev = gen_events(9999, 3.2)
env = make_env()
obs, _ = env.reset(options={"events": ev})
env._ep_cap = 300000
traj = []
done = False
while not done and len(traj) < 20000:
    if env.pending_calls:
        c = env.pending_calls[0]
        load_max = max(e.load_ratio for e in env.elevators)
        a_lb  = min(range(3), key=lambda k: _eta(env, k, c) + LB_LAMBDA * env.elevators[k].load_ratio)
        a_sde = min(range(3), key=lambda k: _eta(env, k, c))
        a_gate = a_lb if load_max > GATE_TH else a_sde
        traj.append((obs.copy(), a_lb, a_sde, a_gate, load_max))
    else:
        a_lb = a_sde = a_gate = 0
    obs, r, done, trunc, _ = env.step(a_lb)
    if trunc:
        break
n = len(traj)
print(f"shared trajectory: {n} decision states", flush=True)

hs = {name: None for name in pols}
acts = {name: np.zeros(n, dtype=int) for name in pols}
for i, (o, *_ ) in enumerate(traj):
    for name, m in pols.items():
        a, hs[name] = decide(m, o, hs[name])
        acts[name][i] = a
print("decisions collected", flush=True)

t_lb = np.array([t[1] for t in traj])
t_sde = np.array([t[2] for t in traj])
t_gate = np.array([t[3] for t in traj])
loads = np.array([t[4] for t in traj])

def match(x, y): return float((x == y).mean())

names = list(pols)
print("\n=== policy x teacher match ===")
print(f"{'':>10} | {'LB-30':>8} | {'ETA':>8} | {'gate':>8}")
for p in names:
    print(f"{p:>10} | {match(acts[p], t_lb)*100:7.1f}% | {match(acts[p], t_sde)*100:7.1f}% | {match(acts[p], t_gate)*100:7.1f}%")
print(f"{'gate teacher':>10} | {match(t_gate, t_lb)*100:7.1f}% | {match(t_gate, t_sde)*100:7.1f}% | {'---':>8}")

print("\n=== policy x policy match ===")
for i, a in enumerate(names):
    for b in names[i+1:]:
        print(f"{a} vs {b}: {match(acts[a], acts[b])*100:.1f}%")

print("\n=== action marginals ===")
for p in names:
    m_ = [float((acts[p] == v).mean())*100 for v in (0,1,2)]
    print(f"{p:>10}: [{m_[0]:.1f}, {m_[1]:.1f}, {m_[2]:.1f}]")
print(f"{'LB teacher':>10}: [{float((t_lb==0).mean())*100:.1f}, {float((t_lb==1).mean())*100:.1f}, {float((t_lb==2).mean())*100:.1f}]")
print(f"{'gate teacher':>10}: [{float((t_gate==0).mean())*100:.1f}, {float((t_gate==1).mean())*100:.1f}, {float((t_gate==2).mean())*100:.1f}]")

print("\n=== confusion: given X's action, Y's distribution ===")
def conf(Y, X):
    out = []
    for v in (0, 1, 2):
        sel = acts[X] == v
        out.append([float((acts[Y][sel] == u).mean())*100 for u in (0, 1, 2)])
    return out

for X, Y in (("bc_lb", "mix_gate"), ("mix_gate", "bc_lb"), ("bc_lb", "sl1"),
             ("sl1", "bc_lb"), ("mix_gate", "sl1"), ("sl1", "mix_gate")):
    print(f"\n  given {X}'s action -> {Y} dist:")
    for v, row in enumerate(conf(Y, X)):
        print(f"    {X}={v}: {[f'{x:.1f}' for x in row]}")

# disagreement conditioned on load (mix_gate vs bc_lb, sl1 vs bc_lb)
print("\n=== disagreement by load regime ===")
for X, Y in (("mix_gate", "bc_lb"), ("sl1", "bc_lb"), ("mix_gate", "sl1")):
    d = acts[X] != acts[Y]
    lo = loads <= 0.6; hi = loads > 0.6
    print(f"{X} vs {Y}: overall {d.mean()*100:.1f}% | low-load {d[lo].mean()*100:.1f}% ({lo.sum()}) | high-load {d[hi].mean()*100:.1f}% ({hi.sum()})")
