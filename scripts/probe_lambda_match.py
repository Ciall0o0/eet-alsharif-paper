# probe_lambda_match.py — eval-trajectory teacher-match per lambda clone, by load regime
import sys, os, warnings
warnings.filterwarnings("ignore")
os.environ["WEIGHT_MODE"] = "normal"
sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, torch
from bc_boost import build_model, gen_events, make_env, _eta

dev = "cuda" if torch.cuda.is_available() else "cpu"

def load(path):
    ck = torch.load(path, map_location=dev, weights_only=False)
    sd = ck.get("policy_state", ck)
    dim_k = "encoder.weight_ih_l0" if "encoder.weight_ih_l0" in sd else "weight_ih_l0"
    m = build_model(sd[dim_k].shape[1]).to(dev)
    m.load_state_dict(sd, strict=False); m.eval()
    return m

MODELS = {
    "lam5": load("checkpoints/bc_lb5_e20.pt"),
    "lam15": load("checkpoints/bc_lb15_e20.pt"),
    "lam30": load("checkpoints/bc_lb_norm_e20.pt"),
}

# shared trajectory (LB-30 teacher rolls out); record obs + all three lambda teacher actions
ev = gen_events(9999, 3.2)
env = make_env()
obs, _ = env.reset(options={"events": ev})
env._ep_cap = 300000
traj = []
done = False
while not done and len(traj) < 20000:
    if env.pending_calls:
        c = env.pending_calls[0]
        acts = {}
        for lam, tag in ((5.0, "lam5"), (15.0, "lam15"), (30.0, "lam30")):
            acts[tag] = min(range(3), key=lambda k: _eta(env, k, c) + lam * env.elevators[k].load_ratio)
        lm = max(e.load_ratio for e in env.elevators)
        traj.append((obs.copy(), acts, lm))
        a = acts["lam30"]
    else:
        a = 0
    obs, r, done, trunc, _ = env.step(a)
    if trunc:
        break
n = len(traj)
print(f"shared trajectory: {n} states", flush=True)

hs = {k: None for k in MODELS}
acts = {k: np.zeros(n, dtype=int) for k in MODELS}
for i, (o, *_ ) in enumerate(traj):
    for k, m in MODELS.items():
        ot = torch.as_tensor(o, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            a, _, _, hs[k], _ = m.get_action(ot, hs[k], deterministic=True)
        acts[k][i] = int(a.item())
print("decisions collected", flush=True)

loads = np.array([t[2] for t in traj])
for k in MODELS:
    t_k = np.array([t[1][k] for t in traj])
    m_all = (acts[k] == t_k).mean()
    lo = loads <= 0.6; hi = loads > 0.6
    m_lo = (acts[k][lo] == t_k[lo]).mean() if lo.sum() else float("nan")
    m_hi = (acts[k][hi] == t_k[hi]).mean() if hi.sum() else float("nan")
    print(f"{k}: match(all)={m_all*100:.1f}%  low-load={m_lo*100:.1f}% ({lo.sum()} states)  high-load={m_hi*100:.1f}% ({hi.sum()} states)", flush=True)

# cross: how similar are the clone's actions to the OTHER lambda teachers?
print("\n=== clone action similarity to each teacher ===")
for k in MODELS:
    for t_name in ("lam5", "lam15", "lam30"):
        t_arr = np.array([t[1][t_name] for t in traj])
        print(f"{k} vs {t_name} teacher: {(acts[k]==t_arr).mean()*100:.1f}%", flush=True)
