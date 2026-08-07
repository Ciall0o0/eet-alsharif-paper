# eval_matrix.py — batch eval of all BC ckpt series on 9 held-out seeds
# Usage: python /tmp/eval_matrix.py   (reads results/eval_matrix.json on completion)
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "/tmp")
import numpy as np, torch
from bc_boost import build_model, gen_events, make_env, run_policy

dev = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS9 = (9999, 10001, 10003, 10005, 10007, 10009, 10011, 10013, 10015)
SEEDS3 = (9999, 10001, 10003)

# (tag, ckpt, seeds, scale, n_floors, obs_dim)
CONFIGS = [
    # seed robustness: e20 sweet spot, 9 seeds
    ("bc_boost",   "bc_boost_e20.pt",   SEEDS9, 3.2, 10, 128),
    ("bc_seed360", "bc_seed360_e20.pt", SEEDS9, 3.2, 10, 128),
    ("bc_seed712", "bc_seed712_e20.pt", SEEDS9, 3.2, 10, 128),
    # data efficiency: e20 sweet spot, 9 seeds
    ("bc_eff5",  "bc_eff5_e20.pt",  SEEDS9, 3.2, 10, 128),
    ("bc_eff10", "bc_eff10_e20.pt", SEEDS9, 3.2, 10, 128),
    ("bc_eff20", "bc_eff20_e20.pt", SEEDS9, 3.2, 10, 128),
    # feature ablation: no-eta (122-dim), 9 seeds
    ("bc_noeta", "bc_noeta_e20.pt", SEEDS9, 3.2, 10, 122),
    # sensitivity curve shape: 3 seeds for e10/30/40/50 of main model
    ("bc_boost_e10",  "bc_boost_e10.pt",  SEEDS3, 3.2, 10, 128),
    ("bc_boost_e30",  "bc_boost_e30.pt",  SEEDS3, 3.2, 10, 128),
    ("bc_boost_e40",  "bc_boost_e40.pt",  SEEDS3, 3.2, 10, 128),
    ("bc_boost_e50",  "bc_boost_e50.pt",  SEEDS3, 3.2, 10, 128),
    # 20F scalability (e20, 9 seeds at 4x to keep 12H complete)
    ("bc_20f", "bc_20f_e20.pt", SEEDS9, 1.6, 20, 202),
]

def load(tag, ckpt):
    p = f"checkpoints/{ckpt}"
    if not os.path.exists(p):
        print(f"MISS {tag}: {p}"); return None
    ck = torch.load(p, map_location=dev, weights_only=False)
    sd = ck.get("policy_state", ck)
    dim_k = "encoder.weight_ih_l0" if "encoder.weight_ih_l0" in sd else "weight_ih_l0"
    odim = sd[dim_k].shape[1]
    m = build_model(odim).to(dev)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m, odim

OUT = {}
for tag, ckpt, seeds, scale, nf, odim in CONFIGS:
    loaded = load(tag, ckpt)
    if loaded is None:
        OUT[tag] = {"error": "missing ckpt"}; continue
    m, real_dim = loaded
    rs = []
    for s in seeds:
        ev = gen_events(s, scale, nf)
        env = make_env(nf, obs_eta=(real_dim >= 128))
        tot = run_policy(env, ev, m, dev)
        rs.append(tot)
        print(f"{tag} seed {s}: {tot:.1f}", flush=True)
    OUT[tag] = {"mean": float(np.mean(rs)), "std": float(np.std(rs, ddof=1)), "per_seed": rs}
    print(f"{tag}: {np.mean(rs):.1f} ± {np.std(rs, ddof=1):.1f} (n={len(rs)})", flush=True)

json.dump(OUT, open("results/eval_matrix.json", "w"), indent=1)
print("done -> results/eval_matrix.json")
