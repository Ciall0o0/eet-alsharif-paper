# eval_density.py — eval a BC checkpoint at multiple densities (4x/8x/12x), 9 seeds
# Usage: TAG=bc_sl1 SCALES=1.6,3.2,4.8 .venv/bin/python /tmp/eval_density.py
import sys, os, json, warnings
warnings.filterwarnings("ignore")
os.environ["WEIGHT_MODE"] = "normal"
sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, torch
from bc_boost import build_model, gen_events, make_env, run_policy

dev = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS9 = (9999, 10001, 10003, 10005, 10007, 10009, 10011, 10013, 10015)
TAG = os.environ.get("TAG", "bc_sl1")
EP = int(os.environ.get("EP", "20"))
SCALES = [float(x) for x in os.environ.get("SCALES", "1.6,4.8").split(",")]
OUTF = os.environ.get("OUTF", f"results/{TAG}_density_eval.json")

p = f"checkpoints/{TAG}_e{EP}.pt"
assert os.path.exists(p), f"missing {p}"
ck = torch.load(p, map_location=dev, weights_only=False)
sd = ck.get("policy_state", ck)
dim_k = "encoder.weight_ih_l0" if "encoder.weight_ih_l0" in sd else "weight_ih_l0"
odim = sd[dim_k].shape[1]
m = build_model(odim).to(dev)
m.load_state_dict(sd, strict=False)
m.eval()
print(f"loaded {p} odim={odim}", flush=True)

OUT = {}
for scale in SCALES:
    rs = []
    for s in SEEDS9:
        ev = gen_events(s, scale, 10)
        env = make_env(10, obs_eta=True)
        tot = run_policy(env, ev, m, dev)
        rs.append(tot)
        print(f"{TAG} e{EP} scale={scale} seed {s}: {tot:.1f}", flush=True)
    key = {1.6: "4x", 3.2: "8x", 4.8: "12x"}.get(scale, f"{scale:g}")
    OUT[key] = {"mean": float(np.mean(rs)), "std": float(np.std(rs, ddof=1)), "per": rs}
    print(f"{TAG} e{EP} scale={scale}: {np.mean(rs):.1f} ± {np.std(rs, ddof=1):.1f}", flush=True)

json.dump(OUT, open(OUTF, "w"), indent=1)
print("done ->", OUTF)
