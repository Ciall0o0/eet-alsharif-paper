# eval_sl_20f.py — SL-BC 20F transfer eval: train on 10F, eval on 20F @ 4x/8x
# Usage: TAG=bc_sl1 EP=20 .venv/bin/python /tmp/eval_sl_20f.py
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
OUTF = os.environ.get("OUTF", f"results/{TAG}_20f_eval.json")

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
for scale, tag in ((1.6, "4x"), (3.2, "8x")):
    rs = []
    for s in SEEDS9:
        ev = gen_events(s, scale, 20)
        env = make_env(20, obs_eta=True)
        tot = run_policy(env, ev, m, dev)
        rs.append(tot)
        print(f"{TAG} e{EP} 20F {tag} seed {s}: {tot:.1f}", flush=True)
    OUT[tag] = {"mean": float(np.mean(rs)), "std": float(np.std(rs, ddof=1)), "per": rs}
    print(f"{TAG} e{EP} 20F {tag}: {np.mean(rs):.1f} ± {np.std(rs, ddof=1):.1f}", flush=True)

json.dump(OUT, open(OUTF, "w"), indent=1)
print("done ->", OUTF)
