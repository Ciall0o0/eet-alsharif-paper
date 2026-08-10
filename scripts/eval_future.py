# eval_future.py — 9-seed eval of BC with oracle future features (env built future_features=True)
import sys, os, json, warnings
warnings.filterwarnings("ignore")
os.environ["WEIGHT_MODE"] = "normal"
sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, torch
from bc_boost import build_model, gen_events, make_env, run_policy

dev = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS9 = (9999, 10001, 10003, 10005, 10007, 10009, 10011, 10013, 10015)
TAG = os.environ.get("TAG", "bc_future")
EP = int(os.environ.get("EP", "20"))
SCALE = float(os.environ.get("SCALE", "3.2"))

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

rs = []
for s in SEEDS9:
    ev = gen_events(s, SCALE, 10, train_mode="group")
    env = make_env(10, obs_eta=True, future_features=True)
    tot = run_policy(env, ev, m, dev)
    rs.append(tot)
    print(f"{TAG} e{EP} seed {s}: {tot:.1f}", flush=True)
print(f"{TAG} e{EP}: {np.mean(rs):.1f} ± {np.std(rs, ddof=1):.1f}", flush=True)
json.dump({"20": {"mean": float(np.mean(rs)), "std": float(np.std(rs, ddof=1)), "per": rs}},
          open(f"results/{TAG}_9seed.json", "w"), indent=1)
print("done ->", f"results/{TAG}_9seed.json")
