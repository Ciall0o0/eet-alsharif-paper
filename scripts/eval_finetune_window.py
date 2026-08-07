import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, torch
from bc_boost import build_model, gen_events, make_env, run_policy

dev = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = (9999, 10001, 10003, 10005, 10007, 10009, 10011, 10013, 10015)

def load(path):
    ck = torch.load(path, map_location=dev, weights_only=False)
    sd = ck.get("policy_state", ck)
    m = build_model(sd["encoder.weight_ih_l0"].shape[1]).to(dev)
    m.load_state_dict(sd, strict=False); m.eval()
    return m

OUT = {}
for tag, path in [("ft2_lr1e5", "checkpoints/probe_ft2_lr1e5_s42/ppo_elevator_best.pt"),
                  ("ft3_lr3e5", "checkpoints/probe_ft3_lr3e5_s42/ppo_elevator_best.pt")]:
    m = load(path)
    rs = []
    for s in SEEDS:
        ev = gen_events(s, 3.2); env = make_env()
        rs.append(run_policy(env, ev, m, dev))
    OUT[tag] = [float(np.mean(rs)), float(np.std(rs, ddof=1)), rs]
    print(f"{tag}: {np.mean(rs):.1f} ± {np.std(rs, ddof=1):.1f}  {[round(x,1) for x in rs]}", flush=True)
json.dump(OUT, open("results/finetune_window_eval.json", "w"), indent=1)
print("done")
