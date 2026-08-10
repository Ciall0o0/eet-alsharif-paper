# eval_val_tau_curve.py — full-curve val-seed eval for the three SL temperatures
import sys, os, json, warnings
warnings.filterwarnings("ignore")
os.environ["WEIGHT_MODE"] = "normal"
sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, torch
from bc_boost import build_model, gen_events, make_env, run_policy

dev = "cuda" if torch.cuda.is_available() else "cpu"
VALSEEDS = (10002, 10004, 10006)

def load(tag, ep):
    p = f"checkpoints/{tag}_e{ep}.pt"
    if not os.path.exists(p):
        return None
    ck = torch.load(p, map_location=dev, weights_only=False)
    sd = ck.get("policy_state", ck)
    dim_k = "encoder.weight_ih_l0" if "encoder.weight_ih_l0" in sd else "weight_ih_l0"
    m = build_model(sd[dim_k].shape[1]).to(dev)
    m.load_state_dict(sd, strict=False); m.eval()
    return m

OUT = {}
for tag in ("bc_sl05", "bc_sl1", "bc_sl2"):
    OUT[tag] = {}
    for ep in (10, 20, 30, 40, 50):
        m = load(tag, ep)
        if m is None:
            continue
        rs = []
        for s in VALSEEDS:
            ev = gen_events(s, 3.2, 10, train_mode="group")
            env = make_env(10, obs_eta=True)
            rs.append(run_policy(env, ev, m, dev))
        OUT[tag][str(ep)] = {"mean": float(np.mean(rs)), "per": rs}
        print(f"{tag} e{ep} val: {np.mean(rs):.1f}", flush=True)
    row = " | ".join(f"e{k}:{OUT[tag][k]['mean']:.0f}" for k in ("10","20","30","40","50") if k in OUT[tag])
    print(f"{tag}: {row}", flush=True)

json.dump(OUT, open("results/val_tau_curve.json", "w"), indent=1)
print("done -> results/val_tau_curve.json")
