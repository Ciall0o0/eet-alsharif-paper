# eval_tagru.py — 9-seed eval of TA-GRU checkpoints (loads with TimeGateGRUSharedActorCritic)
import sys, os, json, warnings
warnings.filterwarnings("ignore")
os.environ["WEIGHT_MODE"] = "normal"
sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, torch
from bc_boost import gen_events, make_env, run_policy
from src.models.gru_ppo import TimeGateGRUSharedActorCritic

dev = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS9 = (9999, 10001, 10003, 10005, 10007, 10009, 10011, 10013, 10015)
TAG = os.environ.get("TAG", "bc_tagru_t60")
TAU = float(os.environ.get("TAU_GATE", "60.0"))

def load(ep):
    p = f"checkpoints/{TAG}_e{ep}.pt"
    if not os.path.exists(p):
        return None
    ck = torch.load(p, map_location=dev, weights_only=False)
    sd = ck.get("policy_state", ck)
    dim_k = "encoder.weight_ih_l0" if "encoder.weight_ih_l0" in sd else "weight_ih_l0"
    odim = sd[dim_k].shape[1]
    m = TimeGateGRUSharedActorCritic(
        state_dim=odim, action_dim=3, aux_prediction=True, num_dest_classes=10,
        use_layer_norm=True, gru_hidden=256, gru_layers=2, gru_dropout=0.1,
        actor_hidden=64, critic_hidden=64, dest_head_on=False, event_head_on=True,
        reward_change_on=False, tau_gate=TAU, elapsed_idx=119, t_max=43200.0,
    ).to(dev)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m

OUT = {}
for ep in (10, 20, 30, 40, 50):
    m = load(ep)
    if m is None:
        continue
    rs = []
    for s in SEEDS9:
        ev = gen_events(s, 3.2, 10)
        env = make_env(10, obs_eta=True)
        tot = run_policy(env, ev, m, dev)
        rs.append(tot)
        print(f"{TAG} e{ep} seed {s}: {tot:.1f}", flush=True)
    OUT[str(ep)] = {"mean": float(np.mean(rs)), "std": float(np.std(rs, ddof=1)), "per": rs}
    print(f"{TAG} e{ep}: {np.mean(rs):.1f} ± {np.std(rs, ddof=1):.1f}", flush=True)

json.dump(OUT, open(f"results/{TAG}_9seed.json", "w"), indent=1)
print("done ->", f"results/{TAG}_9seed.json")
