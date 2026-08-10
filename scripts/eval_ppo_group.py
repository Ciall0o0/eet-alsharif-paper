# eval_ppo_group.py — 9-seed eval of fair group-arrival PPO (GRUSharedActorCritic load)
# Usage: TAG=ppo_group_fair_s42 CKPT=checkpoints/ppo_group_fair_s42/ppo_elevator_best.pt \
#        SCALE=3.2 .venv/bin/python /tmp/eval_ppo_group.py
import sys, os, json, warnings
warnings.filterwarnings("ignore")
os.environ["WEIGHT_MODE"] = "normal"
sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, torch
from bc_boost import gen_events, make_env, run_policy
from src.models.gru_ppo import GRUSharedActorCritic

dev = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS9 = (9999, 10001, 10003, 10005, 10007, 10009, 10011, 10013, 10015)
TAG = os.environ.get("TAG", "ppo_group_fair_s42")
CKPT = os.environ.get("CKPT", "checkpoints/ppo_group_fair_s42/ppo_elevator_best.pt")
SCALE = float(os.environ.get("SCALE", "3.2"))
OUTF = os.environ.get("OUTF", f"results/{TAG}_eval.json")

ck = torch.load(CKPT, map_location=dev, weights_only=False)
sd = ck.get("policy_state", ck)
n_in = sd["encoder.weight_ih_l0"].shape[1]
m = GRUSharedActorCritic(state_dim=n_in, action_dim=3, aux_prediction=False,
                         num_dest_classes=10, use_layer_norm=True, gru_hidden=256,
                         gru_layers=2, gru_dropout=0.1, actor_hidden=64, critic_hidden=64).to(dev)
m.load_state_dict(sd, strict=False)
m.eval()
print(f"loaded {CKPT} state_dim={n_in}", flush=True)

rs = []
for s in SEEDS9:
    ev = gen_events(s, SCALE, 10, train_mode="group")
    env = make_env(10, obs_eta=True)
    tot = run_policy(env, ev, m, dev)
    rs.append(tot)
    print(f"{TAG} seed {s}: {tot:.1f}", flush=True)
print(f"{TAG}: {np.mean(rs):.1f} ± {np.std(rs, ddof=1):.1f}", flush=True)
json.dump({"mean": float(np.mean(rs)), "std": float(np.std(rs, ddof=1)), "per": rs},
          open(OUTF, "w"), indent=1)
print("done ->", OUTF)
