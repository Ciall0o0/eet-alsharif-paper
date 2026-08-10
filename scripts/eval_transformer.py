# eval_transformer.py — 9-seed eval of TF-encoder BC with decision-step sliding window
import sys, os, json, warnings
warnings.filterwarnings("ignore")
os.environ["WEIGHT_MODE"] = "normal"
sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, torch
from collections import deque
from bc_boost import gen_events, make_env
sys.path.insert(0, "/tmp")
from transformer_ppo import TFEncoderActorCritic

dev = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS9 = (9999, 10001, 10003, 10005, 10007, 10009, 10011, 10013, 10015)
TAG = os.environ.get("TAG", "bc_tf")
EP = int(os.environ.get("EP", "20"))
SCALE = float(os.environ.get("SCALE", "3.2"))
WINDOW = 64

p = f"checkpoints/{TAG}_e{EP}.pt"
assert os.path.exists(p), f"missing {p}"
ck = torch.load(p, map_location=dev, weights_only=False)
sd = ck.get("policy_state", ck)
# infer state dim from projection layer
n_in = sd["proj.weight"].shape[1]
m = TFEncoderActorCritic(state_dim=n_in, action_dim=3, d_model=256, nhead=8,
                         n_layers=3, dim_feedforward=512, max_seq=WINDOW).to(dev)
m.load_state_dict(sd, strict=False)
m.eval()
print(f"loaded {p} state_dim={n_in}", flush=True)

def run_episode(seed):
    ev = gen_events(seed, SCALE, 10, train_mode="group")
    env = make_env(10, obs_eta=True)
    obs, _ = env.reset(options={"events": ev})
    env._ep_cap = 300000
    win = deque(maxlen=WINDOW)
    done, steps, tot = False, 0, 0.0
    while not done and steps < 200000:
        if env.pending_calls:
            win.append(obs.copy())
            wt = torch.as_tensor(np.stack(win), dtype=torch.float32, device=dev).unsqueeze(0)
            with torch.no_grad():
                logits, *_ = m.forward(wt, None)
                a = int(logits[0, -1].argmax().item())
        else:
            a = 0
        obs, r, done, trunc, _ = env.step(a)
        tot += r
        steps += 1
        if trunc:
            break
    return tot

OUT = {}
rs = []
for s in SEEDS9:
    tot = run_episode(s)
    rs.append(tot)
    print(f"{TAG} e{EP} seed {s}: {tot:.1f}", flush=True)
OUT[str(EP)] = {"mean": float(np.mean(rs)), "std": float(np.std(rs, ddof=1)), "per": rs}
print(f"{TAG} e{EP}: {np.mean(rs):.1f} ± {np.std(rs, ddof=1):.1f}", flush=True)
json.dump(OUT, open(f"results/{TAG}_9seed.json", "w"), indent=1)
print("done ->", f"results/{TAG}_9seed.json")
