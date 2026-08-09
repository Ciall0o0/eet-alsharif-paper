# bc_margin.py — margin-aware behavior cloning (LB teacher, normal protocol, 8x)
# Modes:
#   MODE=mw  margin-weighted CE:  w = clip(margin/tau_m, 0, 1), loss = w*CE  (boundary samples downweighted)
#   MODE=sl  soft-label CE:       target = softmax(-score/tau_s), loss = KL/soft-CE
# teacher score_k = _eta(env,k,c) + LB_LAMBDA*load_ratio_k  (lower is better)
# Usage: MODE=mw TAU=3.0 OUT_TAG=bc_mw3 EPOCHS=50 SAVE_EVERY=10 .venv/bin/python /tmp/bc_margin.py
import sys, os, json, warnings
warnings.filterwarnings("ignore")
os.environ["WEIGHT_MODE"] = os.environ.get("WEIGHT_MODE", "normal")
sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, torch
import torch.nn.functional as F
from bc_boost import SEQ, gen_events, make_env, run_policy, _eta, LB_LAMBDA
from src.models.gru_ppo import GRUSharedActorCritic

dev = "cuda" if torch.cuda.is_available() else "cpu"
SEED = int(os.environ.get("SEED", "42"))
torch.manual_seed(SEED); np.random.seed(SEED)
OUT_TAG = os.environ.get("OUT_TAG", "bc_mw")
MODE = os.environ.get("MODE", "mw")          # mw | sl
TAU = float(os.environ.get("TAU", "3.0"))     # tau_m for mw, tau_s for sl
N_DEMO = int(os.environ.get("N_DEMO", "40"))
EPOCHS = int(os.environ.get("EPOCHS", "50"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "10"))
SCALE = float(os.environ.get("SCALE", "3.2"))
NF = int(os.environ.get("NF", "10"))
CUT_TIME = float(os.environ.get("CUT_TIME", "43200"))

def collect_one(seed):
    ev = gen_events(seed, SCALE, NF, train_mode="group")
    if CUT_TIME < 43200:
        ev = ev[ev[:, 2] <= CUT_TIME]
    env = make_env(NF, obs_eta=True)
    obs, _ = env.reset(options={"events": ev})
    env._ep_cap = 300000
    obs_list, act_list, score_list = [], [], []
    done, steps = False, 0
    while not done and steps < 40000:
        if env.pending_calls:
            c = env.pending_calls[0]
            sc = np.array([_eta(env, k, c) + LB_LAMBDA * env.elevators[k].load_ratio for k in range(env.num_elevators)], dtype=np.float32)
            a = int(np.argmin(sc))
            obs_list.append(obs.copy()); act_list.append(a); score_list.append(sc)
        else:
            a = 0
        obs, r, done, trunc, _ = env.step(a)
        steps += 1
        if trunc: break
    return (np.stack(obs_list), np.array(act_list, dtype=np.int64), np.stack(score_list))

import multiprocessing as mp
n_workers = min(int(os.environ.get("N_WORKERS", "8")), mp.cpu_count())
demo_seeds = [2000 + i * 13 for i in range(N_DEMO)]
demos = []
with mp.Pool(n_workers) as pool:
    for i, res in enumerate(pool.imap_unordered(collect_one, demo_seeds)):
        demos.append(res)
        if (i + 1) % 10 == 0:
            print(f"collected {i+1}/{N_DEMO} demos", flush=True)
print(f"total decisions: {sum(len(d[1]) for d in demos)}", flush=True)

state_dim = demos[0][0].shape[1]
model = GRUSharedActorCritic(
    state_dim=state_dim, action_dim=3, aux_prediction=True, num_dest_classes=10,
    use_layer_norm=True, gru_hidden=256, gru_layers=2, gru_dropout=0.1,
    actor_hidden=64, critic_hidden=64, dest_head_on=False, event_head_on=True,
    reward_change_on=False,
).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

def make_targets(act_arr, score_arr):
    """Return (target_idx, weights) for mw, or (soft_target, None) for sl."""
    if MODE == "mw":
        # margin = second-smallest - smallest (teacher confidence); w in [0,1]
        s = np.sort(score_arr, axis=-1)
        margin = s[:, 1] - s[:, 0]
        w = np.clip(margin / TAU, 0.0, 1.0).astype(np.float32)
        return act_arr, w
    else:  # sl: softmax(-score/tau) over cars — lower score -> higher prob
        soft = np.exp(-(score_arr - score_arr.min(axis=-1, keepdims=True)) / TAU)
        soft /= soft.sum(axis=-1, keepdims=True)
        return soft.astype(np.float32), None

curve = []
for epoch in range(EPOCHS):
    model.train()
    tl, tn = 0.0, 0
    for obs_arr, act_arr, score_arr in demos:
        n = len(obs_arr) - (len(obs_arr) % SEQ)
        for s in range(0, n, SEQ):
            obs_t = torch.as_tensor(obs_arr[s:s+SEQ], dtype=torch.float32, device=dev).unsqueeze(0)
            hidden = model.get_initial_hidden(1, dev)
            out = model.forward(obs_t, hidden)
            logits = out[0].squeeze(0)
            if MODE == "mw":
                act_t = torch.as_tensor(act_arr[s:s+SEQ], dtype=torch.int64, device=dev)
                _, w = make_targets(act_arr[s:s+SEQ], score_arr[s:s+SEQ])
                w_t = torch.as_tensor(w, dtype=torch.float32, device=dev)
                ce = F.cross_entropy(logits, act_t, reduction="none")
                loss = (ce * w_t).mean()
            else:
                soft, _ = make_targets(act_arr[s:s+SEQ], score_arr[s:s+SEQ])
                soft_t = torch.as_tensor(soft, dtype=torch.float32, device=dev)
                loss = -(soft_t * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); tn += SEQ
    model.eval()
    acc_ok, acc_n = 0, 0
    with torch.no_grad():
        for obs_arr, act_arr, score_arr in demos:
            n = len(obs_arr) - (len(obs_arr) % SEQ)
            for s in range(0, n, SEQ):
                obs_t = torch.as_tensor(obs_arr[s:s+SEQ], dtype=torch.float32, device=dev).unsqueeze(0)
                hidden = model.get_initial_hidden(1, dev)
                out = model.forward(obs_t, hidden)
                pred = out[0].squeeze(0).argmax(-1)
                acc_ok += (pred.cpu().numpy() == act_arr[s:s+SEQ]).sum(); acc_n += SEQ
    curve.append({"epoch": epoch+1, "loss": tl/max(tn,1), "acc": acc_ok/max(acc_n,1)})
    print(f"{MODE}(tau={TAU}) ep{epoch+1}: loss={tl/max(tn,1):.4f} acc={acc_ok/max(acc_n,1):.4f}", flush=True)
    if SAVE_EVERY and (epoch + 1) % SAVE_EVERY == 0:
        torch.save({"policy_state": model.state_dict()}, f"checkpoints/{OUT_TAG}_e{epoch+1}.pt")

json.dump(curve, open(f"results/{OUT_TAG}_curve.json", "w"), indent=1)
print(f"done -> checkpoints/{OUT_TAG}_e*.pt")
