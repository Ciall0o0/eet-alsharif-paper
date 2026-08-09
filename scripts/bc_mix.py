# bc_mix.py — multi-teacher BC: mix LB-30 and SD-ETA teachers
# MIX_MODE=rand (50/50 per-state random) | gate (load_max > GATE_TH -> LB else ETA)
# Usage: MIX_MODE=gate GATE_TH=0.6 N_DEMO=40 EPOCHS=50 SAVE_EVERY=10 OUT_TAG=bc_mix_gate SEED=42 .venv/bin/python /tmp/bc_mix.py
import sys, os, warnings
warnings.filterwarnings("ignore")
os.environ["WEIGHT_MODE"] = "normal"
sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, torch
import torch.nn.functional as F
from bc_boost import gen_events, make_env, _eta, build_model, LB_LAMBDA

NF = int(os.environ.get("NF", "10"))
SEQ = 64
MIX_MODE = os.environ.get("MIX_MODE", "rand")
GATE_TH = float(os.environ.get("GATE_TH", "0.6"))
LAM = float(os.environ.get("LB_LAMBDA", "30"))
CUT_TIME = float(os.environ.get("CUT_TIME", "43200"))
WEIGHT_MODE = os.environ.get("WEIGHT_MODE", "normal")

def teacher_action(env, c):
    """Return (action, teacher_id) under the selected mix mode."""
    if MIX_MODE == "rand":
        if np.random.rand() < 0.5:
            return min(range(env.num_elevators),
                       key=lambda k: _eta(env, k, c) + LAM * env.elevators[k].load_ratio), "lb"
        return min(range(env.num_elevators), key=lambda k: _eta(env, k, c)), "eta"
    # gate
    load_max = max(e.load_ratio for e in env.elevators)
    if load_max > GATE_TH:
        return min(range(env.num_elevators),
                   key=lambda k: _eta(env, k, c) + LAM * env.elevators[k].load_ratio), "lb"
    return min(range(env.num_elevators), key=lambda k: _eta(env, k, c)), "eta"

def collect_demo(env, events, max_steps=40000):
    obs, _ = env.reset(options={"events": events})
    env._ep_cap = 300000
    obs_list, act_list = [], []
    done, steps = False, 0
    while not done and steps < max_steps:
        if env.pending_calls:
            a, _ = teacher_action(env, env.pending_calls[0])
            obs_list.append(obs.copy())
            act_list.append(a)
        else:
            a = 0
        obs, r, done, trunc, _ = env.step(a)
        steps += 1
        if trunc:
            break
    return np.array(obs_list), np.array(act_list)

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = int(os.environ.get("SEED", "42"))
    torch.manual_seed(SEED); np.random.seed(SEED)
    OUT_TAG = os.environ.get("OUT_TAG", "bc_mix")
    n_demo = int(os.environ.get("N_DEMO", "40"))
    TRAIN_SCALE = float(os.environ.get("SCALE", "3.2"))
    import multiprocessing as mp
    n_workers = min(int(os.environ.get("N_WORKERS", "8")), mp.cpu_count())

    def _collect_one(args):
        seed, i = args
        ev = gen_events(seed, TRAIN_SCALE, NF, train_mode="group")
        if CUT_TIME < 43200:
            ev = ev[ev[:, 2] <= CUT_TIME]
        wseed = (7 + i % 3) if WEIGHT_MODE == "normal" else None
        env = make_env(NF, obs_eta=True, weight_seed=wseed)
        return collect_demo(env, ev)

    demo_seeds = [2000 + i * 13 for i in range(n_demo)]
    demos = []
    with mp.Pool(n_workers) as pool:
        for i, (obs_arr, act_arr) in enumerate(pool.imap_unordered(_collect_one,
                                                                   list(zip(demo_seeds, range(n_demo))))):
            demos.append((obs_arr, act_arr))
            if (i + 1) % 10 == 0:
                print(f"collected {i+1}/{n_demo} demos (last {len(obs_arr)} decisions)", flush=True)
    print(f"MIX_MODE={MIX_MODE} GATE_TH={GATE_TH} demos={n_demo} decisions={sum(len(o) for o,_ in demos)}", flush=True)

    state_dim = demos[0][0].shape[1]
    model = build_model(state_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    n_epochs = int(os.environ.get("EPOCHS", "50"))
    for epoch in range(n_epochs):
        model.train()
        total_loss, total_n = 0.0, 0
        for obs_arr, act_arr in demos:
            n = len(obs_arr) - (len(obs_arr) % SEQ)
            for s in range(0, n, SEQ):
                obs_t = torch.as_tensor(obs_arr[s:s+SEQ], dtype=torch.float32, device=dev).unsqueeze(0)
                act_t = torch.as_tensor(act_arr[s:s+SEQ], dtype=torch.int64, device=dev)
                hidden = model.get_initial_hidden(1, dev)
                out, _, _, _, _, _ = model.forward(obs_t, hidden)
                loss = F.cross_entropy(out.squeeze(0), act_t)
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item(); total_n += SEQ
        model.eval()
        acc_ok, acc_n = 0, 0
        with torch.no_grad():
            for obs_arr, act_arr in demos:
                n = len(obs_arr) - (len(obs_arr) % SEQ)
                obs_t = torch.as_tensor(obs_arr[:n], dtype=torch.float32, device=dev).unsqueeze(0)
                out, _, _, _, _, _ = model.forward(obs_t, model.get_initial_hidden(1, dev))
                pred = out.squeeze(0).argmax(-1)
                acc_ok += (pred == torch.as_tensor(act_arr[:n], device=dev)).sum().item()
                acc_n += n
        acc = acc_ok / acc_n
        if (epoch + 1) % 5 == 0 or epoch in (0, n_epochs - 1):
            print(f"BC ep{epoch+1}: loss={total_loss/max(total_n,1):.4f} acc={acc:.4f}", flush=True)
        if (epoch + 1) % 10 == 0:
            torch.save({"policy_state": model.state_dict(), "aux_prediction": True,
                        "event_head_on": True}, f"checkpoints/{OUT_TAG}_e{epoch+1}.pt")
            print(f"  saved {OUT_TAG}_e{epoch+1}.pt", flush=True)
    print("training done", flush=True)
