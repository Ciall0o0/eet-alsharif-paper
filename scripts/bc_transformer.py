# bc_transformer.py — BC with causal Transformer encoder (vs GRU architecture ablation)
# Same protocol as bc_margin/bc_boost: 40 demos LB teacher, 8x group, normal weights,
# seq-64 chunked CE, 50 epochs, eval 9 held-out seeds with sliding window.
import sys, os, warnings
warnings.filterwarnings("ignore")
os.environ["WEIGHT_MODE"] = "normal"
sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, torch
import torch.nn.functional as F
from bc_boost import gen_events, make_env, _eta, LB_LAMBDA
sys.path.insert(0, "/tmp")
from transformer_ppo import TFEncoderActorCritic

NF = int(os.environ.get("NF", "10"))
SEQ = 64
WINDOW = 64

def collect_demo(env, events, max_steps=40000):
    obs, _ = env.reset(options={"events": events})
    env._ep_cap = 300000
    obs_list, act_list = [], []
    done, steps = False, 0
    while not done and steps < max_steps:
        if env.pending_calls:
            c = env.pending_calls[0]
            a = min(range(env.num_elevators),
                    key=lambda k: _eta(env, k, c) + LB_LAMBDA * env.elevators[k].load_ratio)
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
    OUT_TAG = os.environ.get("OUT_TAG", "bc_tf")
    n_demo = int(os.environ.get("N_DEMO", "40"))
    TRAIN_SCALE = float(os.environ.get("SCALE", "3.2"))
    import multiprocessing as mp
    n_workers = min(int(os.environ.get("N_WORKERS", "8")), mp.cpu_count())

    def _collect_one(args):
        seed, i = args
        ev = gen_events(seed, TRAIN_SCALE, NF, train_mode="group")
        wseed = (7 + i % 3)
        env = make_env(NF, obs_eta=True, weight_seed=wseed)
        return collect_demo(env, ev)

    demo_seeds = [2000 + i * 13 for i in range(n_demo)]
    demos = []
    with mp.Pool(n_workers) as pool:
        for i, (obs_arr, act_arr) in enumerate(pool.imap_unordered(_collect_one,
                                                                   list(zip(demo_seeds, range(n_demo))))):
            demos.append((obs_arr, act_arr))
    print(f"demos={n_demo} decisions={sum(len(o) for o,_ in demos)} obs_dim={demos[0][0].shape[1]}", flush=True)

    state_dim = demos[0][0].shape[1]
    model = TFEncoderActorCritic(state_dim=state_dim, action_dim=3, d_model=256, nhead=8,
                                 n_layers=3, dim_feedforward=512, max_seq=SEQ).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TF params: {n_params/1e6:.2f}M", flush=True)
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
                logits, *_ = model.forward(obs_t, None)
                loss = F.cross_entropy(logits.squeeze(0), act_t)
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item(); total_n += SEQ
        model.eval()
        acc_ok, acc_n = 0, 0
        with torch.no_grad():
            for obs_arr, act_arr in demos:
                n = len(obs_arr) - (len(obs_arr) % SEQ)
                for s in range(0, n, SEQ):
                    obs_t = torch.as_tensor(obs_arr[s:s+SEQ], dtype=torch.float32, device=dev).unsqueeze(0)
                    act_t = torch.as_tensor(act_arr[s:s+SEQ], dtype=torch.int64, device=dev)
                    logits, *_ = model.forward(obs_t, None)
                    pred = logits.squeeze(0).argmax(-1)
                    acc_ok += (pred == act_t).sum().item()
                    acc_n += SEQ
        acc = acc_ok / acc_n
        if (epoch + 1) % 5 == 0 or epoch in (0, n_epochs - 1):
            print(f"TF ep{epoch+1}: loss={total_loss/max(total_n,1):.4f} acc={acc:.4f}", flush=True)
        if (epoch + 1) % 10 == 0:
            torch.save({"policy_state": model.state_dict(), "model_type": "tf"},
                       f"checkpoints/{OUT_TAG}_e{epoch+1}.pt")
            print(f"  saved {OUT_TAG}_e{epoch+1}.pt", flush=True)
    print("training done", flush=True)
