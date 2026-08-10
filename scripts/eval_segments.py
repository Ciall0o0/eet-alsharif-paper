# eval_segments.py — time-of-day segmented evaluation (per DAILY_SCHEDULE_12H 7 segments)
# Reports per-segment mean/p95 wait + pax count for all 8 methods @ 8x, 9 held-out seeds.
import sys, os, json, warnings
warnings.filterwarnings("ignore")
os.environ["WEIGHT_MODE"] = "normal"
sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, torch
from bc_boost import build_model, gen_events, make_env, _eta, LB_LAMBDA
from src.models.gru_ppo import GRUSharedActorCritic

dev = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = (9999, 10001, 10003, 10005, 10007, 10009, 10011, 10013, 10015)
SCALE = float(os.environ.get("SCALE", "3.2"))

SCHEDULE = [("up_peak", 0, 90), ("interfloor_am", 90, 150), ("lunch_peak", 150, 210),
            ("interfloor_mid", 210, 390), ("down_peak", 390, 480), ("off_peak", 480, 600),
            ("interfloor_pm", 600, 720)]

def seg_of(tmin):
    for name, a, b in SCHEDULE:
        if a <= tmin < b:
            return name
    return "edge"

def load_bc(path):
    ck = torch.load(path, map_location=dev, weights_only=False)
    sd = ck.get("policy_state", ck)
    dim_k = "encoder.weight_ih_l0" if "encoder.weight_ih_l0" in sd else "weight_ih_l0"
    m = build_model(sd[dim_k].shape[1]).to(dev)
    m.load_state_dict(sd, strict=False); m.eval()
    return m

def load_ppo(path):
    ck = torch.load(path, map_location=dev, weights_only=False)
    sd = ck.get("policy_state", ck)
    n_in = sd["encoder.weight_ih_l0"].shape[1]
    m = GRUSharedActorCritic(state_dim=n_in, action_dim=3, aux_prediction=False,
                             num_dest_classes=10, use_layer_norm=True, gru_hidden=256,
                             gru_layers=2, gru_dropout=0.1, actor_hidden=64, critic_hidden=64).to(dev)
    m.load_state_dict(sd, strict=False); m.eval()
    return m

MODELS = {
    "bc_lb": load_bc("checkpoints/bc_lb_norm_e20.pt"),
    "bc_sdeta": load_bc("checkpoints/bc_sdeta_norm_e20.pt"),
    "ppo": load_ppo("checkpoints/ppo_normal_s42/ppo_elevator_best.pt"),
}

def run_one(env, events, method):
    obs, _ = env.reset(options={"events": events})
    env._ep_cap = 300000
    h = None
    m = MODELS.get(method)
    if m is not None:
        h = m.get_initial_hidden(1, dev)
    done, steps = False, 0
    while not done and steps < 200000:
        if env.pending_calls:
            if m is not None:
                ot = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    a, _, _, h, _ = m.get_action(ot, h, deterministic=True)
                a = int(a.item())
            else:
                c = env.pending_calls[0]
                if method == "sd_eta":
                    a = min(range(3), key=lambda k: _eta(env, k, c))
                elif method == "sd_nearest":
                    a = min(range(3), key=lambda k: abs(env.elevators[k].current_floor - c["floor"]) +
                           (1.5 if env.elevators[k].direction not in (0, 0) else 0.0))
                elif method == "lb15":
                    a = min(range(3), key=lambda k: _eta(env, k, c) + 15.0 * env.elevators[k].load_ratio)
                elif method == "lb30":
                    a = min(range(3), key=lambda k: _eta(env, k, c) + LB_LAMBDA * env.elevators[k].load_ratio)
                else:  # sector
                    c = env.pending_calls[0]
                    a = min(c["floor"] // 4, 2)
        else:
            a = 0
        obs, r, done, trunc, _ = env.step(a)
        steps += 1
        if trunc:
            break
    # aggregate waits by segment (served passengers only, consistent with main table)
    seg_waits = {name: [] for name, *_ in SCHEDULE}
    for p in env.completed_passengers:
        tmin = p.get("delivered_at", 0.0) / 60.0
        s = seg_of(tmin)
        if s in seg_waits:
            seg_waits[s].append(p["wait_time"])
    return seg_waits

METHODS = ["sd_nearest", "sd_eta", "lb15", "lb30", "bc_lb", "bc_sdeta", "ppo", "sector"]
OUT = {}
for method in METHODS:
    agg = {name: {"waits": [], "n": 0} for name, *_ in SCHEDULE}
    for s in SEEDS:
        ev = gen_events(s, SCALE, 10, train_mode="group")
        env = make_env(10, obs_eta=True)
        seg_waits = run_one(env, ev, method)
        for name in agg:
            agg[name]["waits"].extend(seg_waits[name])
            agg[name]["n"] += len(seg_waits[name])
        print(f"{method} seed {s} done", flush=True)
    OUT[method] = {}
    for name, *_ in SCHEDULE:
        w = np.array(agg[name]["waits"])
        OUT[method][name] = {
            "n": agg[name]["n"],
            "mean": float(np.mean(w)) if len(w) else float("nan"),
            "p95": float(np.percentile(w, 95)) if len(w) else float("nan"),
        }
    print(f"{method}: " + " ".join(f"{n}={OUT[method][n]['mean']:.1f}s({OUT[method][n]['n']})" for n, *_ in SCHEDULE), flush=True)

json.dump(OUT, open("results/segments_8x.json", "w"), indent=1)
print("done -> results/segments_8x.json")
