# eval_full_metrics.py — full service-quality metrics (R6 Major 5): reward, service rate,
# unserved, all-passenger wait (mean/median/p95/p99), ride, max queue, utilization,
# empty travel, energy. RULES_ONLY=1 skips model loading (CPU-only rule methods).
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
CKPTS = {
    "bc_lb": "checkpoints/bc_lb_norm_e20.pt",
    "bc_sdeta": "checkpoints/bc_sdeta_norm_e20.pt",
    "ppo": "checkpoints/ppo_normal_s42/ppo_elevator_best.pt",
}

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

RULES_ONLY = os.environ.get("RULES_ONLY", "0") == "1"
MODELS = {} if RULES_ONLY else {"bc_lb": load_bc(CKPTS["bc_lb"]),
                                "bc_sdeta": load_bc(CKPTS["bc_sdeta"]),
                                "ppo": load_ppo(CKPTS["ppo"])}

def rule_act(env, method):
    c = env.pending_calls[0]
    if method == "sd_eta":
        return min(range(3), key=lambda k: _eta(env, k, c))
    if method == "sd_nearest":
        return min(range(3), key=lambda k: abs(env.elevators[k].current_floor - c["floor"]) +
                   (1.5 if env.elevators[k].direction not in (0, 0) else 0.0))
    lam = {"lb15": 15.0, "lb30": 30.0}[method]
    return min(range(3), key=lambda k: _eta(env, k, c) + lam * env.elevators[k].load_ratio)

def run_one(env, events, method):
    obs, _ = env.reset(options={"events": events})
    env._ep_cap = 300000
    h = None
    m = MODELS.get(method)
    if m is not None:
        h = m.get_initial_hidden(1, dev)
    done, steps, tot, qmax = False, 0, 0.0, 0
    while not done and steps < 200000:
        if env.pending_calls:
            qmax = max(qmax, len(env.pending_calls))
            if m is not None:
                ot = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    a, _, _, h, _ = m.get_action(ot, h, deterministic=True)
                a = int(a.item())
            elif method == "sector":
                a = min(env.pending_calls[0]["floor"] // 4, 2)
            else:
                a = rule_act(env, method)
        else:
            a = 0
        obs, r, done, trunc, _ = env.step(a)
        tot += r; steps += 1
        if trunc:
            break
    cp = env.completed_passengers
    waits = np.array([p["wait_time"] for p in cp]) if cp else np.array([])
    rides = np.array([p["ride_time"] for p in cp]) if cp else np.array([])
    em = env.get_episode_metrics()
    total_pax = int(np.asarray(events)[:, 3].sum()) if events is not None and len(events) else 0
    served = len(cp)
    util = em.get("loaded_movement_floors", 0) / max(em.get("empty_movement_floors", 0) + em.get("loaded_movement_floors", 0), 1)
    return {
        "reward": tot, "served": served, "total_pax": total_pax,
        "serve_frac": served / max(total_pax, 1), "unserved": total_pax - served,
        "wait_mean": float(np.mean(waits)) if len(waits) else float("nan"),
        "wait_median": float(np.median(waits)) if len(waits) else float("nan"),
        "wait_p95": float(np.percentile(waits, 95)) if len(waits) else float("nan"),
        "wait_p99": float(np.percentile(waits, 99)) if len(waits) else float("nan"),
        "ride_mean": float(np.mean(rides)) if len(rides) else float("nan"),
        "queue_max": int(qmax),
        "util": float(util),
        "empty_floors": em.get("empty_movement_floors", 0),
        "wh": em.get("Wh", em.get("energy_wh", 0.0)),
        "starts": em.get("start_stop_count", 0),
    }

ALL_METHODS = ["sd_nearest", "sd_eta", "lb15", "lb30", "bc_lb", "bc_sdeta", "ppo", "sector"]
METHODS = ["sd_nearest", "sd_eta", "lb15", "lb30", "sector"] if RULES_ONLY else ALL_METHODS
OUT = {}
for method in METHODS:
    rows = []
    for s in SEEDS:
        ev = gen_events(s, SCALE, 10, train_mode="group")
        env = make_env(10, obs_eta=True)
        rows.append(run_one(env, ev, method))
        print(f"{method} seed {s}: R={rows[-1]['reward']:.0f} w50={rows[-1]['wait_median']:.1f} w95={rows[-1]['wait_p95']:.1f} sf={rows[-1]['serve_frac']*100:.0f}%", flush=True)
    agg = {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows], ddof=1))]
           for k in rows[0]}
    agg["per_seed"] = rows
    OUT[method] = agg
    print(f"{method}: R={agg['reward'][0]:.1f} w50={agg['wait_median'][0]:.1f} w95={agg['wait_p95'][0]:.1f} sf={agg['serve_frac'][0]*100:.0f}%", flush=True)

tag = "rules" if RULES_ONLY else "8x"
json.dump(OUT, open(f"results/full_metrics_{tag}.json", "w"), indent=1)
print("done ->", f"results/full_metrics_{tag}.json")
