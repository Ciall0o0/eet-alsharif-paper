# eval_multi_metric.py — reward + wait + served% + Wh for BC-e20 / PPO-best / rule @ 8x, 9 held-out seeds
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "/tmp")
import numpy as np, torch
from bc_boost import build_model, gen_events, make_env, run_policy, _eta

dev = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = (9999, 10001, 10003, 10005, 10007, 10009, 10011, 10013, 10015)

def load_ckpt(path):
    ck = torch.load(path, map_location=dev, weights_only=False)
    sd = ck.get("policy_state", ck)
    m = build_model().to(dev); m.load_state_dict(sd, strict=False); m.eval()
    return m

def run_and_metrics(env, events, model=None):
    obs, _ = env.reset(options={"events": events})
    env._ep_cap = 300000
    h = None
    done, steps, tot = False, 0, 0.0
    if model is not None:
        h = model.get_initial_hidden(1, dev)
    while not done and steps < 20000:
        if env.pending_calls:
            if model is not None:
                ot = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    a, _, _, h, _ = model.get_action(ot, h, deterministic=True)
                a = int(a.item())
            else:
                c = env.pending_calls[0]
                a = min(range(3), key=lambda k: _eta(env, k, c))
        else:
            a = 0
        obs, r, done, trunc, _ = env.step(a)
        tot += r; steps += 1
        if trunc: break
    m = env.get_episode_metrics()
    waits = [p["wait_time"] for p in env.completed_passengers]
    total_pax = int(np.asarray(events)[:, 3].sum()) if events is not None and len(events) else 0
    served = len(env.completed_passengers)
    return {"reward": tot, "mean_wait": float(np.mean(waits)) if waits else float("nan"),
            "served": served, "total_pax": total_pax, "serve_frac": served / max(total_pax, 1),
            "wh": m.get("Wh", m.get("energy_wh", 0.0)), "starts": m.get("start_stop_count", 0),
            "empty_floors": m.get("empty_movement_floors", 0)}

def eval_agent(label, model=None):
    rows = []
    for s in SEEDS:
        ev = gen_events(s, 3.2); env = make_env()
        rows.append(run_and_metrics(env, ev, model))
        print(f"{label} seed {s}: R={rows[-1]['reward']:.0f} wait={rows[-1]['mean_wait']:.1f}s serve={rows[-1]['serve_frac']*100:.0f}% Wh={rows[-1]['wh']:.0f}", flush=True)
    keys = ["reward", "mean_wait", "serve_frac", "wh", "starts", "empty_floors"]
    out = {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows], ddof=1))] for k in keys}
    print(f"{label} mean: R={out['reward'][0]:.1f}±{out['reward'][1]:.1f} wait={out['mean_wait'][0]:.1f}s serve={out['serve_frac'][0]*100:.1f}% Wh={out['wh'][0]:.0f} starts={out['starts'][0]:.0f} empty={out['empty_floors'][0]:.0f}", flush=True)
    return out

OUT = {}
OUT["bc_e20"] = eval_agent("BC-e20", load_ckpt("checkpoints/bc_boost_e20.pt"))
OUT["ppo_scratch"] = eval_agent("PPO-best", load_ckpt("checkpoints/probe_8x_pposcratch_s42/ppo_elevator_best.pt"))
OUT["rule"] = eval_agent("rule", None)
json.dump(OUT, open("results/multi_metric_8x.json", "w"), indent=1)
print("done -> results/multi_metric_8x.json")
