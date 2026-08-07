# eval_sd_nearest.py — SD-nearest rule baseline (choose elevator with min distance, ignoring ETA)
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "/tmp")
import numpy as np
from bc_boost import gen_events, make_env

SEEDS = (9999, 10001, 10003, 10005, 10007, 10009, 10011, 10013, 10015)

def rule_run(env, events, mode="nearest"):
    obs, _ = env.reset(options={"events": events})
    env._ep_cap = 300000
    done, steps, tot = False, 0, 0.0
    while not done and steps < 20000:
        if env.pending_calls:
            c = env.pending_calls[0]
            if mode == "nearest":
                # pure distance, no ETA/travel-time model
                a = min(range(3), key=lambda k: abs(env.elevators[k].current_floor - c["floor"]))
            else:  # eta
                a = min(range(3), key=lambda k: env.elevators[k].travel_time_for_distance(
                    abs(env.elevators[k].current_floor - c["floor"])))
        else:
            a = 0
        obs, r, done, trunc, _ = env.step(a)
        tot += r; steps += 1
        if trunc: break
    m = env.get_episode_metrics()
    waits = [p["wait_time"] for p in env.completed_passengers]
    total_pax = int(np.asarray(events)[:, 3].sum())
    return {"reward": tot, "wait": float(np.mean(waits)) if waits else float("nan"),
            "serve": len(env.completed_passengers) / max(total_pax, 1),
            "wh": m.get("Wh", 0.0)}

OUT = {}
for mode in ("nearest", "eta"):
    rows = []
    for s in SEEDS:
        ev = gen_events(s, 3.2); env = make_env()
        rows.append(rule_run(env, ev, mode))
        print(f"SD-{mode} seed {s}: R={rows[-1]['reward']:.0f} wait={rows[-1]['wait']:.1f}s", flush=True)
    OUT[mode] = {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows], ddof=1))]
                 for k in ("reward", "wait", "serve", "wh")}
    print(f"SD-{mode}: R={OUT[mode]['reward'][0]:.1f}±{OUT[mode]['reward'][1]:.1f} "
          f"wait={OUT[mode]['wait'][0]:.1f}s serve={OUT[mode]['serve'][0]*100:.0f}% Wh={OUT[mode]['wh'][0]:.0f}", flush=True)
json.dump(OUT, open("results/sd_rules_8x.json", "w"), indent=1)
print("done -> results/sd_rules_8x.json")
