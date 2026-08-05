"""Batch group-aligned evaluation of all 10F trained agents + SD baselines.

Protocol: real group sizes (pax-aligned x2.5 rates /3.71), 12 episodes,
deterministic greedy. Writes CSV to results/group_eval_matrix.csv
"""
import sys, json, csv, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import torch

from src.train import DAILY_SCHEDULE_12H, _adapt_gen_events
from src.traffic.generator import TrafficGenerator
from src.env.elevator_env import ElevatorEnv
from src.models.gru_ppo import GRUSharedActorCritic

RATES_25 = {"up_peak": 7.5 / 3.71, "down_peak": 7.5 / 3.71, "lunch_peak": 5.0 / 3.71,
            "interfloor": 4.0 / 3.71, "off_peak": 1.5 / 3.71}  # PAX-ALIGNED
VAL_SEED = 9999

MODELS = [
    # (name, ckpt_dir)
    ("noaux92_s42", "checkpoints/shared_noaux_v2c92"),
    ("noaux122_s42", "checkpoints/shared_noaux_v2c"),
    ("zoneaux_s42", "checkpoints/shared_zone_s42"),
    ("zoneaux_s360", "checkpoints/shared_daux_s360"),
    ("zoneaux_s712", "checkpoints/shared_daux_s712"),
    ("eventaux92_s42", "checkpoints/shared_event_d92_s42"),
    ("mixaux122_s42", "checkpoints/shared_mix_aux122_6h_s42"),
]


def gen_episode(seed):
    g = TrafficGenerator(n_floors=10, seed=seed, schedule=DAILY_SCHEDULE_12H,
                         arrival_rates=RATES_25)
    raw = g.generate_episode_multi_segment(n_segments=1, seed_shift=seed + 7,
                                           schedule=DAILY_SCHEDULE_12H,
                                           max_events=3500)[0]
    return _adapt_gen_events(np.array(raw))


def make_env(events, state_dim=122):
    return ElevatorEnv(config={"num_floors": 10, "num_elevators": 3, "max_load_kg": 900,
                               "max_total_time": 43200.0, "max_dt": 30.0, "reward_scale": 0.01,
                               "obs_car_calls_dist": state_dim > 92})


def _eta(env, k, call):
    el = env.elevators[k]
    tt = el.travel_time_for_distance(abs(el.current_floor - call["floor"]))
    if el.state == "moving":
        tt += (el.travel_time_for_distance(abs(el.target_floor - el.current_floor))
               + el.door_open_time + el.door_close_time)
    return tt


def run_rule(env, events, mode):
    obs, _ = env.reset(options={"events": events})
    steps, tot = 0, 0.0
    done = False
    while not done and steps < 20000:
        if env.pending_calls:
            c = env.pending_calls[0]
            if mode == "sd_eta":
                a = min(range(env.num_elevators), key=lambda k: _eta(env, k, c))
            else:
                a = min(range(env.num_elevators),
                        key=lambda k: abs(env.elevators[k].current_floor - c["floor"]))
        else:
            a = 0
        obs, r, done, trunc, _ = env.step(a)
        tot += r
        steps += 1
        if trunc:
            break
    return tot, env.get_episode_metrics()


def load_model(ckpt_dir):
    ck = torch.load(f"{ckpt_dir}/ppo_elevator_best.pt", map_location="cpu", weights_only=False)
    sd = ck["policy_state"]
    state_dim = next(v.shape[1] for k, v in sd.items() if "weight_ih_l0" in k)
    num_dest = int(sd["dest_head.2.weight"].shape[0]) if "dest_head.2.weight" in sd else 10
    m = GRUSharedActorCritic(
        state_dim=state_dim, action_dim=3,
        aux_prediction=bool(ck.get("aux_prediction", False)),
        num_dest_classes=num_dest, use_layer_norm=True,
        gru_hidden=256, gru_layers=2, gru_dropout=0.1,
        actor_hidden=64, critic_hidden=64,
        dest_head_on="dest_head.0.weight" in sd,
        event_head_on="event_head.0.weight" in sd,
        reward_change_on="reward_change_head.0.weight" in sd)
    m.load_state_dict(sd)
    m.eval()
    return m


def run_policy(env, events, model):
    obs, _ = env.reset(options={"events": events})
    h = model.get_initial_hidden(1, "cpu")
    steps, tot = 0, 0.0
    done = False
    while not done and steps < 20000:
        if env.pending_calls:
            ot = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                a, _, _, h, _ = model.get_action(ot, h, deterministic=True)
            a = int(a.item())
        else:
            a = 0
        obs, r, done, trunc, _ = env.step(a)
        tot += r
        steps += 1
        if trunc:
            break
    return tot, env.get_episode_metrics()


def main():
    n_ep = 12
    rows = []
    agents = []
    for name, ckpt in MODELS:
        try:
            model = load_model(ckpt)
            agents.append((name, model))
            print(f"loaded {name}", flush=True)
        except Exception as e:
            print(f"SKIP {name}: {str(e)[:80]}", flush=True)

    for name, model in agents:
        rs, ps, ws = [], [], []
        sd0 = model.encoder.input_size
        for i in range(n_ep):
            ev = gen_episode(VAL_SEED + i)
            env = make_env(ev, sd0)
            r, m = run_policy(env, ev, model)
            rs.append(r); ps.append(m["total_passengers"]); ws.append(m["avg_wait_time"])
        row = {"agent": name, "type": "rl",
               "reward_mean": float(np.mean(rs)), "reward_sd": float(np.std(rs)),
               "pax": float(np.mean(ps)), "wait": float(np.mean(ws))}
        rows.append(row)
        np.save(f"results/raw_{name}.npy", np.array(rs))
        print(f"{name}: {row['reward_mean']:.0f} ± {row['reward_sd']:.0f} | pax {row['pax']:.0f} | wait {row['wait']:.1f}", flush=True)

    for mode in ["sd_eta", "sd_nearest"]:
        rs, ps, ws = [], [], []
        for i in range(n_ep):
            ev = gen_episode(VAL_SEED + i)
            env = make_env(ev)
            r, m = run_rule(env, ev, mode)
            rs.append(r); ps.append(m["total_passengers"]); ws.append(m["avg_wait_time"])
        np.save(f"results/raw_{mode}.npy", np.array(rs))
        rows.append({"agent": mode, "type": "rule",
                     "reward_mean": float(np.mean(rs)), "reward_sd": float(np.std(rs)),
                     "pax": float(np.mean(ps)), "wait": float(np.mean(ws))})
        print(f"{mode}: {np.mean(rs):.0f} ± {np.std(rs):.0f} | pax {np.mean(ps):.0f} | wait {np.mean(ws):.1f}", flush=True)

    import os
    os.makedirs("results", exist_ok=True)
    with open("results/group_eval_matrix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["agent", "type", "reward_mean", "reward_sd", "pax", "wait"])
        w.writeheader()
        w.writerows(rows)
    print("WROTE results/group_eval_matrix.csv")


if __name__ == "__main__":
    main()