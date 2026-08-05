"""Independent 12H evaluation under REAL group sizes (n_pax from gen col5).

Loads a trained model, evaluates at x2.5 (collapse zone) with group_size
injected via _adapt_gen_events, plus SD-ETA / SD-nearest rule baselines under
the same protocol. 12 episodes per agent, deterministic greedy action.

Usage:
  .venv/bin/python /tmp/eval_group_independent.py --ckpt checkpoints/gru_daux_group_s42 --tag group_s42 [--sd-eta] [--sd-nearest]
"""
import argparse, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import torch

from src.train import DAILY_SCHEDULE_12H, _adapt_gen_events
from src.traffic.generator import TrafficGenerator
from src.env.elevator_env import ElevatorEnv
from src.models.gru_ppo import GRUSharedActorCritic

RATES_25 = {"up_peak": 7.5 / 3.71, "down_peak": 7.5 / 3.71, "lunch_peak": 5.0 / 3.71,
            "interfloor": 4.0 / 3.71, "off_peak": 1.5 / 3.71}  # PAX-ALIGNED (/mean group)
VAL_SEED = 9999


def make_env(events):
    return ElevatorEnv(config={"num_floors": 10, "num_elevators": 3, "max_load_kg": 900,
                               "max_total_time": 43200.0, "max_dt": 30.0, "reward_scale": 0.01})


def gen_episode(seed):
    g = TrafficGenerator(n_floors=10, seed=seed, schedule=DAILY_SCHEDULE_12H,
                         arrival_rates=RATES_25)
    raw = g.generate_episode_multi_segment(n_segments=1, seed_shift=seed + 7,
                                           schedule=DAILY_SCHEDULE_12H,
                                           max_events=3500)[0]
    return _adapt_gen_events(np.array(raw))


def _eta(env, k, call):
    el = env.elevators[k]
    tt = el.travel_time_for_distance(abs(el.current_floor - call["floor"]))
    if el.state == "moving":
        tt += (el.travel_time_for_distance(abs(el.target_floor - el.current_floor))
               + el.door_open_time + el.door_close_time)
    return tt


def run_rule(env, events, mode):
    done = False
    obs, _ = env.reset(options={"events": events})
    steps, tot = 0, 0.0
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


def run_policy(env, events, model):
    done = False
    obs, _ = env.reset(options={"events": events})
    h = model.get_initial_hidden(1, "cpu")
    steps, tot = 0, 0.0
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


def load_model(ckpt_dir):
    ck = torch.load(f"{ckpt_dir}/ppo_elevator_best.pt", map_location="cpu", weights_only=False)
    sd = ck["policy_state"]
    state_dim = next(v.shape[1] for k, v in sd.items() if "weight_ih_l0" in k)
    m = GRUSharedActorCritic(
        state_dim=state_dim, action_dim=3,
        aux_prediction=bool(ck.get("aux_prediction", False)),
        num_dest_classes=10, use_layer_norm=True,
        gru_hidden=256, gru_layers=2, gru_dropout=0.1,
        actor_hidden=64, critic_hidden=64,
        dest_head_on="dest_head.0.weight" in sd,
        event_head_on="event_head.0.weight" in sd,
        reward_change_on="reward_change_head.0.weight" in sd)
    m.load_state_dict(sd)
    m.eval()
    return m


def run_policy_old(env, events, model):
    done = False
    obs, _ = env.reset(options={"events": events})
    while not done:
        o = torch_tensor(obs)
        with torch.no_grad():
            action = model.get_action(o, deterministic=True)
        obs, r, done, trunc, _ = env.step(int(action))
        if trunc:
            break
    return env.get_episode_metrics()


def torch_tensor(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float32)).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gru_daux_group_s42")
    ap.add_argument("--tag", default="group_s42")
    ap.add_argument("--n-episodes", type=int, default=12)
    ap.add_argument("--sd-eta", action="store_true")
    ap.add_argument("--sd-nearest", action="store_true")
    args = ap.parse_args()

    model = load_model(args.ckpt)
    print(f"== model loaded, aux {model.aux_prediction}")

    results = {}
    for i in range(args.n_episodes):
        ev = gen_episode(VAL_SEED + i)
        env = make_env(ev)
        r, m = run_policy(env, ev, model)
        for k, v in m.items():
            results.setdefault(k, []).append(v)
        results.setdefault("total_reward", []).append(r)
        if i % 3 == 0:
            print(f"  ep {i}: reward {r:.0f} pax {m.get('total_passengers', 0)}")
    print(f"== {args.tag}: reward {np.mean(results['total_reward']):.1f} ± {np.std(results['total_reward']):.1f} | "
          f"pax {np.mean(results['total_passengers']):.1f} | wait {np.mean(results['avg_wait_time']):.1f}")

    for mode, flag in [("sd_eta", args.sd_eta), ("sd_nearest", args.sd_nearest)]:
        if not flag:
            continue
        rs, ps_, ws = [], [], []
        for i in range(args.n_episodes):
            ev = gen_episode(VAL_SEED + i)
            env = make_env(ev)
            r, m = run_rule(env, ev, mode)
            rs.append(r); ps_.append(m["total_passengers"]); ws.append(m["avg_wait_time"])
        print(f"== {mode}: reward {np.mean(rs):.1f} ± {np.std(rs):.1f} | pax {np.mean(ps_):.1f} | wait {np.mean(ws):.1f}")


if __name__ == "__main__":
    main()
