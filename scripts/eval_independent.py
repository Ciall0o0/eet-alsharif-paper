"""Independent evaluation reproducing the paper's main tables and statistics.

Usage (from repo root):
    python scripts/eval_independent.py \
        --config configs/config_gru_shared_event_d.yaml \
        --ckpt-dir checkpoints/zone_aux_seed42 \
        --seed 42 --n-episodes 12
    python scripts/eval_independent.py --sd-eta --n-episodes 12

Design: greedy deterministic actions, fixed held-out episode seeds
(9999..9999+n-1), per-episode GRU hidden reset, full 12h schedule.
Prints per-agent reward/pax/wait and paired t-test + Cohen's d vs the
rule baseline, matching Table I and Section V of the paper.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import DAILY_SCHEDULE_12H, _adapt_gen_events  # noqa: E402
from src.traffic.generator import TrafficGenerator  # noqa: E402
from src.env.elevator_env import ElevatorEnv  # noqa: E402
from src.models.gru_ppo import GRUSharedActorCritic  # noqa: E402

BASE_RATES = {"up_peak": 3.0, "down_peak": 3.0, "lunch_peak": 2.0,
              "interfloor": 1.6, "off_peak": 0.6}
MAX_STEPS = 60000


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["policy_state"]
    state_dim = next(v.shape[1] for k, v in sd.items() if "weight_ih_l0" in k)
    aux = bool(ck.get("aux_prediction", False))
    m = GRUSharedActorCritic(
        state_dim=state_dim, action_dim=3, aux_prediction=aux,
        num_dest_classes=10, use_layer_norm=True,
        gru_hidden=256, gru_layers=2, gru_dropout=0.1,
        actor_hidden=64, critic_hidden=64,
        dest_head_on="dest_head.0.weight" in sd,
        event_head_on="event_head.0.weight" in sd,
        reward_change_on="reward_change_head.0.weight" in sd)
    m.load_state_dict(sd)
    m.to(device); m.eval()
    return m


def _eta(env, k, call):
    el = env.elevators[k]
    tt = el.travel_time_for_distance(abs(el.current_floor - call["floor"]))
    if el.state == "moving":
        tt += (el.travel_time_for_distance(abs(el.target_floor - el.current_floor))
               + el.door_open_time + el.door_close_time)
    return tt


def run_episode(env_cfg, ep, model=None, rule=None, device="cuda"):
    env = ElevatorEnv(config=env_cfg)
    obs, _ = env.reset(options={"events": _adapt_gen_events(ep)})
    h = model.get_initial_hidden(1, device) if model is not None else None
    done, tot, steps = False, 0.0, 0
    while not done and steps < MAX_STEPS:
        if env.pending_calls:
            c = env.pending_calls[0]
            if rule == "nearest":
                a = min(range(env.num_elevators),
                        key=lambda k: abs(env.elevators[k].current_floor - c["floor"]))
            elif rule == "eta":
                a = min(range(env.num_elevators), key=lambda k: _eta(env, k, c))
            else:
                ot = (torch.as_tensor(obs, dtype=torch.float32, device=device)
                      .unsqueeze(0).unsqueeze(0))
                a, _, _, h, _ = model.get_action(ot, h, deterministic=True)
                a = a.item()
        else:
            a = 0
        obs, r, done, trunc, _ = env.step(a)
        tot += r; steps += 1
        if trunc:
            break
    met = env.get_episode_metrics()
    return tot, met


def eval_agent(env_cfg, rates, max_events, n_ep, seed0,
               model=None, rule=None, device="cuda"):
    R, P, W = [], [], []
    for i in range(n_ep):
        g = TrafficGenerator(n_floors=env_cfg.get("num_floors", 10),
                             seed=9999 + i, schedule=DAILY_SCHEDULE_12H,
                             arrival_rates=rates)
        ep = g.generate_episode_multi_segment(
            n_segments=1, seed_shift=10000 + i,
            schedule=DAILY_SCHEDULE_12H, max_events=max_events)[0]
        r, met = run_episode(env_cfg, ep, model, rule, device)
        R.append(r); P.append(met["total_passengers"]); W.append(met["avg_wait_time"])
    return (np.array(R), np.array(P), np.array(W))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/config_gru_noaux.yaml"))
    ap.add_argument("--ckpt-dir", help="checkpoint dir containing ppo_elevator_best.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-episodes", type=int, default=12)
    ap.add_argument("--rates-scale", type=float, default=2.5)
    ap.add_argument("--max-events", type=int, default=3500)
    ap.add_argument("--sd-eta", action="store_true", help="evaluate SD-ETA rule")
    ap.add_argument("--sd-nearest", action="store_true", help="evaluate SD-nearest rule")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    env_cfg = dict(cfg["env"])
    rates = {k: v * args.rates_scale for k, v in BASE_RATES.items()}

    agents = []
    if args.sd_eta:
        agents.append(("SD-ETA", None, "eta"))
    if args.sd_nearest:
        agents.append(("SD-nearest", None, "nearest"))
    if args.ckpt_dir:
        model = load_model(Path(args.ckpt_dir) / "ppo_elevator_best.pt", args.device)
        agents.append((f"Zone-Aux-s{args.seed}", model, None))

    from scipy import stats
    for name, model, rule in agents:
        R, P, W = eval_agent(env_cfg, rates, args.max_events, args.n_episodes,
                             args.seed, model, rule, args.device)
        se = R.std() / np.sqrt(len(R))
        print(f"{name:16s}: reward={R.mean():+9.1f}±{se:7.1f}  "
              f"pax={P.mean():5.0f}  wait={W.mean():5.1f}s")
        if len(agents) > 1 and rule is None:
            rl, sd = R, None
            for n2, _, r2 in agents:
                if r2 == "eta":
                    sd = None  # placeholder for paired stats
            # unpaired t vs first rule agent
            sdR = next(R2 for n2, _, r2 in agents if r2 == "eta" or r2 == "nearest")
            t, p = stats.ttest_ind(rl, sdR)
            d = (rl.mean() - sdR.mean()) / np.sqrt((rl.var() + sdR.var()) / 2)
            print(f"  vs SD: t={t:+.3f} p={p:.2e} Cohen's d={d:+.3f}")


if __name__ == "__main__":
    main()
