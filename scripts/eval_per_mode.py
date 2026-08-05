#!/usr/bin/env python3
"""Evaluate paper checkpoints per traffic mode -> CSV for Origin plotting.

Replays the SAME fixed validation set as training:
  - TrafficGenerator.generate_validation_set(arrival_rates=<from config>,
      n_per_mode=<val_n_per_mode from config>, seed=9999)
  - deterministic greedy action selection via model.get_action
  - env.step(action: int) single-elevator-index action

Emits per-mode mean reward + aggregate for each model group (unreal/noaux).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

from src.env.elevator_env import ElevatorEnv
from src.models.gru_ppo import GRUActorCritic
from src.traffic.generator import TrafficGenerator

TRAFFIC_MODES = ["up_peak", "down_peak", "lunch_peak", "interfloor", "off_peak"]
VAL_SEED = 9999
MAX_VAL_STEPS = 16000


def _adapt_gen_events(gen_events):
    """Mirror src/train.py _adapt_gen_events."""
    if gen_events.size == 0:
        return gen_events
    n = gen_events.shape[0]
    ncols = max(10, gen_events.shape[1])
    out = np.zeros((n, ncols), dtype=np.float32)
    out[:, 0] = gen_events[:, 0]
    out[:, 1] = gen_events[:, 1]
    out[:, 2] = 1.0
    out[:, 3] = gen_events[:, 2]
    for c in range(4, min(gen_events.shape[1], ncols)):
        out[:, c] = gen_events[:, c]
    return out


def build_policy(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dim = None
    for k, v in ck["policy_state"].items():
        if "weight_ih_l0" in k:
            state_dim = v.shape[1]
            break
    assert state_dim is not None, "cannot derive state_dim"
    num_dest_classes = 0
    for k, v in ck["policy_state"].items():
        if k.startswith("dest_head.2.") and v.ndim == 2:
            num_dest_classes = v.shape[0]
    if num_dest_classes == 0:
        num_dest_classes = 10  # no dest_head (noaux)
    aux = bool(ck.get("aux_prediction", False))
    use_ln = any("layer_norm" in k for k in ck["policy_state"])
    model = GRUActorCritic(
        state_dim=state_dim,
        action_dim=3,
        aux_prediction=aux,
        num_dest_classes=num_dest_classes,
        use_layer_norm=use_ln,
    )
    model.load_state_dict(ck["policy_state"])
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, val_items, device):
    results = {m: [] for m in TRAFFIC_MODES}
    env = ElevatorEnv(config={"num_floors": 10, "num_elevators": 3})
    for mode, events_arr in val_items:
        if events_arr.size == 0 or events_arr.shape[0] == 0:
            continue
        adapted = _adapt_gen_events(events_arr)
        obs, _ = env.reset(options={"events": adapted})
        total_reward = 0.0
        done = False
        hidden = None
        for _ in range(MAX_VAL_STEPS):
            if done:
                break
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            obs_t = obs_t.unsqueeze(0).unsqueeze(0)  # (1,1,state_dim)
            with torch.no_grad():
                out = model.get_action(obs_t, hidden=hidden, deterministic=True)
                action = int(out[0].squeeze().item())
                hidden = out[3]
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            done = bool(terminated or truncated)
        results[mode].append(total_reward)
    return {m: (float(np.mean(v)) if v else float("nan")) for m, v in results.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--tag", type=str, default="", help="checkpoint dir suffix, e.g. _b27")
    parser.add_argument("--group", type=str, default="", help="only eval this group (unreal/noaux)")
    parser.add_argument("--rates-scale", type=float, default=1.0,
                        help="scale arrival_rates (0.5 = 1x real density for per-mode analysis)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    cfg = yaml.safe_load((_PROJ_ROOT / "config" / "config_gru_dest.yaml").read_text())
    traffic_cfg = cfg.get("traffic", {})
    arrival_rates = traffic_cfg.get("arrival_rates")
    if args.rates_scale != 1.0:
        arrival_rates = {k: v * args.rates_scale for k, v in arrival_rates.items()}
    val_n_per_mode = traffic_cfg.get("val_n_per_mode", 3)
    print(f"val_n_per_mode={val_n_per_mode}, arrival_rates={arrival_rates} (x{args.rates_scale})")

    # Per-mode analysis: 90 min pure-mode blocks @ (optionally scaled) rates.
    # (generate_validation_set's 8 h pure blocks overload 3 cars: sustained
    # peak -> est_wait queue explodes -> meaningless -60k rewards.)
    MODE_MIN = 90
    val_max_events = traffic_cfg.get("val_max_events", 500)
    val_items = []
    for mode in TRAFFIC_MODES:
        for i in range(val_n_per_mode):
            g = TrafficGenerator(seed=VAL_SEED + i, schedule=[(0, MODE_MIN, mode)],
                                 arrival_rates=arrival_rates)
            ep = g.generate_episode(duration_seconds=MODE_MIN * 60.0, max_events=val_max_events)
            if ep.size > 0 and ep.shape[0] > 0:
                val_items.append((mode, ep))
    print(f"val episodes: {len(val_items)} ({val_n_per_mode}/mode x {len(TRAFFIC_MODES)} modes)")

    ckpt_dir = _PROJ_ROOT / "checkpoints"
    tag = args.tag
    models = {
        "unreal": [f"unreal_s{s}{tag}" for s in (42, 360, 712)],
        "unreal_zone": [f"unreal_zone_s{s}{tag}" for s in (42, 360, 712)],
        "noaux": [f"noaux_s{s}{tag}" for s in (42, 360, 712)],
    }
    if args.group:
        models = {k: v for k, v in models.items() if k == args.group}

    rows = []
    for group, names in models.items():
        per_mode_all = {m: [] for m in TRAFFIC_MODES}
        for name in names:
            p = ckpt_dir / name / "ppo_elevator_best.pt"
            model = build_policy(p, device)
            res = evaluate_model(model, val_items, device)
            print(f"[{group} {name}] " + ", ".join(f"{m}={v:.0f}" for m, v in res.items()), flush=True)
            for m, v in res.items():
                per_mode_all[m].append(v)
        for m in TRAFFIC_MODES:
            vals = per_mode_all[m]
            rows.append({
                "group": group,
                "mode": m,
                "s42": vals[0] if len(vals) > 0 else float("nan"),
                "s360": vals[1] if len(vals) > 1 else float("nan"),
                "s712": vals[2] if len(vals) > 2 else float("nan"),
                "mean": float(np.mean(vals)) if vals else float("nan"),
                "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
            })

    out_path = Path(args.out) if args.out else _PROJ_ROOT / "per_mode_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
