"""Reproduce Table VIII: OD zone-prior statistics for 10F and 20F buildings.

Usage: python scripts/od_prior.py [--n-episodes 30] [--seed0 1000]
Prints per-building argmax-prior accuracy vs chance, matching the
"Generalization to a 20-Floor Building" section.
"""
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import DAILY_SCHEDULE_12H  # noqa: E402
from src.traffic.generator import TrafficGenerator  # noqa: E402
from src.zone_map import zone_label  # noqa: E402

BASE = {"up_peak": 3.0, "down_peak": 3.0, "lunch_peak": 2.0,
        "interfloor": 1.6, "off_peak": 0.6}


def od_stats(n_floors, mode, label, n_eps, seed0):
    all_od = []
    for i in range(n_eps):
        g = TrafficGenerator(n_floors=n_floors, seed=seed0 + i,
                             schedule=DAILY_SCHEDULE_12H, arrival_rates=BASE)
        ep = g.generate_episode_multi_segment(
            n_segments=1, seed_shift=5000 + i,
            schedule=DAILY_SCHEDULE_12H, max_events=1400)[0]
        for ev in ep:
            all_od.append((int(ev[0]), int(ev[1])))
    all_od = np.array(all_od)
    n_z = 4 if mode == "functional" else 3
    P = np.zeros((n_z, n_z))
    for o, d in all_od:
        P[zone_label(o, mode, n_floors), zone_label(d, mode, n_floors)] += 1
    P /= P.sum()
    prior_acc = sum(max(P[z] / P[z].sum()) * P[z].sum() for z in range(n_z))
    corr = np.corrcoef(all_od[:, 0], all_od[:, 1])[0, 1]
    print(f"{label:22s}: argmax-prior={prior_acc:.3f}  chance={1.0/n_z:.3f}  "
          f"floor-r={corr:+.3f}  (n={len(all_od)})")
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1000)
    args = ap.parse_args()
    od_stats(10, "height", "10F height", args.n_episodes, args.seed0)
    od_stats(20, "height", "20F height", args.n_episodes, args.seed0)
    od_stats(20, "functional", "20F functional", args.n_episodes, args.seed0)


if __name__ == "__main__":
    main()
