"""Pool per-seed evaluation CSVs into the paper's 3-seed t-test + Cohen's d.

Usage: python scripts/pool_stats.py --rewards 42:-4922.3,360:-8451.2,712:-1867.5 \
       --sd -32084.9 --n-episodes 12
Or:    python scripts/pool_stats.py --csv results/main_results_perseed.csv
Prints pooled mean/SE, t, p, and Cohen's d (paper Section V).
"""
import argparse, csv, math
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewards", help="comma list seed:mean,seed:mean,...")
    ap.add_argument("--sd", type=float, help="SD baseline mean")
    ap.add_argument("--sd-se", type=float, default=3013.8)
    ap.add_argument("--n-episodes", type=int, default=12)
    ap.add_argument("--csv", help="alt: read per-seed csv (RewardMean col)")
    args = ap.parse_args()

    means, n = [], args.n_episodes
    if args.csv:
        with open(ROOT / args.csv) as f:
            for r in csv.DictReader(f):
                means.append(float(r["RewardMean"]))
    else:
        for tok in args.rewards.split(","):
            _, m = tok.split(":")
            means.append(float(m))

    all_r = np.concatenate([np.full(n, m) for m in means])
    sd_r = np.full(n * len(means), args.sd)
    t, p = stats.ttest_ind(all_r, sd_r)
    d = (all_r.mean() - sd_r.mean()) / np.sqrt((all_r.var() + sd_r.var()) / 2)
    print(f"pooled mean={all_r.mean():+.1f}  SE={all_r.std()/np.sqrt(len(all_r)):.1f}  "
          f"(n={len(all_r)})")
    print(f"t={t:+.3f}  p={p:.2e}  Cohen's d={d:+.3f}")


if __name__ == "__main__":
    main()
