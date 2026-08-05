"""Pool per-episode raw results into the paper's 3-seed Welch t-test + Cohen's d.

Preferred usage (real per-episode data, exactly the paper numbers):
    python scripts/pool_stats.py --npy results --proposed zoneaux_s42_main zoneaux_s360 zoneaux_s712 --sd results/raw_sd_eta.npy

Legacy usage (kept for backward compatibility; synthetic mean-fill, less accurate):
    python scripts/pool_stats.py --rewards 42:-804.8,360:-721.6,712:-852.1 --sd -2417.7 --n-episodes 12
"""
import argparse, csv, glob
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def welch(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    # Cohen's d with population variance (ddof=0) — matches the paper
    d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    return t, p, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", help="directory containing raw_*.npy per-episode files")
    ap.add_argument("--proposed", nargs="+", help="raw file stems for the proposed agent seeds")
    ap.add_argument("--sd", help="raw file stem (or mean value) of the rule baseline")
    ap.add_argument("--rewards", help="legacy: comma list seed:mean,seed:mean,...")
    ap.add_argument("--n-episodes", type=int, default=12)
    ap.add_argument("--csv", help="legacy: read per-seed csv (RewardMean col)")
    args = ap.parse_args()

    if args.npy:
        prop = [np.load(Path(args.npy) / f"raw_{k}.npy") for k in args.proposed]
        a = np.concatenate(prop)
        sd_path = Path(args.sd) if Path(args.sd).is_absolute() or "/" in args.sd else Path(args.npy) / args.sd
        b = np.load(sd_path)
        t, p, d = welch(a, b)
        print(f"proposed pooled mean={a.mean():+.1f}  SE={a.std(ddof=1)/np.sqrt(len(a)):.1f}  (n={len(a)})")
        print(f"baseline         mean={b.mean():+.1f}  SE={b.std(ddof=1)/np.sqrt(len(b)):.1f}  (n={len(b)})")
        print(f"Welch t={t:+.3f}  p={p:.2e}  Cohen's d={d:+.3f}  (ddof=0 variance, paper convention)")
        # per-seed rows
        for k, v in zip(args.proposed, prop):
            t1, p1, d1 = welch(v, b)
            print(f"  {k}: {v.mean():+.1f} ± {v.std(ddof=1):.1f}  t={t1:+.3f} p={p1:.2e} d={d1:+.3f}")
        return

    # legacy path (synthetic means — kept for compatibility)
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
    try:
        sd_r = np.full(n * len(means), float(args.sd))
    except ValueError:
        sd_r = np.load(ROOT / args.sd)
    t, p = stats.ttest_ind(all_r, sd_r)
    d = (all_r.mean() - sd_r.mean()) / np.sqrt((all_r.var() + sd_r.var()) / 2)
    print(f"pooled mean={all_r.mean():+.1f}  SE={all_r.std()/np.sqrt(len(all_r)):.1f}  (n={len(all_r)})")
    print(f"t={t:+.3f}  p={p:.2e}  Cohen's d={d:+.3f}")


if __name__ == "__main__":
    main()
