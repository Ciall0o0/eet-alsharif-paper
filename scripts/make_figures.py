"""Reproduce all paper figures (matplotlib) from results/*.csv.

Usage: python scripts/make_figures.py [--out figures]
Produces: fig1_main.png, fig2_density.png, fig3_synergy.png,
fig4_coverage.png, fig5_od_prior.png
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def load_csv(p):
    import csv
    with open(p) as f:
        rows = list(csv.reader(f))
    hdr = rows[0]
    return hdr, rows[1:]


def fig1_main(out):
    hdr, rows = load_csv(ROOT / "results/main_results.csv")
    names = [r[0] for r in rows]
    mean = [float(r[1]) for r in rows]
    se = [float(r[2]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#888"] * 2 + ["#6baed6"] * 3 + ["#d62728", "#ff9896"]
    ax.bar(range(len(mean)), mean, yerr=se, color=colors, capsize=4, alpha=0.9)
    ax.set_xticks(range(len(mean)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Mean per-episode reward (12h, 2.5x)")
    ax.set_title("Main results: rule-collapse regime")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fig1_main.png", dpi=200)
    plt.close(fig)


def fig2_density(out):
    hdr, rows = load_csv(ROOT / "results/density_profile.csv")
    xs = [float(r[0].rstrip("x")) for r in rows]
    agents = [hdr[1], hdr[2], hdr[3]]
    fig, ax = plt.subplots(figsize=(6, 4))
    styles = {"SD-ETA": "k--o", "Zone-Aux": "r-s", "MIX": "b-^"}
    for j, a in enumerate(agents):
        ys = [float(r[j + 1]) for r in rows]
        ax.plot(xs, ys, styles[a], label=a, lw=1.8)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Arrival-rate multiplier")
    ax.set_ylabel("Mean per-episode reward")
    ax.set_title("Density--reward profile")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fig2_density.png", dpi=200)
    plt.close(fig)


def fig3_synergy(out):
    hdr, rows = load_csv(ROOT / "results/synergy_ablation.csv")
    labels = [r[0] for r in rows]
    no_aux = [float(r[1]) for r in rows]
    aux = [float(r[2]) for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(x - 0.18, no_aux, 0.36, label="No aux", color="#6baed6")
    ax.bar(x + 0.18, aux, 0.36, label="Zone-Aux", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Mean reward (2.5x)")
    ax.set_title("Feature x auxiliary synergy")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fig3_synergy.png", dpi=200)
    plt.close(fig)


def fig4_coverage(out):
    hdr, rows = load_csv(ROOT / "results/coverage_ablation.csv")
    labels = [r[0] for r in rows]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for j, col in enumerate(["Eval2_5x", "Eval3_0x"]):
        ys = [float(r[j + 1]) for r in rows]
        ax.bar(x + j * 0.3 - 0.15, ys, 0.28, label=col.replace("Eval", "@"))
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Mean reward")
    ax.set_title("Training-distribution ablation")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fig4_coverage.png", dpi=200)
    plt.close(fig)


def fig5_od_prior(out):
    hdr, rows = load_csv(ROOT / "results/od_prior_20f.csv")
    labels = [f"{r[0]} {r[1]}" for r in rows]
    prior = [float(r[3]) for r in rows]
    chance = [float(r[4]) for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(x, prior, 0.5, color="#2ca02c", label="argmax prior")
    ax.plot(x, chance, "k--", label="chance")
    for xi, p, c in zip(x, prior, chance):
        ax.text(xi, p + 0.01, f"{p:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("Predictability")
    ax.set_title("OD zone prior: 10F vs 20F")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fig5_od_prior.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "figures"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True, parents=True)
    fig1_main(out); fig2_density(out); fig3_synergy(out)
    fig4_coverage(out); fig5_od_prior(out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
