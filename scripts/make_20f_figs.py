"""20F generalization figures (IEEE style, matches 10F group-protocol figures).
Outputs:
  fig_20f_main_results_ieee.pdf/.png   — 20F bar chart vs rule baselines
  fig_20f_distribution_ieee.pdf/.png   — per-episode box+strip showing separation
  fig_10f_20f_transfer_ieee.pdf/.png   — two-panel 10F vs 20F comparison
Copied into paper_access/figs, paper_csmag/figs, figs_ieee/.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os, shutil
from scipy import stats

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": "black",
    "axes.linewidth": 0.8, "savefig.facecolor": "white",
})
OKABE = {"blue": "#0072BD", "orange": "#D95319", "green": "#009E73",
         "red": "#D55E00", "grey": "#999999", "purple": "#7E2F8E"}

R = "results"
def load(k): return np.load(f"{R}/raw_{k}.npy")

def welch(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    return t, p, d

# ---------------- 20F data ----------------
z42, z360, z712 = load("zone20f_s42"), load("zone20f_s360"), load("zone20f_s712")
za20 = np.concatenate([z42, z360, z712])          # n=36
sd_eta20, sd_near20 = load("20f_sd_eta"), load("20f_sd_nearest")   # n=12 each

t_e, p_e, d_e = welch(za20, sd_eta20)
t_n, p_n, d_n = welch(za20, sd_near20)
print(f"20F pooled={za20.mean():.1f}±{za20.std():.1f} (n={len(za20)})")
print(f"vs SD-ETA:  t={t_e:.3f} p={p_e:.2e} d={d_e:.3f}")
print(f"vs SD-near: t={t_n:.3f} p={p_n:.2e} d={d_n:.3f}")

# ---------------- Fig 1: 20F main results bar ----------------
agents = ["SD-\nnearest", "SD-ETA", "Zone-Aux\n(20F)"]
rewards = [sd_near20.mean(), sd_eta20.mean(), za20.mean()]
errs = [sd_near20.std()/np.sqrt(12), sd_eta20.std()/np.sqrt(12), za20.std()/np.sqrt(36)]
colors = [OKABE["grey"], OKABE["grey"], OKABE["blue"]]

fig, ax = plt.subplots(figsize=(4.6, 2.6), dpi=150)
bars = ax.bar(agents, rewards, yerr=errs, capsize=3, color=colors,
              edgecolor="black", linewidth=0.6, error_kw=dict(lw=0.8))
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Mean per-episode reward")
ax.set_title("20F: 10F-trained agent under real group arrivals (2.5$\\times$ pax-aligned)", fontsize=9)
for b, r in zip(bars, rewards):
    ax.text(b.get_x() + b.get_width()/2, r - 90, f"{r:,.0f}",
            ha="center", va="top", fontsize=8)
# significance bracket: SD-ETA (x=1) -> Zone-Aux (x=2)
y1 = max(rewards[0], rewards[1]) + 350
ax.annotate("", xy=(1, y1), xytext=(2, y1),
            arrowprops=dict(arrowstyle="-", lw=0.8))
ax.text(1.5, y1 + 80, f"$t={t_e:.2f}$, $p={p_e:.1e}$, $d={d_e:.2f}$", ha="center", fontsize=7.5)
ax.set_ylim(-4300, 200)
ax.tick_params(axis="x", labelsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(f"/tmp/fig_20f_main_results_ieee.{ext}", bbox_inches="tight")
plt.close(fig)
print("fig1 done")

# ---------------- Fig 2: per-episode distribution (box + strip) ----------------
fig, ax = plt.subplots(figsize=(4.6, 2.8), dpi=150)
data = [sd_near20, sd_eta20, za20]
labels = ["SD-\nnearest", "SD-ETA", "Zone-Aux\n(20F)"]
cols = [OKABE["grey"], OKABE["grey"], OKABE["blue"]]
bp = ax.boxplot(data, tick_labels=labels, widths=0.5, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", lw=1.0))
for patch, c in zip(bp["boxes"], cols):
    patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor("black"); patch.set_linewidth(0.6)
for i, (d, c) in enumerate(zip(data, cols)):
    rng = np.random.default_rng(i)
    x = rng.normal(i + 1, 0.04, len(d))
    ax.scatter(x, d, s=14, color=c, edgecolor="black", linewidth=0.3, alpha=0.85, zorder=3)
ax.set_ylabel("Per-episode reward")
ax.set_title("20F: per-episode rewards (n = 36 vs 12)", fontsize=9)
ymax = max(d.max() for d in data)
y1 = ymax + 250
ax.annotate("", xy=(1, y1), xytext=(3, y1), arrowprops=dict(arrowstyle="-", lw=0.8))
ax.text(2, y1 + 60, "$p=0.0037$", ha="center", fontsize=7.5)
ax.set_ylim(-5200, y1 + 260)
ax.tick_params(axis="x", labelsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(f"/tmp/fig_20f_distribution_ieee.{ext}", bbox_inches="tight")
plt.close(fig)
print("fig2 done")

# ---------------- Fig 3: 10F vs 20F transfer (two panels) ----------------
z10 = np.concatenate([load("zoneaux_s42_main"), load("zoneaux_s360"), load("zoneaux_s712")])
sd_eta10, sd_near10 = load("sd_eta"), load("sd_nearest")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), dpi=150, sharey=True)
for ax, (ttl, pr, et, nr) in zip(axes, [
    ("10F (training building)", z10.mean(), sd_eta10.mean(), sd_near10.mean()),
    ("20F (unseen building)", za20.mean(), sd_eta20.mean(), sd_near20.mean()),
]):
    vals = [nr, et, pr]
    errs3 = [sd_near10.std()/np.sqrt(12) if ttl.startswith("10") else sd_near20.std()/np.sqrt(12),
             sd_eta10.std()/np.sqrt(12) if ttl.startswith("10") else sd_eta20.std()/np.sqrt(12),
             z10.std()/np.sqrt(36) if ttl.startswith("10") else za20.std()/np.sqrt(36)]
    bars = ax.bar(["SD-\nnearest", "SD-ETA", "Zone-\nAux"], vals, yerr=errs3,
                  capsize=3, color=[OKABE["grey"], OKABE["grey"], OKABE["blue"]],
                  edgecolor="black", linewidth=0.6, error_kw=dict(lw=0.8))
    ax.axhline(0, color="black", lw=0.8)
    for b, r in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, r - 90, f"{r:,.0f}", ha="center", va="top", fontsize=7.5)
    ax.set_title(ttl, fontsize=9)
    ax.set_ylim(-4300, 200)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Mean per-episode reward")
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(f"/tmp/fig_10f_20f_transfer_ieee.{ext}", bbox_inches="tight")
plt.close(fig)
print("fig3 done")

# ---------------- copy to paper dirs (main repo + paper repo) ----------------
for dst in ["paper_access/figs", "paper_csmag/figs", "../eet-alsharif-paper/paper_access/figs",
            "../eet-alsharif-paper/paper_csmag/figs", "../eet-alsharif-paper/figs_ieee"]:
    os.makedirs(dst, exist_ok=True)
    for f in ["fig_20f_main_results_ieee.pdf", "fig_20f_main_results_ieee.png",
              "fig_20f_distribution_ieee.pdf", "fig_20f_distribution_ieee.png",
              "fig_10f_20f_transfer_ieee.pdf", "fig_10f_20f_transfer_ieee.png"]:
        shutil.copy(f"/tmp/{f}", f"{dst}/{f}")
    print(f"copied -> {dst}")
