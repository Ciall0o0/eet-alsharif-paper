# make_fig_segments.py — time-of-day segment heatmap (8 methods x 7 segments, mean wait)
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/segments_8x.json"))
SEGS = ["up_peak", "interfloor_am", "lunch_peak", "interfloor_mid",
        "down_peak", "off_peak", "interfloor_pm"]
SEG_LABELS = ["Up-peak", "Interfloor\n(AM)", "Lunch", "Interfloor\n(mid)", "Down-peak",
              "Off-peak", "Interfloor\n(PM)"]
METHODS = ["lb15", "lb30", "bc_lb", "bc_sdeta", "sd_eta", "sd_nearest", "ppo", "sector"]
M_LABELS = ["LB-15", "LB-30", "BC-LB", "BC", "SD-ETA", "SD-nearest", "PPO", "Sectoring"]

M = np.array([[d[m][s]["mean"] for s in SEGS] for m in METHODS])  # [8, 7]
L = np.log10(np.clip(M, 5, None))  # log scale, floor 5s

fig, ax = plt.subplots(figsize=(7.2, 3.6))
im = ax.imshow(L, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(7)); ax.set_xticklabels(SEG_LABELS, fontsize=7)
ax.set_yticks(range(8)); ax.set_yticklabels(M_LABELS, fontsize=8)
# value labels
for i in range(8):
    for j in range(7):
        v = M[i, j]
        color = "white" if L[i, j] > (L.min() + L.max()) / 2 else "black"
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color=color)
ax.set_xlabel("Time-of-day segment (12-h schedule)", fontsize=8)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("Mean wait (s, log)", fontsize=7)
cb.ax.tick_params(labelsize=6)
# highlight the best in each column
for j in range(7):
    best = int(np.argmin(M[:, j]))
    ax.add_patch(plt.Rectangle((j - 0.5, best - 0.5), 1, 1, fill=False,
                               edgecolor="green", lw=1.6))
fig.tight_layout()
fig.savefig("paper_eaaai/figs/fig_segments_ieee.pdf")
fig.savefig("paper_eaaai/figs/fig_segments_ieee.png", dpi=300)
print("saved fig_segments_ieee.pdf/png")
print("per-segment best:", [METHODS[int(np.argmin(M[:, j]))] for j in range(7)])
