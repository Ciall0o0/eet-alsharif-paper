# make_fig_softlabel.py — soft-label vs hard-label BC held-out reward curves (e10-e50)
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def curve(tag):
    d = json.load(open(f"results/{tag}_9seed.json"))
    eps = [10, 20, 30, 40, 50]
    m = [d[str(e)]["mean"] for e in eps]
    s = [d[str(e)]["std"] for e in eps]
    return eps, m, s

eps, m_h, s_h = curve("bc_lb_norm")     # hard-label BC-LB
_, m_t05, s_t05 = curve("bc_sl05")
_, m_t1, s_t1 = curve("bc_sl1")
_, m_t2, s_t2 = curve("bc_sl2")

fig, ax = plt.subplots(figsize=(6.6, 3.9))
styles = [
    (m_h, s_h, "black", "--", 1.4, "Hard-label clone (BC-LB)", True),
    (m_t05, s_t05, "#d62728", "-", 1.2, r"Soft-label $\tau=0.5$", False),
    (m_t1, s_t1, "#1f77b4", "-", 2.2, r"Soft-label $\tau=1$", True),
    (m_t2, s_t2, "#2ca02c", "-", 1.2, r"Soft-label $\tau=2$", False),
]
for m, s, c, ls, lw, lab, shade in styles:
    ax.plot(eps, m, ls, color=c, lw=lw, label=lab, marker="o", ms=3.5)
    if shade:
        ax.fill_between(eps, np.array(m) - np.array(s), np.array(m) + np.array(s),
                        color=c, alpha=0.12, linewidth=0)

# annotate epoch-20 values, staggered to avoid overlap
# blue(-9.1) above, black(-15.3) below, green(-37.5) above, red(-87.2) below
for m, c, dy, fs in ((m_t1, "#1f77b4", 14, 8), (m_h, "black", -15, 8),
                     (m_t2, "#2ca02c", 10, 8), (m_t05, "#d62728", -16, 8)):
    ax.annotate(f"{m[1]:.1f}", (20, m[1]), textcoords="offset points", xytext=(4, dy),
                fontsize=fs, color=c, fontweight="bold")

ax.set_xlabel("Training epoch", fontsize=8)
ax.set_ylabel("Held-out reward (12 h, nine seeds)", fontsize=8)
ax.set_xticks([10, 20, 30, 40, 50])
ax.set_xlim(8, 52)
ax.tick_params(labelsize=7)
ax.axvline(20, color="gray", lw=0.7, ls=":")
ax.text(20.6, ax.get_ylim()[1] * 0.97, "epoch 20", fontsize=6.5, color="gray")
ax.legend(fontsize=6.8, loc="lower left", framealpha=0.9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("paper_eaaai/figs/fig_softlabel_ieee.pdf")
fig.savefig("paper_eaaai/figs/fig_softlabel_ieee.png", dpi=300)
print("saved fig_softlabel_ieee.pdf/png")
print("hard e20:", m_h[1], "| t1 e20:", m_t1[1], "| t05 e20:", m_t05[1], "| t2 e20:", m_t2[1])
