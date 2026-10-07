#!/usr/bin/env python
"""Visualize success-rate vs validation metrics over training steps.

No dual-axis charts (different scales): success rate and the validation
distances are shown as small multiples sharing a common training-step x-axis,
plus a correlation-scatter panel per metric.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

RUN = Path("/scratch/work/zhangy50/RL/Spot_VLA/openvla/runs/MERGED_33999_continuous_7_8")
report = json.loads((RUN / "checkpoint_ranking_report.json").read_text())

# ---- data ----------------------------------------------------------------
mh = report["metric_history"]
steps = np.array(sorted(int(s) for s in mh))
METRICS = ["val_dtw_distance", "val_ot_distance", "val_cosine_distance", "val_nll"]
LABELS = {
    "val_dtw_distance": "Val DTW distance",
    "val_ot_distance": "Val OT distance",
    "val_cosine_distance": "Val cosine distance",
    "val_nll": "Val NLL",
}
curves = {m: np.array([mh[str(s)][m] for s in steps]) for m in METRICS}

qbs = report["quality_by_step"]
eval_steps = np.array(sorted(int(s) for s in qbs))
success = np.array([qbs[str(s)] for s in eval_steps])
best_step = int(eval_steps[np.argmax(success)])  # true best by success rate

ranking = {r["metric"]: r for r in report["ranking"]}

# ---- colors (from validated dataviz palette) -----------------------------
C_SUCCESS = "#2a78d6"  # blue
C_METRIC = {
    "val_dtw_distance": "#eb6834",   # orange
    "val_ot_distance": "#4a3aa7",    # violet
    "val_cosine_distance": "#1baf7a",  # aqua
    "val_nll": "#e34948",            # red
}
INK, INK2, GRID, GUIDE = "#0b0b0b", "#52514e", "#e4e3df", "#a8a7a1"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK, "figure.dpi": 130,
})

# ==========================================================================
# Figure 1 — time series, small multiples, shared step axis
# ==========================================================================
fig, axes = plt.subplots(5, 1, figsize=(9, 11), sharex=True,
                         gridspec_kw={"hspace": 0.18})

def mark_evals(ax):
    # dashed vertical guides at the 11 evaluated checkpoints (unevenly spaced);
    # distinct from the faint reference grid at the round axis ticks.
    for s in eval_steps:
        ax.axvline(s, color=GUIDE, lw=0.8, ls=(0, (4, 3)), zorder=1)

# success rate panel
ax = axes[0]
mark_evals(ax)
ax.plot(eval_steps, success, "-o", color=C_SUCCESS, lw=2, ms=7,
        mec="white", mew=1.2, zorder=3)
ax.scatter([best_step], [success.max()], s=180, marker="*",
           color="#eda100", edgecolor=INK, lw=0.8, zorder=4,
           label=f"best SR = {success.max():.2f} @ {best_step}")
ax.set_ylabel("Success rate", color=INK)
ax.set_ylim(-0.05, 0.9)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("LIBERO-goal success rate vs. validation metrics over training "
             "(MERGED_33999_continuous_7_8)", fontsize=11, fontweight="bold",
             loc="left", pad=10)

# validation metric panels
for ax, m in zip(axes[1:], METRICS):
    mark_evals(ax)
    ax.plot(steps, curves[m], "-", color=C_METRIC[m], lw=1.6, zorder=3)
    ax.plot(steps, curves[m], ".", color=C_METRIC[m], ms=3, zorder=3)
    sel = ranking[m]["selected_checkpoint"]
    sel_val = mh[str(sel)][m]
    ax.scatter([sel], [sel_val], s=150, marker="*", color=C_METRIC[m],
               edgecolor=INK, lw=0.8, zorder=4)
    rho = ranking[m]["spearman_rho"]
    reg = ranking[m]["regret"]
    ax.set_ylabel(LABELS[m] + "  (↓)", color=INK)
    ax.annotate(f"selected ckpt {sel}\nSpearman ρ={rho:.2f}   regret={reg:.2f}",
                xy=(0.985, 0.92), xycoords="axes fraction", ha="right", va="top",
                fontsize=8, color=INK2,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID, lw=0.8))

axes[-1].set_xlabel("Training step", color=INK)
fig.text(0.5, 0.005,
         "Faint solid lines at round values = reference grid (axis ticks).  "
         "Dashed vertical lines = the 11 evaluated checkpoints (unevenly spaced: "
         "~every 4000 steps, plus an extra probe at 21499 and the final merged "
         "checkpoint 33999).  ★ = checkpoint each metric would select "
         "(metrics: lower is better).",
         ha="center", fontsize=7.5, color=INK2, wrap=True)
fig.savefig(RUN / "success_vs_val_metrics_timeseries.png",
            bbox_inches="tight", facecolor="white")
print("wrote", RUN / "success_vs_val_metrics_timeseries.png")

# ==========================================================================
# Figure 2 — correlation scatter (validation metric vs success rate)
# ==========================================================================
fig2, axs = plt.subplots(2, 2, figsize=(9, 8))
axs = axs.ravel()
for ax, m in zip(axs, METRICS):
    x = np.array([mh[str(s)][m] for s in eval_steps])
    y = success
    ax.scatter(x, y, s=70, color=C_METRIC[m], edgecolor="white", lw=1.2, zorder=3)
    for xi, yi, s in zip(x, y, eval_steps):
        ax.annotate(str(s), (xi, yi), fontsize=6.5, color=INK2,
                    xytext=(3, 3), textcoords="offset points")
    rho, p = spearmanr(x, y)
    ax.set_xlabel(LABELS[m] + "  (↓ better)", color=INK)
    ax.set_ylabel("Success rate", color=INK)
    ax.set_title(f"ρ={rho:.2f}  (p={p:.3f})", fontsize=10, loc="left", color=INK)
    ax.set_ylim(-0.05, 0.9)
fig2.suptitle("Does the validation metric track success? "
              "(11 evaluated checkpoints)", fontsize=12, fontweight="bold")
fig2.tight_layout(rect=(0, 0, 1, 0.97))
fig2.savefig(RUN / "success_vs_val_metrics_scatter.png",
             bbox_inches="tight", facecolor="white")
print("wrote", RUN / "success_vs_val_metrics_scatter.png")
