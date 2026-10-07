#!/usr/bin/env python
"""Validation-metric and success-rate plots for every run under experiment_data/.

For each of the 5 fine-tuning runs (OpenVLA x{150,342 traj}, pi0.5 x{150,342
traj}, SmolVLA-150) this loads that folder's densest available per-checkpoint
metric log plus its checkpoint_ranking_report's quality_by_step (LIBERO
success rate), applies the same lower-is-better early-stopping rule used by
checkpoint_selection.py (tex eq. k*_m = min{k: m(k) <= min_j m(j) + eta_m}),
and writes two PNGs into that folder:

  success_vs_val_metrics_timeseries.png
      success rate + every offline metric vs. training step, small multiples
      on a shared step axis, with a star at each metric's selected checkpoint.
  success_vs_val_metrics_scatter.png
      metric value vs. success rate at the evaluated checkpoints, one panel
      per metric, annotated with Spearman rho.

Metric identity -> color is fixed across all 5 runs (validated categorical
palette; dataviz skill references/palette.md) so the same metric always reads
as the same color everywhere in the report.
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Canonical metric identity: fixed color / label / direction, shared by every
# run (only the subset a given run actually logs gets plotted for that run).
# val_accuracy reuses the success-rate blue since the two never share an axes
# (both are 0-1, higher-is-better fractions) and slots 2-8 cover the seven
# distance/loss metrics.
# ---------------------------------------------------------------------------
METRIC_SPEC = {
    "val_l2_loss":         dict(label="L2 loss",        color="#008300", lower_better=True),
    "val_ce_loss":         dict(label="CE loss",         color="#e87ba4", lower_better=True),
    "val_l1_loss":         dict(label="L1 loss",         color="#eda100", lower_better=True),
    "val_cosine_distance": dict(label="Cosine distance", color="#1baf7a", lower_better=True),
    "val_dtw_distance":    dict(label="DTW distance",    color="#eb6834", lower_better=True),
    "val_ot_distance":     dict(label="OT distance",     color="#4a3aa7", lower_better=True),
    "val_nll":             dict(label="NLL",             color="#e34948", lower_better=True),
    "val_accuracy":        dict(label="Token accuracy",  color="#2a78d6", lower_better=False),
}
# keys that are redundant with another canonical key and should be dropped
# rather than plotted as their own panel:
#   val_log_likelihood duplicates val_nll (unnormalized vs. normalized token NLL)
#   val_fm_loss is logged under the *same value* as val_nll for flow-matching
#   policies (their "NLL" *is* the flow-matching validation loss) -- keep one
#   panel, relabeled "NLL / FM loss".
DROP_KEYS = {"val_log_likelihood"}
FM_ALIAS_KEY = "val_fm_loss"

C_SUCCESS = "#2a78d6"
INK, INK2, GRID, GUIDE = "#0b0b0b", "#52514e", "#e4e3df", "#a8a7a1"
# sequential blue ramp (light->dark), dataviz skill's default sequential hue --
# used to encode training step as scatter-point fill instead of a text label.
STEP_CMAP = LinearSegmentedColormap.from_list(
    "step_seq", ["#cde2fb", "#3987e5", "#0d366b"]
)

RUNS = [
    dict(
        folder="openvla_150_traj",
        title="OpenVLA-150 (150 traj)",
        metric_source="metrics_history.json",
        report="checkpoint_ranking_report.json",
    ),
    dict(
        folder="openvla_342_traj",
        title="OpenVLA-full (342 traj)",
        metric_source="metrics_history_enriched.json",
        report="checkpoint_ranking_report.json",
    ),
    dict(
        folder="pi0.5_150_traj",
        title=r"$\pi_{0.5}$-150 (150 traj)",
        metric_source="metrics_history.json",
        report="checkpoint_ranking_report.json",
    ),
    dict(
        folder="pi0.5_342_traj",
        title=r"$\pi_{0.5}$-full (342 traj)",
        metric_source="metrics_history_full.json",
        report="checkpoint_ranking_report.json",
        # This run logs offline metrics to step 73499 but its last sim rollout is at 52499.
        # Selecting past the evaluated range forced Q(k*) to be read 3.5k-8k steps away from k*,
        # which collapsed DTW/OT/cosine/L1/L2 onto one shared Q and one degenerate regret. The
        # selection range is truncated to the evaluated range; rho is unaffected, since it only
        # ever used checkpoints that have both a metric value and a Q(k).
        max_step=52499,
    ),
    dict(
        folder="smolvla_150_traj",
        title="SmolVLA-150 (150 traj)",
        metric_source="metrics_history.json",
        report="checkpoint_ranking_report_smolvla.json",
    ),
]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK, "figure.dpi": 130,
})


def load_metric_history(folder: Path, source: str, max_step=None) -> dict:
    """Returns {step: {metric_key: value}} from a metrics_history*.json,
    which is a list of {"step": k, <metric>: value, ...} records. max_step
    truncates the curve (see the pi0.5-full entry in RUNS for why)."""
    rows = json.loads((folder / source).read_text())
    history = {}
    for row in rows:
        step = int(row["step"])
        if max_step is not None and step > max_step:
            continue
        history[step] = {k: v for k, v in row.items() if k != "step"}
    return dict(sorted(history.items()))


def resolve_metrics(history: dict) -> list:
    """Which canonical metric keys this run actually logged, in a fixed
    (palette-slot) order, with the flow-matching NLL alias folded in."""
    present = set()
    has_fm_alias = False
    for vals in history.values():
        present |= set(vals.keys())
        if FM_ALIAS_KEY in vals:
            has_fm_alias = True
    present -= DROP_KEYS
    present.discard(FM_ALIAS_KEY)
    ordered = [k for k in METRIC_SPEC if k in present]
    return ordered, has_fm_alias


def select_checkpoint(steps, raw_values, lower_better, tol_frac=0.01):
    v = np.asarray(raw_values, dtype=float)
    best = v.min() if lower_better else v.max()
    eta = tol_frac * abs(v[0] - best)
    for s, x in zip(steps, v):
        ok = (x <= best + eta) if lower_better else (x >= best - eta)
        if ok:
            return s
    return steps[int(np.argmin(v) if lower_better else np.argmax(v))]


def oriented(raw_values, lower_better):
    v = np.asarray(raw_values, dtype=float)
    return -v if lower_better else v


def mark_evals(ax, eval_steps):
    for s in eval_steps:
        ax.axvline(s, color=GUIDE, lw=0.8, ls=(0, (4, 3)), zorder=1)


SUMMARY = []

for run in RUNS:
    folder = BASE / run["folder"]
    history = load_metric_history(folder, run["metric_source"], run.get("max_step"))
    report = json.loads((folder / run["report"]).read_text())
    qbs = {int(k): v for k, v in report["quality_by_step"].items()}

    metric_keys, has_fm_alias = resolve_metrics(history)
    steps = np.array(sorted(history))
    eval_steps = np.array(sorted(qbs))
    success = np.array([qbs[s] for s in eval_steps])
    best_step = int(eval_steps[np.argmax(success)])

    labels = {
        k: (METRIC_SPEC[k]["label"] + " / FM loss" if (k == "val_nll" and has_fm_alias)
            else METRIC_SPEC[k]["label"])
        for k in metric_keys
    }

    # per-metric selection + Spearman against Q, over checkpoints that have
    # both a value and a Q(k) -- same restriction as checkpoint_selection.py
    stats = {}
    for m in metric_keys:
        lb = METRIC_SPEC[m]["lower_better"]
        m_steps = [s for s in steps if m in history[s]]
        m_vals = [history[s][m] for s in m_steps]
        sel = select_checkpoint(m_steps, m_vals, lb)
        common = [s for s in m_steps if s in qbs]
        if len(common) >= 2:
            x = oriented([history[s][m] for s in common], lb)
            y = [qbs[s] for s in common]
            rho, pval = spearmanr(x, y)
        else:
            rho, pval = float("nan"), float("nan")
        # Q(k) exists only at the sim-evaluated checkpoints, so Q(k*) is read at the evaluated
        # step nearest k* -- the same convention the ranking table uses
        # (crossmodel_early_stopping.py). q_step is reported next to the regret whenever it is
        # not k* itself, so a snapped regret can never be mistaken for one measured at the
        # selected checkpoint.
        q_step = int(min(qbs, key=lambda s: abs(s - sel)))
        regret = success.max() - qbs[q_step]
        stats[m] = dict(steps=m_steps, vals=m_vals, sel=sel, q_step=q_step,
                         sel_val=history[sel][m], rho=rho, pval=pval, regret=regret)
        SUMMARY.append(dict(
            run=run["folder"], title=run["title"], metric=m, label=labels[m],
            n_checkpoints=len(m_steps), n_eval=len(common),
            selected_checkpoint=int(sel), spearman_rho=None if rho != rho else round(float(rho), 4),
            p_value=None if pval != pval else round(float(pval), 4),
            regret=None if regret != regret else round(float(regret), 4),
            regret_q_step=q_step,
        ))

    n = len(metric_keys)

    # ================= Figure 1: time series, small multiples ==============
    fig, axes = plt.subplots(n + 1, 1, figsize=(9, 2.0 * (n + 1) + 1.2),
                              sharex=True, gridspec_kw={"hspace": 0.25})

    ax = axes[0]
    mark_evals(ax, eval_steps)
    ax.plot(eval_steps, success, "-o", color=C_SUCCESS, lw=2, ms=6,
            mec="white", mew=1.1, zorder=3)
    ax.scatter([best_step], [success.max()], s=170, marker="*",
               color="#eda100", edgecolor=INK, lw=0.8, zorder=4,
               label=f"best SR = {success.max():.2f} @ {best_step}")
    ax.set_ylabel("Success rate", color=INK)
    ax.set_ylim(-0.05, max(0.9, success.max() + 0.15))
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(f"{run['title']}: LIBERO-goal success rate vs. validation "
                 f"metrics over training", fontsize=11, fontweight="bold",
                 loc="left", pad=10)

    for ax, m in zip(axes[1:], metric_keys):
        spec = METRIC_SPEC[m]
        st = stats[m]
        mark_evals(ax, eval_steps)
        ax.plot(st["steps"], st["vals"], "-", color=spec["color"], lw=1.6, zorder=3)
        ax.plot(st["steps"], st["vals"], ".", color=spec["color"], ms=3, zorder=3)
        ax.scatter([st["sel"]], [st["sel_val"]], s=140, marker="*",
                   color=spec["color"], edgecolor=INK, lw=0.8, zorder=4)
        arrow = "↓" if spec["lower_better"] else "↑"
        reg_txt = (f"{st['regret']:.2f}" if st["regret"] == st["regret"] else "n/a")
        if st["q_step"] != st["sel"]:
            reg_txt += f" (Q@{st['q_step']})"
        ax.set_ylabel(f"{labels[m]}  ({arrow})", color=INK)
        ax.annotate(
            f"selected ckpt {st['sel']}\nSpearman ρ={st['rho']:.2f}   regret={reg_txt}",
            xy=(0.985, 0.92), xycoords="axes fraction", ha="right", va="top",
            fontsize=8, color=INK2,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID, lw=0.8),
        )

    axes[-1].set_xlabel("Training step", color=INK)
    fig.text(
        0.5, 0.005,
        "Dashed vertical lines = evaluated (sim-rollout) checkpoints. "
        "★ = checkpoint each metric would select (metrics: lower is better "
        "unless marked ↑); regret = best success rate minus the success rate "
        "at that metric's selected checkpoint. \"Q@step\" marks a regret read "
        "at the nearest evaluated checkpoint because the selected one was "
        "never sim-evaluated.",
        ha="center", fontsize=7.5, color=INK2, wrap=True,
    )
    out1 = folder / "success_vs_val_metrics_timeseries.png"
    fig.savefig(out1, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out1)

    # ================= Figure 2: correlation scatter ========================
    # Training step is encoded as fill color (shared sequential blue ramp, one
    # colorbar for the whole figure) rather than a text label on every dot --
    # at up to 32 evaluated checkpoints per run, per-point labels overlap into
    # unreadable clutter. Point *edge* stays the metric's categorical color,
    # linking each panel back to its line color in the timeseries figure.
    # Only the trajectory endpoints (first/last evaluated checkpoint) are
    # labeled directly, per the "label the endpoint, not every point" rule.
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig2, axs = plt.subplots(nrows, ncols, figsize=(4.9 * ncols, 3.7 * nrows))
    axs = np.atleast_1d(axs).ravel()
    step_vmin, step_vmax = int(eval_steps.min()), int(eval_steps.max())
    mappable = None
    for ax, m in zip(axs, metric_keys):
        spec = METRIC_SPEC[m]
        x = np.array([history[s][m] for s in eval_steps if m in history[s]])
        s_used = np.array([s for s in eval_steps if m in history[s]])
        y = np.array([qbs[s] for s in s_used])
        mappable = ax.scatter(x, y, c=s_used, cmap=STEP_CMAP, vmin=step_vmin, vmax=step_vmax,
                               s=80, edgecolor=spec["color"], linewidths=1.4, zorder=3)
        for xi, yi, s in ((x[0], y[0], s_used[0]), (x[-1], y[-1], s_used[-1])):
            ax.annotate(str(s), (xi, yi), fontsize=7, color=INK2,
                        xytext=(5, 4), textcoords="offset points")
        # rho_m = Spearman(-m(k), Q(k)) for lower-is-better metrics (tex eq.) so
        # "the metric tracks success" reads as positive rho, matching the
        # timeseries panel above and every rho reported elsewhere in this report.
        rho, p = (spearmanr(oriented(x, spec["lower_better"]), y)
                  if len(x) >= 2 else (float("nan"), float("nan")))
        arrow = "↓ better" if spec["lower_better"] else "↑ better"
        ax.set_xlabel(f"{labels[m]}  ({arrow})", color=INK)
        ax.set_ylabel("Success rate", color=INK)
        ax.set_title(f"ρ={rho:.2f}  (p={p:.3f}, n={len(x)})", fontsize=10,
                     loc="left", color=INK)
        ax.set_ylim(-0.05, max(0.9, success.max() + 0.15))
    for ax in axs[n:]:
        ax.axis("off")
    fig2.suptitle(
        f"{run['title']}: does the validation metric track success? "
        f"({len(eval_steps)} evaluated checkpoints)",
        fontsize=12, fontweight="bold",
    )
    fig2.tight_layout(rect=(0, 0, 0.92, 0.96))
    cbar = fig2.colorbar(mappable, ax=axs[:n].tolist(), shrink=0.9, pad=0.02, aspect=35,
                          location="right")
    cbar.set_label("Training step  (dot fill; ring color = metric, matches "
                    "timeseries figure)", color=INK, fontsize=8.5)
    cbar.ax.tick_params(colors=INK2, labelsize=8)
    out2 = folder / "success_vs_val_metrics_scatter.png"
    fig2.savefig(out2, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print("wrote", out2)

summary_path = BASE / "validation_metrics_summary.json"
summary_path.write_text(json.dumps(SUMMARY, indent=2))
print("wrote", summary_path)
