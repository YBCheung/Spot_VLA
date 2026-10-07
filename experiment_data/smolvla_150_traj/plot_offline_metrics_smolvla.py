"""
plot_offline_metrics_smolvla.py

Visualize the SmolVLA offline validation-metric curves vs training progress, and overlay the
training loss so the train-vs-val gap (the overfitting signal) is visible.

  * Offline val metrics: read from <run_dir>/offline_metrics/step_*.json (the source of truth that
    select_and_eval_checkpoints_smolvla.py writes -- more robust than metrics_history.json, which
    may be mid-rewrite while an eval job runs).
  * Training loss: parsed from the finetune stdout log. lerobot logs one line every `log_freq`
    steps in order, so the printed `step:19K` abbreviation is ignored and the exact step is the
    line index * log_freq. This is the same flow-matching loss as val_fm_loss -> directly comparable.

Writes a PNG and prints a per-metric best-checkpoint / overfitting-shape summary.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VAL_METRICS = [
    ("val_fm_loss", "Val FM loss (NLL surrogate)"),
    ("val_dtw_distance", "Val DTW"),
    ("val_ot_distance", "Val OT"),
    ("val_cosine_distance", "Val cosine dist"),
    ("val_l1_loss", "Val L1"),
    ("val_l2_loss", "Val L2"),
]


def load_offline_metrics(run_dir: Path) -> dict[str, np.ndarray]:
    files = sorted((run_dir / "offline_metrics").glob("step_*.json"))
    if not files:
        raise SystemExit(f"No offline_metrics/step_*.json under {run_dir}")
    recs = [json.loads(f.read_text()) for f in files]
    recs.sort(key=lambda r: r["step"])
    out = {"step": np.array([r["step"] for r in recs], dtype=float)}
    for key, _ in VAL_METRICS:
        out[key] = np.array([r.get(key, np.nan) for r in recs], dtype=float)
    return out


def parse_train_loss(out_path: Path, log_freq: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, loss) for the training flow-matching loss.

    lerobot logs a line every `log_freq` steps in order; the i-th (0-based) train line is step
    (i+1)*log_freq. We match lines carrying a `loss:` field but not the eval/val markers.
    """
    if not out_path.is_file():
        return np.array([]), np.array([])
    losses = []
    for line in out_path.read_text(errors="ignore").splitlines():
        if "loss:" not in line or "grdn:" not in line:  # grdn: is only on training log lines
            continue
        m = re.search(r"loss:([0-9.]+)", line)
        if m:
            losses.append(float(m.group(1)))
    loss = np.array(losses, dtype=float)
    steps = (np.arange(len(loss)) + 1) * log_freq
    return steps, loss


def describe_shape(step: np.ndarray, y: np.ndarray) -> str:
    """Classify the val-metric curve: overfitting = interior min followed by a real rise."""
    if len(y) < 3 or np.all(np.isnan(y)):
        return "n/a"
    imin = int(np.nanargmin(y))
    best_step, best = step[imin], y[imin]
    final = y[-1]
    rise = (final - best) / (abs(best) + 1e-9)
    where = "early" if imin < len(y) * 0.33 else ("mid" if imin < len(y) * 0.66 else "late")
    if imin >= len(y) - 2:
        shape = "still improving (best at end)"
    elif rise > 0.05:
        shape = f"OVERFIT: +{rise*100:.0f}% from best by end"
    else:
        shape = f"plateau (only +{rise*100:.0f}% after best)"
    return f"best={best:.4f} @ step {int(best_step)} ({where}); {shape}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="/scratch/work/zhangy50/RL/Spot_VLA/lerobot/runs/smolvla_libero_goal_lora_r32_overtrain_100k")
    ap.add_argument("--train_out", default="/scratch/work/zhangy50/RL/Spot_VLA/lerobot/sbatch/finetune_smolvla_libero_lora.out")
    ap.add_argument("--log_freq", type=int, default=200)
    ap.add_argument("--out", default=None, help="PNG path (default <run_dir>/offline_metrics_plot.png)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    m = load_offline_metrics(run_dir)
    tstep, tloss = parse_train_loss(Path(args.train_out), args.log_freq)
    n = len(m["step"])
    print(f"Loaded {n} val checkpoints (steps {int(m['step'][0])}..{int(m['step'][-1])}); "
          f"{len(tloss)} train-loss points.")

    print("\n=== per-metric summary (lower is better) ===")
    for key, label in VAL_METRICS:
        print(f"  {label:<28} {describe_shape(m['step'], m[key])}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"SmolVLA LoRA overtraining -- offline val metrics vs training loss ({n} checkpoints)", fontsize=14)

    # (0,0) train vs val FM loss -- same units, so same axis. The overfitting diagnostic.
    ax = axes[0, 0]
    if len(tloss):
        ax.plot(tstep, tloss, color="tab:gray", lw=1, alpha=0.8, label="train FM loss")
    ax.plot(m["step"], m["val_fm_loss"], color="tab:red", marker="o", ms=3, label="val FM loss")
    ax.set_title("Train vs Val flow-matching loss")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3)

    # Remaining val metrics, one panel each; mark the min (best/selected checkpoint).
    panels = [(0, 1, "val_dtw_distance"), (0, 2, "val_ot_distance"),
              (1, 0, "val_cosine_distance"), (1, 1, "val_l1_loss"), (1, 2, "val_l2_loss")]
    label_of = dict(VAL_METRICS)
    for r, c, key in panels:
        ax = axes[r, c]
        y = m[key]
        ax.plot(m["step"], y, color="tab:blue", marker="o", ms=3)
        if not np.all(np.isnan(y)):
            imin = int(np.nanargmin(y))
            ax.axvline(m["step"][imin], color="tab:green", ls="--", lw=1)
            ax.plot(m["step"][imin], y[imin], "s", color="tab:green", ms=7,
                    label=f"best @ {int(m['step'][imin])}")
            ax.legend(fontsize=8)
        ax.set_title(label_of[key]); ax.set_xlabel("step"); ax.set_ylabel(key); ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(args.out) if args.out else run_dir / "offline_metrics_plot.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved figure to: {out}")


if __name__ == "__main__":
    main()
