"""
build_merged_run.py

Stitch several *continuously fine-tuned* OpenVLA run dirs (each a resumed stage of one long
training trajectory) into a single logical run so that select_and_eval_checkpoints.py can evaluate
and rank ALL their checkpoints on one global-step axis -- the axis the tex's early-stopping
experiment (wenyan_experiment_instruction.tex sections 8, 12) actually assumes: one sequence
theta_0..theta_K saved at fixed intervals.

Why not just rename the step_* folders into one directory?
  1. Every stage restarts its local step numbering at 0, so all stages share step_499, step_999,
     ... -- a flat merge would collide and silently overwrite checkpoints.
  2. The offline metrics (DTW/OT/cosine/NLL) live in each stage's OWN wandb run, keyed by that
     run's LOCAL _step. Renaming the checkpoint dirs to global steps without also re-keying the
     metrics breaks the driver's history<->checkpoint matching (it would skip every checkpoint as
     "metrics reference steps with no saved checkpoint").
  3. It's destructive to run dirs that are currently working.

So instead this builds a NON-DESTRUCTIVE merged view:
  * a new run dir full of SYMLINKS  step_<global> -> <stage>/step_<local>  (unique global names,
    no collisions, originals untouched; find_available_checkpoints() and run_libero_eval.py's
    --adapter_dir both follow symlinks fine).
  * a generated metrics_history.json with GLOBAL steps, assembled by pulling each stage's wandb
    metrics (via checkpoint_selection.load_metrics_history) and adding that stage's offset. Having
    it on disk means the downstream driver needs no wandb flags at all.

Global-step offset for a stage = how many real training steps precede this stage's local step 0,
i.e. the cumulative step COUNT completed by the predecessor at the checkpoint this stage resumed
from. Note the checkpoints are 0-indexed: step_K is saved after K+1 gradient steps (step_8499 =
8500 steps), so a resume from step_8499 contributes an offset of 8500, not 8499. See STAGES /
stage_offsets() below.

Usage:
    # Build the merged view (pulls metrics from wandb -> writes metrics_history.json):
    python experiments/robot/libero/build_merged_run.py \
        --wandb_entity yibo-zhang --wandb_project openvla_spot \
        --out_dir /scratch/.../openvla/runs/MERGED_8499_continuous

    # Symlinks only, skip the wandb metric pull (offline / no creds):
    python experiments/robot/libero/build_merged_run.py --out_dir ... --skip_metrics

    # Then evaluate + rank every Nth of the 68 checkpoints on the continuous axis:
    python experiments/robot/libero/select_and_eval_checkpoints.py \
        --run_dir /scratch/.../openvla/runs/MERGED_8499_continuous \
        --task_suite_name libero_goal --num_trials_per_task 10 --eval_every_n 5
"""

import argparse
import json
import re
import shutil
from pathlib import Path

from checkpoint_selection import METRIC_KEYS, load_metrics_history

# build_merged_run.py -> libero -> robot -> experiments -> openvla/
OPENVLA_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = OPENVLA_ROOT / "runs"

# The continuous fine-tuning chain, earliest stage first. `resume_at_prev` is the LOCAL step of the
# PREVIOUS stage's checkpoint that this stage was resumed from (ignored for the first stage, which
# starts from the base model). Verified against the folders: stage k's predecessor really does
# contain a step_<resume_at_prev> checkpoint. Because checkpoints are 0-indexed (step_K = K+1
# completed steps), the offset contributed by resuming at step_K is K+1 -- see stage_offsets().
STAGES = [
    {
        "folder": "8499_new_openvla-7b+dataset+libero_goal_no_noops+b56+lr-0.0005+shf1000+lora-r32+dropout-0.0--image_aug",
        "wandb_run_id": "sfm8130e",
        "resume_at_prev": 0,
    },
    {
        "folder": "8499+2000_openvla-7b+dataset+libero_goal_no_noops+b56+lr-0.0005+shf1000+lora-r32+dropout-0.0--image_aug",
        "wandb_run_id": "s1wzhuua",
        "resume_at_prev": 8499,  # resumed from stage 1's step_8499
    },
    {
        "folder": "8500+2000+12000_time2026-07-08_02-56-07+openvla-7b+dataset+libero_goal_no_noops+b56+lr-0.0005+shf1000+adapter-step_1999+lora-r32+dropout-0.0--image_aug",
        "wandb_run_id": "fp1ahq16",
        "resume_at_prev": 1999,  # resumed from stage 2's step_1999
    },
    {
        "folder": "8500+2000+12000+11500_time2026-07-08_14-12-29+openvla-7b+dataset+libero_goal_no_noops+b56+lr-0.0005+shf1000+adapter-step_11999+lora-r32+dropout-0.0--image_aug",
        "wandb_run_id": "5nz5ic1k",
        "resume_at_prev": 11999,  # resumed from stage 3's step_11999
    },
]


def stage_offsets() -> list[int]:
    """Global-step offset added to each stage's local steps.

    offset[0] = 0 (first stage starts from the base model). For i>0, offset[i] = offset[i-1] plus
    the step COUNT of the predecessor's resume checkpoint, which is resume_at_prev + 1 because
    checkpoints are 0-indexed (step_K = K+1 completed steps). So resuming at step_8499 adds 8500.
    """
    offsets = [0]
    for st in STAGES[1:]:
        offsets.append(offsets[-1] + st["resume_at_prev"] + 1)
    return offsets


def find_local_steps(stage_dir: Path) -> dict[int, Path]:
    """Maps local step -> checkpoint dir for every step_<k> subfolder of one stage."""
    out = {}
    for p in stage_dir.iterdir():
        m = re.fullmatch(r"step_(\d+)", p.name)
        if p.is_dir() and m:
            out[int(m.group(1))] = p
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", required=True, help="Merged run dir to create (holds symlinks + metrics_history.json)")
    parser.add_argument("--wandb_entity", default=None, help="wandb entity for pulling each stage's offline metrics")
    parser.add_argument("--wandb_project", default=None, help="wandb project for pulling each stage's offline metrics")
    parser.add_argument("--skip_metrics", action="store_true", help="Only build the checkpoint symlinks; do not pull wandb metrics")
    parser.add_argument("--force", action="store_true", help="Overwrite out_dir if it already exists")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        if not args.force:
            raise SystemExit(f"{out_dir} already exists. Pass --force to rebuild it.")
        # Only ever remove a dir we built (all-symlinks + our generated json); never a real run dir.
        real_ckpts = [p for p in out_dir.iterdir() if re.fullmatch(r"step_\d+", p.name) and not p.is_symlink()]
        if real_ckpts:
            raise SystemExit(
                f"Refusing to --force-overwrite {out_dir}: it contains real (non-symlink) checkpoint "
                f"dirs ({[p.name for p in real_ckpts[:3]]}...). This does not look like a merged view."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    offsets = stage_offsets()
    global_to_local: dict[int, tuple[int, int]] = {}  # global_step -> (stage_idx, local_step), for reporting

    # 1) Symlink every stage's checkpoints under global names.
    n_links = 0
    for idx, (st, offset) in enumerate(zip(STAGES, offsets)):
        stage_dir = RUNS_DIR / st["folder"]
        if not stage_dir.is_dir():
            raise SystemExit(f"Stage {idx} folder not found: {stage_dir}")
        local_steps = find_local_steps(stage_dir)
        if not local_steps:
            raise SystemExit(f"Stage {idx} has no step_* checkpoints: {stage_dir}")
        for local_step, ckpt in sorted(local_steps.items()):
            g = offset + local_step
            if g in global_to_local:
                prev = global_to_local[g]
                raise SystemExit(
                    f"Global step collision at {g}: stage {idx} local {local_step} vs "
                    f"stage {prev[0]} local {prev[1]}. Check STAGES offsets."
                )
            global_to_local[g] = (idx, local_step)
            (out_dir / f"step_{g}").symlink_to(ckpt)  # absolute target (ckpt is resolved)
            n_links += 1

    print(f"Linked {n_links} checkpoints across {len(STAGES)} stages into {out_dir}")
    for idx, (st, offset) in enumerate(zip(STAGES, offsets)):
        gsteps = sorted(g for g, (s, _) in global_to_local.items() if s == idx)
        print(f"  stage {idx} ({st['wandb_run_id']}): offset +{offset} -> global {gsteps[0]}..{gsteps[-1]} ({len(gsteps)} ckpts)")

    # Copy dataset_statistics.json (identical across stages -- same dataset/seed) so downstream code
    # that expects it next to the run finds it.
    stats = RUNS_DIR / STAGES[0]["folder"] / "dataset_statistics.json"
    if stats.is_file():
        shutil.copy(stats, out_dir / "dataset_statistics.json")

    # 2) Assemble the merged, global-step metrics_history.json from each stage's wandb metrics.
    if args.skip_metrics:
        print("--skip_metrics set: wrote symlinks only. Re-run without it (with --wandb_entity/"
              "--wandb_project) to generate metrics_history.json before ranking.")
        return

    if not (args.wandb_entity and args.wandb_project):
        raise SystemExit("Need --wandb_entity and --wandb_project to pull metrics (or pass --skip_metrics).")

    records: list[dict] = []
    for idx, (st, offset) in enumerate(zip(STAGES, offsets)):
        stage_dir = RUNS_DIR / st["folder"]
        print(f"Pulling metrics for stage {idx} from wandb run {st['wandb_run_id']} ...")
        history = load_metrics_history(str(stage_dir), args.wandb_entity, args.wandb_project, st["wandb_run_id"])
        local_ckpt_steps = set(find_local_steps(stage_dir))
        for local_step, vals in sorted(history.items()):
            # wandb logs metrics at many _steps; keep only those that correspond to a saved checkpoint,
            # so the merged history lines up 1:1 with the symlinked step_<global> dirs.
            if local_step not in local_ckpt_steps:
                continue
            rec = {"step": offset + local_step}
            rec.update({k: v for k, v in vals.items() if k in METRIC_KEYS})
            records.append(rec)

    records.sort(key=lambda r: r["step"])
    (out_dir / "metrics_history.json").write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} metric records to {out_dir / 'metrics_history.json'}")

    linked = set(global_to_local)
    with_metrics = {r["step"] for r in records}
    missing = sorted(linked - with_metrics)
    if missing:
        print(f"WARNING: {len(missing)} linked checkpoints have no offline metrics (global steps): {missing}")


if __name__ == "__main__":
    main()
