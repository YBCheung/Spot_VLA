#!/bin/bash
# filepath: /scratch/work/zhangy50/RL/Spot_VLA/openvla/experiments/robot/libero/run_parallel_eval.sh

#SBATCH --job-name=libero_eval_parallel
#SBATCH --partition=gpu-h100-80g,gpu-h200-141g-ellis,gpu-h200-141g-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=0  # Adjust based on number of adapters
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err

# Create logs directory if it doesn't exist
mkdir -p logs

# List of adapter directories - ADD YOUR ADAPTER PATHS HERE
ADAPTER_DIRS=(
        # "/script/work/zhangy50/RL/Spot_VLA/openvla/runs/5800_seg_+dataset+libero_goal_no_noops+b56+lr-0.0005+shf1000+lora-r32+dropout-0.0--image_aug/val_loss"
        "/script/work/zhangy50/RL/Spot_VLA/openvla/runs/5800_seg_+dataset+libero_goal_no_noops+b56+lr-0.0005+shf1000+lora-r32+dropout-0.0--image_aug/val_l1"
        # "/script/work/zhangy50/RL/Spot_VLA/openvla/runs/5800_seg_+dataset+libero_goal_no_noops+b56+lr-0.0005+shf1000+lora-r32+dropout-0.0--image_aug/val_l2"
        # "/script/work/zhangy50/RL/Spot_VLA/openvla/runs/5800_seg_+dataset+libero_goal_no_noops+b56+lr-0.0005+shf1000+lora-r32+dropout-0.0--image_aug/val_dtw_seg"
        # "/script/work/zhangy50/RL/Spot_VLA/openvla/runs/5800_seg_+dataset+libero_goal_no_noops+b56+lr-0.0005+shf1000+lora-r32+dropout-0.0--image_aug/val_ot_seg"
        # "/script/work/zhangy50/RL/Spot_VLA/openvla/runs/5800_seg_+dataset+libero_goal_no_noops+b56+lr-0.0005+shf1000+lora-r32+dropout-0.0--image_aug/val_cosine_distance"
        # "/script/work/zhangy50/RL/Spot_VLA/openvla/runs/5800+5000_openvla-7b+dataset+libero_goal_no_noops+b56+lr-0.0005+shf1000+lora-r32+dropout-0.0--image_aug/val_dtw"
        # "/script/work/zhangy50/RL/Spot_VLA/openvla/runs/5800+5000_openvla-7b+dataset+libero_goal_no_noops+b56+lr-0.0005+shf1000+lora-r32+dropout-0.0--image_aug/val_accuracy"
        # "/script/work/zhangy50/RL/Spot_VLA/openvla/runs/5800+5000_openvla-7b+dataset+libero_goal_no_noops+b56+lr-0.0005+shf1000+lora-r32+dropout-0.0--image_aug/val_ot"
)

# Get the adapter directory for this array job
ADAPTER_DIR=${ADAPTER_DIRS[$SLURM_ARRAY_TASK_ID]}

echo "Starting evaluation for adapter: $ADAPTER_DIR"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $HOSTNAME"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"

# Load necessary modules (adjust based on your cluster setup)
module load python/3.9
module load cuda/11.8

# Activate your conda environment (adjust the name)
source activate openvla

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Run the evaluation
python experiments/robot/libero/run_libero_eval.py  \
    --model_family openvla \
    --pretrained_checkpoint None \
    --task_suite_name libero_goal \
    --center_crop True \
    --load_from_adapter True \
    --adapter_dir "$ADAPTER_DIR" \
    --num_trials_per_task 5 \
    --seed 7 \
    --run_id_note "parallel_eval_${SLURM_ARRAY_TASK_ID}" \
    --use_wandb False

echo "Evaluation completed for adapter: $ADAPTER_DIR"