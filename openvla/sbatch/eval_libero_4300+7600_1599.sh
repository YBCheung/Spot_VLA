#!/bin/bash
#SBATCH --time=05:30:00 # 14.33-15.08, suddenly stoped. only reserved 15 min. 
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4  # 2 is too tight, lead to low GPU utilization. maybe decrease task time. 
#SBATCH --mem=40G
#SBATCH --partition=gpu-h100-80g,gpu-h200-141g-ellis,gpu-h200-141g-short
#SBATCH --output=sbatch/0eval_629_4300+7600_1599.out

echo "Hello $USER! You are on node $HOSTNAME. SLURM_JOB_ID: $SLURM_JOB_ID. The time is $(date)."

module load mamba
module spider cuda
module load cuda/12.2.1
source activate openvla-spot

nvidia-smi

pwd

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM partition: $SLURM_JOB_PARTITION"

# # check GPU related versionss
# nvcc --version  # System CUDA compiler version
# echo "Pytorch ≥1.12, CUDA toolkit ≥11.0 for stable bfloat16 support"
# python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_bf16_supported())" 
# vla_path openvla/openvla-7b-finetuned-libero-object
torchrun --standalone --nnodes 1 --nproc-per-node 1 experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --load_from_adapter True \
  --vla_path openvla/openvla-7b \
  --num_trials_per_task 10 \
  --seed 7 \
  --adapter_dir /scratch/work/zhangy50/RL/Spot_VLA/openvla/runs/val_loss_4300+7600+dataset+libero_goal_no_noops+b56+lr-0.0005+shf1000+lora-r32+dropout-0.0--image_aug/val_ot_1599 \
  --task_suite_name libero_goal \
  --center_crop True

nvidia-smi
