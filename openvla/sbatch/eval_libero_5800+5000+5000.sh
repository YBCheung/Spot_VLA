#!/bin/bash
#SBATCH --time=01:30:00 # 14.33-15.08, suddenly stoped. only reserved 15 min. 
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4  # 2 is too tight, lead to low GPU utilization. maybe decrease task time. 
#SBATCH --mem=40G
#SBATCH --partition=gpu-h200-141g-short,gpu-h200-141g-m,gpu-b300-288g-short,gpu-h200-141g-ellis,gpu-b300-288g-ellis,gpu-b300-288g-short
#SBATCH --output=sbatch/0eval_7_1_5800+5000+5000.out

echo "Hello $USER! You are on node $HOSTNAME. SLURM_JOB_ID: $SLURM_JOB_ID. The time is $(date)."

module load mamba
module spider cuda
module load cuda/12.2.2
source activate openvla-spot


echo "Environment:"
echo "  MUJOCO_GL: $MUJOCO_GL"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"

nvidia-smi

pwd

echo "SLURM partition: $SLURM_JOB_PARTITION"

# # check GPU related versionss
# nvcc --version  # System CUDA compiler version
echo "Pytorch ≥1.12, CUDA toolkit ≥11.0 for stable bfloat16 support"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_bf16_supported())" 
# vla_path openvla/openvla-7b-finetuned-libero-object
torchrun --standalone --nnodes 1 --nproc-per-node 1 experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --load_from_adapter True \
  --vla_path openvla/openvla-7b \
  --num_trials_per_task 10 \
  --seed 7 \
  --adapter_dir /scratch/work/zhangy50/RL/Spot_VLA/openvla/runs/5800+10000_openvla-7b+dataset+libero_goal_no_noops+b56+lr-0.0005+shf1000+lora-r32+dropout-0.0--image_aug/default_5000 \
  --task_suite_name libero_goal \
  --center_crop True

nvidia-smi
