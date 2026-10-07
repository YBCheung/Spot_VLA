#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --partition=gpu-h200-141g-ellis,gpu-h200-141g-short
#SBATCH --output=sbatch/finetune_4300+7600+.out

echo "Hello $USER! You are on node $HOSTNAME, $SLURM_JOB_ID.  The time is $(date)."

module load mamba
module spider cuda
module load cuda/12.2
module load seff-gpu
source activate openvla-spot

nvidia-smi

pwd

# # check GPU related versions
# nvcc --version  # System CUDA compiler version
# echo "Pytorch ≥1.12, CUDA toolkit ≥11.0 for stable bfloat16 support"
# python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_bf16_supported())" 
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_new_metrics.py --validation_flag False

echo "Finish time is $(date)."
nvidia-smi
seff $SLURM_JOB_ID

