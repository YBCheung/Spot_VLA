#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --partition=gpu-h200-71g-ia
#SBATCH --output=test_env_%j.out
#SBATCH --error=test_env_%j.err

echo "=== Starting environment test ==="
echo "Node: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"


module load mamba
module spider cuda
module load cuda/12.6.2
source activate openvla-spot

nvidia-smi

# Environment setup
export TORCH_CUDA_ARCH_LIST="9.0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=osmesa
export CUDA_LAUNCH_BLOCKING=1

echo "Environment variables set:"
echo "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "  MUJOCO_GL: $MUJOCO_GL"

# Show GPU info
nvidia-smi

# Run the test
cd /scratch/work/zhangy50/RL/Spot_VLA/openvla || exit

echo ""
echo "=== Running environment test ==="
python scripts/test_env.py

echo ""
echo "=== Test completed ==="