#!/bin/bash
#SBATCH -p gpu
#SBATCH -n4
#SBATCH --gres gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/lustre/home/%u/Kursinis/logs/slurm_arcface_%j.out
#SBATCH --error=/scratch/lustre/home/%u/Kursinis/logs/slurm_arcface_%j.err
#SBATCH --job-name=arcface_train

# Change to project directory on Lustre (fast storage)
cd /scratch/lustre/home/$USER/Kursinis

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate kursinis

# Run training
echo "Starting ArcFace Loss training..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working dir: $(pwd)"
nvidia-smi

# Force unbuffered Python output so we can see progress in real-time
export PYTHONUNBUFFERED=1

# OPTIMIZED for V100 32GB:
# - Batch size 1024 (4x more, better GPU utilization)
# - Learning rate scaled: 0.0001 * 4 = 0.0004 (linear scaling rule)
# - Larger batches = more classes per batch = better ArcFace training
python3 -u src/train.py --loss arcface --epochs 20 --batch-size 1024 --lr 0.0004 --checkpoint-minutes 30 --eval-epochs 5

echo "Training complete!"

