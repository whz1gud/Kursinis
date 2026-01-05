#!/bin/bash
#SBATCH -p gpu
#SBATCH -n4
#SBATCH --gres gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/lustre/home/%u/Kursinis/logs/slurm_baseline_%j.out
#SBATCH --error=/scratch/lustre/home/%u/Kursinis/logs/slurm_baseline_%j.err
#SBATCH --job-name=baseline_eval

# Change to project directory
cd /scratch/lustre/home/$USER/Kursinis

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate kursinis

echo "Running Baseline Evaluation..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
nvidia-smi

python3 src/run_baseline.py --batch-size 128 --workers 4

echo "Baseline evaluation complete!"


