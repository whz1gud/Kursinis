#!/bin/bash
#SBATCH -p gpu
#SBATCH -n4
#SBATCH --gres gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/lustre/home/%u/Kursinis/logs/slurm_contrastive_%j.out
#SBATCH --error=/scratch/lustre/home/%u/Kursinis/logs/slurm_contrastive_%j.err
#SBATCH --job-name=contrastive_train

# Change to project directory on Lustre (fast storage)
cd /scratch/lustre/home/$USER/Kursinis

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate kursinis

# Run training
echo "Starting Contrastive Loss training..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working dir: $(pwd)"
nvidia-smi

# Force unbuffered Python output so we can see progress in real-time
export PYTHONUNBUFFERED=1
python3 -u src/train.py --loss contrastive --epochs 20 --batch-size 256 --checkpoint-minutes 30 --eval-epochs 5

echo "Training complete!"

