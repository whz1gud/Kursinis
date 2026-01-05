# HPC Training Guide

## Understanding the File Systems

**IMPORTANT:** uosis and HPC have DIFFERENT home directories!

| Location | Access | Speed | Use For |
|----------|--------|-------|---------|
| uosis home (WinSCP) | uosis only | - | Upload point |
| HPC home `~/` | HPC only | Slow | Small files |
| `/scratch/lustre/home/$USER/` | HPC only | Fast | Training data & jobs |

## First Time Setup

### Step 1: Upload via WinSCP (to uosis.mif.vu.lt)

Upload your `Kursinis` folder to your home directory in WinSCP.

### Step 2: SSH to HPC and pull files from uosis

```bash
# Connect to HPC (from uosis)
ssh hpc

# Pull files from uosis (linux1) directly to Lustre
mkdir -p /scratch/lustre/home/$USER/Kursinis
scp -r linux1:~/Kursinis/* /scratch/lustre/home/$USER/Kursinis/

# Go there
cd /scratch/lustre/home/$USER/Kursinis
```

### Step 3: Set up Conda environment

```bash
# Make scripts executable
chmod +x hpc/*.sh

# Set up conda environment (uses your existing miniconda3)
bash hpc/setup_env.sh
```

## Running Training Jobs

### Submit jobs:
```bash
cd /scratch/lustre/home/$USER/Kursinis
sbatch hpc/train_triplet.sh
sbatch hpc/train_arcface.sh
sbatch hpc/train_contrastive.sh
```

### Monitor jobs:
```bash
squeue -u $USER          # See your queued/running jobs
squeue -p gpu            # See all GPU queue jobs
scancel <job_id>         # Cancel a job
```

### Check job output:
```bash
tail -f /scratch/lustre/home/$USER/Kursinis/logs/slurm_*.out
```

## After Training - Copy Results Back to uosis

```bash
# On HPC: Copy results back to uosis (accessible via WinSCP)
scp -r /scratch/lustre/home/$USER/Kursinis/checkpoints linux1:~/Kursinis/
scp -r /scratch/lustre/home/$USER/Kursinis/logs linux1:~/Kursinis/
scp -r /scratch/lustre/home/$USER/Kursinis/results linux1:~/Kursinis/
```

Then download from WinSCP (uosis home directory).

## Useful Commands

```bash
nvidia-smi                    # Check GPU (only on compute nodes)
df -h /scratch/lustre         # Check Lustre disk space
du -sh data/                  # Check folder size
```

