# Image Similarity Detection with DINOv2

Coursework comparing different loss functions (Contrastive, Triplet, ArcFace) for fine-tuning DINOv2 on the DISC21 dataset.

## Overview

This project evaluates three metric learning loss functions for image similarity detection:
- **Contrastive Loss**: Pairwise distance-based learning
- **Triplet Loss**: Anchor-positive-negative triplet learning  
- **ArcFace Loss**: Angular margin-based classification

All methods fine-tune a pre-trained DINOv2 ViT-S/14 model on the DISC21 dataset.

## Results

ArcFace achieved the best performance:
- **P@1**: 58.41% (+1.7% over baseline)
- **μAP**: 50.04% (+8.6% over baseline)

See `bachelor_thesis_template_vu_mif_se/bakalaurinis.pdf` for full results and analysis.

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Data

Download the DISC21 dataset and extract to `data/`:
- `data/train/` - Training images
- `data/queries_dev/` - Development query images
- `data/queries_test/` - Test query images
- `data/refs/` - Reference database images

## Usage

### Training

```bash
# Contrastive Loss
python src/train.py --loss contrastive --epochs 20 --batch-size 256

# Triplet Loss
python src/train.py --loss triplet --epochs 20 --batch-size 256

# ArcFace Loss
python src/train.py --loss arcface --epochs 20 --batch-size 256
```

### Evaluation

```bash
# Evaluate a trained model
python src/evaluate.py --checkpoint checkpoints/arcface_best.pt

# Run baseline (no fine-tuning)
python src/run_baseline.py
```

### HPC Training

For HPC cluster usage, see `hpc/README_HPC.md`.

```bash
# Submit training jobs
sbatch hpc/train_contrastive.sh
sbatch hpc/train_triplet.sh
sbatch hpc/train_arcface.sh
```

## Project Structure

```
.
├── src/                    # Source code
│   ├── train.py           # Main training script
│   ├── evaluate.py        # Evaluation script
│   ├── run_baseline.py    # Baseline evaluation
│   ├── data/              # Dataset classes
│   ├── models/            # Model architectures
│   └── evaluation/        # Evaluation metrics
├── hpc/                    # HPC SLURM scripts
├── checkpoints/            # Model checkpoints (not in git)
├── logs/                   # Training logs (not in git)
└── bachelor_thesis_template_vu_mif_se/  # LaTeX thesis
```

## Citation

If you use this code, please cite:
- DISC21 Dataset: [DTP+21]
- DINOv2: [Oqu24]
- ArcFace: [Den22]

Full citations available in `bachelor_thesis_template_vu_mif_se/bibliografija.bib`.

## License

This project is for academic coursework purposes.

