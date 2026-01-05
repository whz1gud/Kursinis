"""
Evaluation script for trained models.

Loads a checkpoint and runs full evaluation on dev or test set.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import DISC21EvalDataset, load_groundtruth
from src.models.similarity_model import SimilarityModel
from src.evaluation.metrics import evaluate_retrieval, extract_embeddings


def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cuda') -> SimilarityModel:
    """Load a trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    embedding_dim = checkpoint.get('embedding_dim', 128)
    loss_name = checkpoint.get('loss_name', 'unknown')
    epoch = checkpoint.get('epoch', -1)
    best_metric = checkpoint.get('best_metric', 0)
    
    print(f"  Loss function: {loss_name}")
    print(f"  Epoch: {epoch}")
    print(f"  Best metric: {best_metric:.4f}")
    
    # Create model
    model = SimilarityModel(
        backbone_name='dinov2_vits14',
        embedding_dim=embedding_dim,
        freeze_backbone=True,  # Frozen for inference
        head_type='linear',
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def evaluate_checkpoint(
    checkpoint_path: str,
    split: str = 'dev',
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = 'cuda',
    output_dir: str = 'results',
):
    """
    Evaluate a trained checkpoint on dev or test set.
    
    Args:
        checkpoint_path: Path to checkpoint file
        split: 'dev' or 'test'
        batch_size: Batch size for inference
        num_workers: Data loading workers
        device: Device to use
        output_dir: Where to save results
    """
    print("\n" + "=" * 70)
    print(f"EVALUATION - {split.upper()} SET")
    print("=" * 70)
    
    # Load model
    model, checkpoint_info = load_model_from_checkpoint(checkpoint_path, device)
    loss_name = checkpoint_info.get('loss_name', 'unknown')
    
    # Set up paths based on split
    if split == 'dev':
        queries_dir = 'data/queries_dev'
        groundtruth_path = 'data/dev_queries_groundtruth.csv'
    else:  # test
        queries_dir = 'data/queries_test'
        groundtruth_path = 'data/test_queries_groundtruth.csv'
    
    refs_dir = 'data/refs'
    
    # Load data
    print(f"\nLoading {split} data...")
    query_dataset = DISC21EvalDataset(queries_dir)
    ref_dataset = DISC21EvalDataset(refs_dir)
    groundtruth = load_groundtruth(groundtruth_path)
    
    query_loader = DataLoader(
        query_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    ref_loader = DataLoader(
        ref_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    query_embeddings, query_ids = extract_embeddings(model, query_loader, device)
    ref_embeddings, ref_ids = extract_embeddings(model, ref_loader, device)
    
    print(f"Query embeddings shape: {query_embeddings.shape}")
    print(f"Reference embeddings shape: {ref_embeddings.shape}")
    
    # Evaluate
    print("\nComputing metrics...")
    results = evaluate_retrieval(
        query_embeddings, ref_embeddings,
        query_ids, ref_ids,
        groundtruth,
        k_values=[1, 5, 10, 50, 100],
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    results_with_meta = {
        'experiment': f'{loss_name}_finetuned',
        'checkpoint': str(checkpoint_path),
        'split': split,
        'loss_name': loss_name,
        'epoch': checkpoint_info.get('epoch', -1),
        'num_queries': len(query_ids),
        'num_refs': len(ref_ids),
        'num_groundtruth': len(groundtruth),
        'timestamp': datetime.now().isoformat(),
        'metrics': results,
    }
    
    output_path = Path(output_dir) / f"{loss_name}_{split}_results.json"
    with open(output_path, 'w') as f:
        json.dump(results_with_meta, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"RESULTS - {loss_name.upper()} on {split.upper()}")
    print("=" * 70)
    print(f"\n{'Metric':<20} {'Score':>10}")
    print("-" * 30)
    for metric, score in results.items():
        print(f"{metric:<20} {score:>10.4f}")
    print("-" * 30)
    
    return results


def compare_all_models(split: str = 'dev', output_dir: str = 'results'):
    """Compare baseline and all trained models."""
    
    checkpoint_dir = Path('checkpoints')
    results_dir = Path(output_dir)
    
    # Load baseline
    baseline_path = results_dir / 'baseline_results.json'
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
    else:
        baseline = None
    
    # Find all best checkpoints
    models = {
        'Baseline': baseline['metrics'] if baseline else None,
    }
    
    for loss_name in ['contrastive', 'triplet', 'arcface']:
        ckpt_path = checkpoint_dir / f"{loss_name}_best.pt"
        results_path = results_dir / f"{loss_name}_{split}_results.json"
        
        if results_path.exists():
            with open(results_path) as f:
                models[loss_name.capitalize()] = json.load(f)['metrics']
        elif ckpt_path.exists():
            print(f"\nEvaluating {loss_name}...")
            results = evaluate_checkpoint(str(ckpt_path), split=split)
            models[loss_name.capitalize()] = results
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    metrics = ['P@1', 'P@5', 'P@10', 'mAP']
    
    # Header
    header = f"{'Model':<15}"
    for m in metrics:
        header += f" {m:>10}"
    print(header)
    print("-" * 55)
    
    # Rows
    for model_name, model_results in models.items():
        if model_results is None:
            continue
        row = f"{model_name:<15}"
        for m in metrics:
            score = model_results.get(m, 0)
            row += f" {score:>10.4f}"
        print(row)
    
    print("-" * 55)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--split', type=str, default='dev',
                        choices=['dev', 'test'],
                        help='Which split to evaluate on')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--compare', action='store_true',
                        help='Compare all available models')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.compare:
        compare_all_models(split=args.split)
    else:
        evaluate_checkpoint(
            args.checkpoint,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.workers,
            device=device,
        )


if __name__ == "__main__":
    main()


