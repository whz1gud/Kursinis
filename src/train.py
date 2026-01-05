"""
Training script for image similarity with different loss functions.

Supports:
- Contrastive Loss
- Triplet Loss  
- ArcFace Loss

Features:
- Checkpoint saving every epoch and every N minutes
- Resume from checkpoint
- Validation during training
- Logging to file
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# pytorch-metric-learning for loss functions
from pytorch_metric_learning import losses, miners

from src.data.dataset import DISC21TrainDataset, DISC21EvalDataset, load_groundtruth
from src.models.similarity_model import SimilarityModel, ArcFaceHead
from src.evaluation.metrics import evaluate_retrieval, extract_embeddings


class Trainer:
    """
    Trainer class for image similarity models.
    
    Handles training loop, checkpointing, logging, and evaluation.
    """
    
    def __init__(
        self,
        loss_name: str = 'contrastive',
        embedding_dim: int = 128,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs',
        checkpoint_every_minutes: int = 10,
        eval_every_epochs: int = 2,
        device: str = 'cuda',
        num_workers: int = 4,
    ):
        self.loss_name = loss_name
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_every_minutes = checkpoint_every_minutes
        self.eval_every_epochs = eval_every_epochs
        self.device = device
        self.num_workers = num_workers
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.last_checkpoint_time = time.time()
        self.training_history = []
        
        # Initialize model, loss, optimizer
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_data()
        
    def _setup_model(self):
        """Initialize the similarity model."""
        print(f"\n[Setup] Loading DINOv2 model...")
        self.model = SimilarityModel(
            backbone_name='dinov2_vits14',
            embedding_dim=self.embedding_dim,
            freeze_backbone=True,  # FREEZE backbone - only train projection head
            head_type='linear',
        ).to(self.device)
        
        # For ArcFace, we need an additional classification head
        # Initialize to None for all losses, will be set up in _setup_data for arcface
        self.arcface_head = None
        
        print(f"[Setup] Model ready on {self.device}")
        
    def _setup_loss(self):
        """Initialize the loss function."""
        print(f"[Setup] Initializing {self.loss_name} loss...")
        
        if self.loss_name == 'contrastive':
            # Contrastive loss with online pair mining
            self.loss_fn = losses.ContrastiveLoss(pos_margin=0.0, neg_margin=1.0)
            self.miner = miners.PairMarginMiner(pos_margin=0.0, neg_margin=1.0)
            
        elif self.loss_name == 'triplet':
            # Triplet margin loss with hard negative mining
            self.loss_fn = losses.TripletMarginLoss(margin=0.5)
            self.miner = miners.TripletMarginMiner(margin=0.5, type_of_triplets="hard")
            
        elif self.loss_name == 'arcface':
            # ArcFace loss (will be fully initialized after data setup)
            self.loss_fn = None  # Set after knowing num_classes
            self.miner = None  # ArcFace doesn't use mining
            
        else:
            raise ValueError(f"Unknown loss: {self.loss_name}")
            
        print(f"[Setup] Loss function ready: {self.loss_name}")
        
    def _setup_optimizer(self):
        """Initialize the optimizer."""
        # Different learning rates for backbone and head
        backbone_params = list(self.model.backbone.parameters())
        head_params = list(self.model.head.parameters())
        
        param_groups = [
            {'params': backbone_params, 'lr': self.learning_rate * 0.1},  # Lower LR for backbone
            {'params': head_params, 'lr': self.learning_rate},
        ]
        
        self.optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs, eta_min=1e-6
        )
        
        print(f"[Setup] Optimizer ready: AdamW with lr={self.learning_rate}")
        
    def _setup_data(self):
        """Initialize data loaders."""
        print(f"[Setup] Loading datasets...")
        
        # Training data
        self.train_dataset = DISC21TrainDataset('data/train')
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        # Validation data (dev queries + refs)
        self.val_query_dataset = DISC21EvalDataset('data/queries_dev')
        self.val_ref_dataset = DISC21EvalDataset('data/refs')
        self.val_groundtruth = load_groundtruth('data/dev_queries_groundtruth.csv')
        
        self.val_query_loader = DataLoader(
            self.val_query_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.val_ref_loader = DataLoader(
            self.val_ref_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        # For ArcFace, initialize classification head with num_classes
        if self.loss_name == 'arcface':
            num_classes = len(self.train_dataset)
            self.arcface_head = ArcFaceHead(
                embedding_dim=self.embedding_dim,
                num_classes=num_classes,
                scale=30.0,
                margin=0.5,
            ).to(self.device)
            
            # Add arcface head params to optimizer
            self.optimizer.add_param_group({
                'params': self.arcface_head.parameters(),
                'lr': self.learning_rate,
            })
            
            # Use CrossEntropyLoss with ArcFace head
            self.loss_fn = nn.CrossEntropyLoss()
            
            print(f"[Setup] ArcFace head ready with {num_classes} classes")
        
        print(f"[Setup] Training samples: {len(self.train_dataset)}")
        print(f"[Setup] Validation queries: {len(self.val_query_dataset)}")
        print(f"[Setup] Validation refs: {len(self.val_ref_dataset)}")
        
    def save_checkpoint(self, epoch: int, is_best: bool = False, reason: str = "epoch"):
        """Save a checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'loss_name': self.loss_name,
            'embedding_dim': self.embedding_dim,
            'training_history': self.training_history,
        }
        
        # Add ArcFace head if applicable
        if self.arcface_head is not None:
            checkpoint['arcface_head_state_dict'] = self.arcface_head.state_dict()
        
        # Save with timestamp and reason
        filename = f"{self.loss_name}_epoch_{epoch:03d}_{reason}.pt"
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"[Checkpoint] Saved: {filename}")
        
        # Save as best if applicable
        if is_best:
            best_path = self.checkpoint_dir / f"{self.loss_name}_best.pt"
            torch.save(checkpoint, best_path)
            print(f"[Checkpoint] New best model saved!")
        
        # Keep only last 3 regular checkpoints (not best)
        self._cleanup_old_checkpoints()
        
        self.last_checkpoint_time = time.time()
        
    def _cleanup_old_checkpoints(self):
        """Keep only the last 3 checkpoints (excluding best)."""
        pattern = f"{self.loss_name}_epoch_*.pt"
        checkpoints = sorted(self.checkpoint_dir.glob(pattern))
        
        # Don't delete best checkpoint
        checkpoints = [c for c in checkpoints if 'best' not in c.name]
        
        # Keep last 3
        if len(checkpoints) > 3:
            for old_ckpt in checkpoints[:-3]:
                old_ckpt.unlink()
                print(f"[Checkpoint] Removed old: {old_ckpt.name}")
                
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint and resume training."""
        print(f"\n[Resume] Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        self.training_history = checkpoint.get('training_history', [])
        
        if self.arcface_head is not None and 'arcface_head_state_dict' in checkpoint:
            self.arcface_head.load_state_dict(checkpoint['arcface_head_state_dict'])
        
        print(f"[Resume] Resuming from epoch {self.start_epoch}")
        print(f"[Resume] Best metric so far: {self.best_metric:.4f}")
        
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        if self.arcface_head is not None:
            self.arcface_head.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            embeddings = self.model(images)
            
            # Compute loss based on loss type
            if self.loss_name == 'arcface':
                logits = self.arcface_head(embeddings, labels)
                loss = self.loss_fn(logits, labels)
            else:
                # Use miner for contrastive/triplet
                hard_pairs = self.miner(embeddings, labels)
                loss = self.loss_fn(embeddings, labels, hard_pairs)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update stats
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}",
            })
            
            # Time-based checkpoint
            elapsed = time.time() - self.last_checkpoint_time
            if elapsed > self.checkpoint_every_minutes * 60:
                self.save_checkpoint(epoch, reason="timed")
        
        # Update learning rate
        self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        current_lr = self.optimizer.param_groups[0]['lr']
        
        return {
            'epoch': epoch,
            'avg_loss': avg_loss,
            'learning_rate': current_lr,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and return metrics."""
        print("\n[Validation] Extracting embeddings...")
        
        self.model.eval()
        
        # Extract query embeddings
        query_embeddings, query_ids = extract_embeddings(
            self.model, self.val_query_loader, self.device
        )
        
        # Extract reference embeddings  
        ref_embeddings, ref_ids = extract_embeddings(
            self.model, self.val_ref_loader, self.device
        )
        
        # Evaluate
        print("[Validation] Computing metrics...")
        results = evaluate_retrieval(
            query_embeddings, ref_embeddings,
            query_ids, ref_ids,
            self.val_groundtruth,
            k_values=[1, 5, 10],
        )
        
        return results
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        
        # Resume if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        print("\n" + "=" * 70)
        print(f"TRAINING: {self.loss_name.upper()} LOSS")
        print("=" * 70)
        print(f"Epochs: {self.start_epoch} -> {self.num_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Device: {self.device}")
        print(f"Checkpoint every: {self.checkpoint_every_minutes} minutes")
        print("=" * 70 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.num_epochs):
            epoch_start = time.time()
            
            # Train one epoch
            train_stats = self.train_one_epoch(epoch)
            
            epoch_time = time.time() - epoch_start
            print(f"\n[Epoch {epoch}] Loss: {train_stats['avg_loss']:.4f} | "
                  f"LR: {train_stats['learning_rate']:.2e} | "
                  f"Time: {epoch_time/60:.1f}min")
            
            # Save checkpoint after each epoch
            self.save_checkpoint(epoch, reason="epoch")
            
            # Validate periodically
            if (epoch + 1) % self.eval_every_epochs == 0 or epoch == self.num_epochs - 1:
                val_results = self.validate()
                
                # Check if this is the best model
                current_metric = val_results['P@1']
                is_best = current_metric > self.best_metric
                
                if is_best:
                    self.best_metric = current_metric
                    self.save_checkpoint(epoch, is_best=True, reason="best")
                
                # Log results
                train_stats['val_results'] = val_results
                print(f"[Epoch {epoch}] Validation P@1: {current_metric:.4f} "
                      f"(Best: {self.best_metric:.4f})")
            
            # Save training history
            self.training_history.append(train_stats)
            
        # Final summary
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best P@1: {self.best_metric:.4f}")
        print(f"Best checkpoint: {self.checkpoint_dir}/{self.loss_name}_best.pt")
        print("=" * 70)
        
        # Save training history
        history_path = self.log_dir / f"{self.loss_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Training history saved to: {history_path}")
        
        return self.best_metric


def main():
    parser = argparse.ArgumentParser(description="Train image similarity model")
    
    # Loss function
    parser.add_argument('--loss', type=str, default='contrastive',
                        choices=['contrastive', 'triplet', 'arcface'],
                        help='Loss function to use')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    # Checkpointing
    parser.add_argument('--checkpoint-minutes', type=int, default=10,
                        help='Save checkpoint every N minutes')
    parser.add_argument('--eval-epochs', type=int, default=2,
                        help='Validate every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Hardware
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: Running on CPU - this will be slow!")
    
    # Create trainer
    trainer = Trainer(
        loss_name=args.loss,
        embedding_dim=128,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        checkpoint_every_minutes=args.checkpoint_minutes,
        eval_every_epochs=args.eval_epochs,
        device=device,
        num_workers=args.workers,
    )
    
    # Train
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()

