#!/usr/bin/env python3
"""Training script for Audio-Visual Source Separation."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import os
import yaml
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import wandb
from tqdm import tqdm
import numpy as np

from src.models.av_separation import AVSourceSeparator
from src.data.av_dataset import create_datasets, create_dataloader
from src.losses.av_losses import MultiTaskLoss
from src.eval.metrics import AVSourceSeparationEvaluator
from src.viz.visualizer import AVVisualizer
from src.utils.device import get_device, set_seed, ensure_dir
from src.utils.audio import compute_si_sdr, compute_sdr


class Trainer:
    """Trainer for Audio-Visual Source Separation model."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_device(config.get("device", "auto"))
        
        # Set random seed
        set_seed(config.get("seed", 42), config.get("deterministic", True))
        
        # Create directories
        self.checkpoint_dir = Path(config["checkpoint_dir"])
        self.log_dir = Path(config["log_dir"])
        ensure_dir(self.checkpoint_dir)
        ensure_dir(self.log_dir)
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = MultiTaskLoss(
            separation_weight=config.get("separation_weight", 1.0),
            matching_weight=config.get("matching_weight", 0.1),
            spectral_weight=config.get("spectral_weight", 0.5)
        )
        
        # Initialize evaluator and visualizer
        self.evaluator = AVSourceSeparationEvaluator(
            sample_rate=config.get("sample_rate", 16000)
        )
        self.visualizer = AVVisualizer(
            sample_rate=config.get("sample_rate", 16000),
            save_dir=str(self.log_dir / "visualizations")
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {}
        self.val_metrics = {}
        
        # Initialize wandb if enabled
        if config.get("use_wandb", False):
            wandb.init(
                project=config.get("wandb_project", "av-source-separation"),
                config=config,
                name=f"run_{config.get('seed', 42)}"
            )
    
    def _create_model(self) -> AVSourceSeparator:
        """Create the model."""
        model_config = self.config.get("model", {})
        
        return AVSourceSeparator(
            visual_backbone=model_config.get("visual_backbone", "resnet50"),
            audio_encoder_type=model_config.get("audio_encoder_type", "conv1d"),
            hidden_dim=model_config.get("hidden_dim", 512),
            num_sources=model_config.get("num_sources", 2),
            fusion_type=model_config.get("fusion_type", "cross_attention")
        )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        optimizer_name = self.config.get("optimizer", "adamw")
        learning_rate = self.config.get("learning_rate", 1e-4)
        weight_decay = self.config.get("weight_decay", 1e-5)
        
        if optimizer_name.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self) -> Any:
        """Create learning rate scheduler."""
        scheduler_name = self.config.get("scheduler", "cosine")
        
        if scheduler_name.lower() == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get("num_epochs", 100)
            )
        elif scheduler_name.lower() == "step":
            return StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = []
        epoch_metrics = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            audio = batch["audio"].to(self.device)
            frames = batch["frames"].to(self.device)
            target_sources = batch["target_sources"].to(self.device)
            face_boxes = batch["face_boxes"]
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(audio, frames, face_boxes)
            
            # Compute loss
            targets = {"target_sources": target_sources}
            losses = self.criterion(outputs, targets)
            
            total_loss = losses["total"]
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if self.config.get("gradient_clip_norm"):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["gradient_clip_norm"]
                )
            
            self.optimizer.step()
            
            # Compute metrics
            metrics = self.evaluator.evaluate(outputs, targets, face_boxes)
            
            epoch_losses.append(total_loss.item())
            epoch_metrics.append(metrics)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "si_sdr": f"{metrics.get('si_sdr_mean', 0):.2f}"
            })
            
            # Log to wandb
            if self.config.get("use_wandb", False) and batch_idx % self.config.get("log_every", 100) == 0:
                wandb.log({
                    "train/batch_loss": total_loss.item(),
                    "train/si_sdr": metrics.get("si_sdr_mean", 0),
                    "epoch": self.current_epoch,
                    "batch": batch_idx
                })
        
        # Average metrics
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {}
        
        for metric_dict in epoch_metrics:
            for key, value in metric_dict.items():
                if key not in avg_metrics:
                    avg_metrics[key] = []
                avg_metrics[key].append(value)
        
        for key in avg_metrics:
            avg_metrics[key] = np.mean(avg_metrics[key])
        
        return {"loss": avg_loss, **avg_metrics}
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        epoch_losses = []
        epoch_metrics = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            
            for batch in pbar:
                # Move data to device
                audio = batch["audio"].to(self.device)
                frames = batch["frames"].to(self.device)
                target_sources = batch["target_sources"].to(self.device)
                face_boxes = batch["face_boxes"]
                
                # Forward pass
                outputs = self.model(audio, frames, face_boxes)
                
                # Compute loss
                targets = {"target_sources": target_sources}
                losses = self.criterion(outputs, targets)
                
                total_loss = losses["total"]
                
                # Compute metrics
                metrics = self.evaluator.evaluate(outputs, targets, face_boxes)
                
                epoch_losses.append(total_loss.item())
                epoch_metrics.append(metrics)
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{total_loss.item():.4f}",
                    "si_sdr": f"{metrics.get('si_sdr_mean', 0):.2f}"
                })
        
        # Average metrics
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {}
        
        for metric_dict in epoch_metrics:
            for key, value in metric_dict.items():
                if key not in avg_metrics:
                    avg_metrics[key] = []
                avg_metrics[key].append(value)
        
        for key in avg_metrics:
            avg_metrics[key] = np.mean(avg_metrics[key])
        
        return {"loss": avg_loss, **avg_metrics}
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "config": self.config
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.train_metrics = checkpoint.get("train_metrics", {})
        self.val_metrics = checkpoint.get("val_metrics", {})
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Main training loop."""
        num_epochs = self.config.get("num_epochs", 100)
        patience = self.config.get("patience", 20)
        min_delta = self.config.get("min_delta", 1e-4)
        
        epochs_without_improvement = 0
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics["loss"])
            
            # Update training metrics
            for key, value in train_metrics.items():
                if key not in self.train_metrics:
                    self.train_metrics[key] = []
                self.train_metrics[key].append(value)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_metrics["loss"])
            
            # Update validation metrics
            for key, value in val_metrics.items():
                if key not in self.val_metrics:
                    self.val_metrics[key] = []
                self.val_metrics[key].append(value)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Check for improvement
            val_loss = val_metrics["loss"]
            is_best = val_loss < self.best_val_loss - min_delta
            
            if is_best:
                self.best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Save checkpoint
            if epoch % self.config.get("save_every", 10) == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # Log to wandb
            if self.config.get("use_wandb", False):
                log_dict = {
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "val/loss": val_loss,
                    "learning_rate": self.optimizer.param_groups[0]["lr"]
                }
                
                # Add metrics
                for key, value in train_metrics.items():
                    if key != "loss":
                        log_dict[f"train/{key}"] = value
                
                for key, value in val_metrics.items():
                    if key != "loss":
                        log_dict[f"val/{key}"] = value
                
                wandb.log(log_dict)
            
            # Print epoch summary
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val SI-SDR: {val_metrics.get('si_sdr_mean', 0):.2f}")
            print(f"Best Val Loss: {self.best_val_loss:.4f}")
            print("-" * 50)
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break
        
        # Save final checkpoint
        self.save_checkpoint()
        
        # Create visualizations
        self.visualizer.visualize_training_curves(
            self.train_losses,
            self.val_losses,
            self.train_metrics,
            self.val_metrics
        )
        
        print("Training completed!")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Audio-Visual Source Separation model")
    parser.add_argument("--config", type=str, default="configs/train.yaml",
                       help="Path to training configuration file")
    parser.add_argument("--model_config", type=str, default="configs/model.yaml",
                       help="Path to model configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--data_dir", type=str, default="data/",
                       help="Path to dataset directory")
    
    args = parser.parse_args()
    
    # Load configurations
    with open(args.config, 'r') as f:
        train_config = yaml.safe_load(f)
    
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Merge configurations
    config = {**train_config, **model_config}
    config["data_dir"] = args.data_dir
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        data_dir=config["data_dir"],
        train_split=config.get("train_split", 0.8),
        val_split=config.get("val_split", 0.1),
        test_split=config.get("test_split", 0.1),
        sample_rate=config.get("sample_rate", 16000),
        video_fps=config.get("video_fps", 25),
        max_audio_length=config.get("max_audio_length", 10.0),
        max_frames=config.get("max_frames", 250)
    )
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=True,
        num_workers=config.get("num_workers", 4)
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=False,
        num_workers=config.get("num_workers", 4)
    )
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
