#!/usr/bin/env python3
"""Evaluation script for Audio-Visual Source Separation."""

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
import numpy as np
from tqdm import tqdm
import pandas as pd

from src.models.av_separation import AVSourceSeparator
from src.data.av_dataset import create_datasets, create_dataloader
from src.eval.metrics import AVSourceSeparationEvaluator
from src.viz.visualizer import AVVisualizer
from src.utils.device import get_device, set_seed, ensure_dir


class Evaluator:
    """Evaluator for Audio-Visual Source Separation model."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_device(config.get("device", "auto"))
        
        # Set random seed
        set_seed(config.get("seed", 42), config.get("deterministic", True))
        
        # Create directories
        self.output_dir = Path(config.get("output_dir", "assets/evaluation"))
        ensure_dir(self.output_dir)
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize evaluator and visualizer
        self.evaluator = AVSourceSeparationEvaluator(
            sample_rate=config.get("sample_rate", 16000)
        )
        self.visualizer = AVVisualizer(
            sample_rate=config.get("sample_rate", 16000),
            save_dir=str(self.output_dir / "visualizations")
        )
        
        # Results storage
        self.results = []
        self.metrics_summary = {}
    
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
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def evaluate_dataset(self, data_loader: DataLoader, split_name: str = "test") -> Dict[str, float]:
        """Evaluate model on a dataset."""
        self.model.eval()
        
        all_metrics = []
        sample_results = []
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc=f"Evaluating {split_name}")
            
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                audio = batch["audio"].to(self.device)
                frames = batch["frames"].to(self.device)
                target_sources = batch["target_sources"].to(self.device)
                face_boxes = batch["face_boxes"]
                sample_ids = batch["sample_ids"]
                
                # Forward pass
                outputs = self.model(audio, frames, face_boxes)
                
                # Compute metrics
                targets = {"target_sources": target_sources}
                metrics = self.evaluator.evaluate(outputs, targets, face_boxes)
                
                all_metrics.append(metrics)
                
                # Store sample results
                for i, sample_id in enumerate(sample_ids):
                    sample_result = {
                        "sample_id": sample_id,
                        "split": split_name,
                        "batch_idx": batch_idx,
                        "sample_idx": i
                    }
                    
                    # Add metrics for this sample
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            sample_result[key] = value
                        elif isinstance(value, (list, tuple)) and len(value) > i:
                            sample_result[key] = value[i]
                    
                    sample_results.append(sample_result)
                
                # Update progress bar
                pbar.set_postfix({
                    "si_sdr": f"{metrics.get('si_sdr_mean', 0):.2f}",
                    "sdr": f"{metrics.get('sdr_mean', 0):.2f}"
                })
                
                # Save visualizations for first few samples
                if batch_idx == 0 and self.config.get("save_visualizations", True):
                    self._save_sample_visualizations(
                        batch, outputs, metrics, split_name
                    )
        
        # Compute average metrics
        avg_metrics = {}
        for metric_dict in all_metrics:
            for key, value in metric_dict.items():
                if key not in avg_metrics:
                    avg_metrics[key] = []
                avg_metrics[key].append(value)
        
        for key in avg_metrics:
            avg_metrics[key] = np.mean(avg_metrics[key])
        
        # Store results
        self.results.extend(sample_results)
        self.metrics_summary[split_name] = avg_metrics
        
        return avg_metrics
    
    def _save_sample_visualizations(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        metrics: Dict[str, float],
        split_name: str
    ) -> None:
        """Save visualizations for sample results."""
        # Get first sample from batch
        audio = batch["audio"][0].cpu()
        frames = batch["frames"][0].cpu()
        target_sources = batch["target_sources"][0].cpu()
        separated_sources = outputs["separated_sources"][0].cpu()
        face_boxes = batch["face_boxes"][0]
        
        # Convert frames to numpy
        frames_np = []
        for i in range(frames.shape[0]):
            frame = frames[i].permute(1, 2, 0).numpy()
            # Denormalize
            frame = frame * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            frame = np.clip(frame, 0, 1)
            frames_np.append((frame * 255).astype(np.uint8))
        
        # Create visualizations
        sample_id = batch["sample_ids"][0]
        
        # Audio separation visualization
        self.visualizer.visualize_audio_separation(
            audio,
            separated_sources,
            target_sources,
            save_path=self.output_dir / f"visualizations/{split_name}_{sample_id}_audio.png"
        )
        
        # Face detection visualization
        self.visualizer.visualize_face_detection(
            frames_np,
            face_boxes,
            save_path=self.output_dir / f"visualizations/{split_name}_{sample_id}_faces.png"
        )
        
        # Summary visualization
        self.visualizer.create_summary_visualization(
            audio,
            separated_sources,
            frames_np,
            face_boxes,
            metrics,
            save_path=self.output_dir / f"visualizations/{split_name}_{sample_id}_summary.png"
        )
    
    def create_leaderboard(self) -> pd.DataFrame:
        """Create a leaderboard from evaluation results."""
        if not self.results:
            return pd.DataFrame()
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Group by split and compute statistics
        leaderboard = []
        
        for split in df["split"].unique():
            split_df = df[df["split"] == split]
            
            split_stats = {
                "Split": split,
                "Samples": len(split_df),
                "SI-SDR (dB)": f"{split_df['si_sdr_mean'].mean():.2f} ± {split_df['si_sdr_mean'].std():.2f}",
                "SDR (dB)": f"{split_df['sdr_mean'].mean():.2f} ± {split_df['sdr_mean'].std():.2f}",
                "PESQ": f"{split_df['pesq_mean'].mean():.2f} ± {split_df['pesq_mean'].std():.2f}",
                "STOI": f"{split_df['stoi_mean'].mean():.2f} ± {split_df['stoi_mean'].std():.2f}",
                "Face Detection Acc": f"{split_df['face_detection_accuracy'].mean():.2f} ± {split_df['face_detection_accuracy'].std():.2f}",
                "Speaker Matching Acc": f"{split_df['speaker_matching_accuracy'].mean():.2f} ± {split_df['speaker_matching_accuracy'].std():.2f}"
            }
            
            leaderboard.append(split_stats)
        
        return pd.DataFrame(leaderboard)
    
    def save_results(self) -> None:
        """Save evaluation results."""
        # Save detailed results
        if self.results:
            results_df = pd.DataFrame(self.results)
            results_df.to_csv(
                self.output_dir / "detailed_results.csv",
                index=False
            )
        
        # Save leaderboard
        leaderboard_df = self.create_leaderboard()
        if not leaderboard_df.empty:
            leaderboard_df.to_csv(
                self.output_dir / "leaderboard.csv",
                index=False
            )
            
            # Print leaderboard
            print("\n" + "="*80)
            print("EVALUATION LEADERBOARD")
            print("="*80)
            print(leaderboard_df.to_string(index=False))
            print("="*80)
        
        # Save metrics summary
        if self.metrics_summary:
            with open(self.output_dir / "metrics_summary.yaml", 'w') as f:
                yaml.dump(self.metrics_summary, f, default_flow_style=False)
        
        print(f"\nResults saved to {self.output_dir}")
    
    def run_ablation_study(self, test_loader: DataLoader) -> None:
        """Run ablation study on different model components."""
        print("Running ablation study...")
        
        ablation_results = {}
        
        # Original model
        print("Evaluating original model...")
        ablation_results["original"] = self.evaluate_dataset(test_loader, "ablation_original")
        
        # Ablation: No visual features
        print("Evaluating without visual features...")
        original_forward = self.model.forward
        
        def no_visual_forward(audio, frames, face_boxes=None):
            # Use only audio features
            audio_features = self.model.audio_encoder(audio)
            # Create dummy visual features
            batch_size, num_frames = frames.shape[:2]
            dummy_visual = torch.zeros(batch_size, num_frames, audio_features.shape[1], device=audio.device)
            
            if self.model.fusion_type == "cross_attention":
                fused_features = self.model.fusion(audio_features, dummy_visual)
            else:
                audio_flat = audio_features.mean(dim=-1)
                visual_flat = dummy_visual.mean(dim=1)
                fused_flat = torch.cat([audio_flat, visual_flat], dim=-1)
                fused_features = self.model.fusion(fused_flat).unsqueeze(-1).expand(-1, -1, audio_features.shape[-1])
            
            separated_sources = self.model.decoder(fused_features)
            
            return {
                "separated_sources": separated_sources,
                "audio_features": audio_features,
                "visual_features": dummy_visual,
                "fused_features": fused_features
            }
        
        self.model.forward = no_visual_forward
        ablation_results["no_visual"] = self.evaluate_dataset(test_loader, "ablation_no_visual")
        
        # Restore original forward
        self.model.forward = original_forward
        
        # Save ablation results
        ablation_df = pd.DataFrame(ablation_results).T
        ablation_df.to_csv(self.output_dir / "ablation_study.csv")
        
        print("\nAblation Study Results:")
        print(ablation_df.to_string())
        
        # Create visualization
        self.visualizer.visualize_metrics_comparison(
            ablation_results,
            list(ablation_results.keys()),
            save_path=self.output_dir / "ablation_comparison.png"
        )


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Audio-Visual Source Separation model")
    parser.add_argument("--config", type=str, default="configs/eval.yaml",
                       help="Path to evaluation configuration file")
    parser.add_argument("--model_config", type=str, default="configs/model.yaml",
                       help="Path to model configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/",
                       help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="assets/evaluation/",
                       help="Path to output directory")
    parser.add_argument("--ablation", action="store_true",
                       help="Run ablation study")
    
    args = parser.parse_args()
    
    # Load configurations
    with open(args.config, 'r') as f:
        eval_config = yaml.safe_load(f)
    
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Merge configurations
    config = {**eval_config, **model_config}
    config["data_dir"] = args.data_dir
    config["output_dir"] = args.output_dir
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        data_dir=config["data_dir"],
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        sample_rate=config.get("sample_rate", 16000),
        video_fps=config.get("video_fps", 25),
        max_audio_length=config.get("max_audio_length", 10.0),
        max_frames=config.get("max_frames", 250)
    )
    
    # Create data loaders
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config.get("batch_size", 16),
        shuffle=False,
        num_workers=config.get("num_workers", 4)
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.get("batch_size", 16),
        shuffle=False,
        num_workers=config.get("num_workers", 4)
    )
    
    # Create evaluator
    evaluator = Evaluator(config)
    
    # Load checkpoint
    evaluator.load_checkpoint(args.checkpoint)
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_metrics = evaluator.evaluate_dataset(val_loader, "val")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = evaluator.evaluate_dataset(test_loader, "test")
    
    # Run ablation study if requested
    if args.ablation:
        evaluator.run_ablation_study(test_loader)
    
    # Save results
    evaluator.save_results()
    
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
