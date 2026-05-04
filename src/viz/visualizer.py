"""Visualization tools for audio-visual source separation."""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union
import seaborn as sns
from pathlib import Path

from ..utils.audio import compute_stft, compute_mel_spectrogram
from ..utils.video import detect_faces


class AVVisualizer:
    """Visualizer for audio-visual source separation results."""
    
    def __init__(self, sample_rate: int = 16000, save_dir: str = "assets/visualizations"):
        self.sample_rate = sample_rate
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def visualize_audio_separation(
        self,
        mixed_audio: torch.Tensor,
        separated_sources: torch.Tensor,
        target_sources: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Visualize audio source separation results.
        
        Args:
            mixed_audio: Mixed audio waveform [T].
            separated_sources: Separated sources [num_sources, T].
            target_sources: Target sources [num_sources, T].
            save_path: Path to save the visualization.
        """
        num_sources = separated_sources.shape[0]
        
        # Create figure
        fig, axes = plt.subplots(
            num_sources + 1, 2,
            figsize=(15, 3 * (num_sources + 1)),
            gridspec_kw={'hspace': 0.3}
        )
        
        # Plot mixed audio
        time_axis = torch.linspace(0, len(mixed_audio) / self.sample_rate, len(mixed_audio))
        
        axes[0, 0].plot(time_axis, mixed_audio.numpy())
        axes[0, 0].set_title("Mixed Audio (Waveform)")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].grid(True)
        
        # Plot mixed audio spectrogram
        mixed_stft = compute_stft(mixed_audio)
        mixed_mag = torch.abs(mixed_stft)
        
        axes[0, 1].imshow(
            torch.log(mixed_mag + 1e-8).numpy(),
            aspect='auto',
            origin='lower',
            cmap='viridis'
        )
        axes[0, 1].set_title("Mixed Audio (Spectrogram)")
        axes[0, 1].set_xlabel("Time Frames")
        axes[0, 1].set_ylabel("Frequency Bins")
        
        # Plot separated sources
        for i in range(num_sources):
            source = separated_sources[i]
            
            # Create time axis for this source
            source_time_axis = torch.linspace(0, len(source) / self.sample_rate, len(source))
            
            # Waveform
            axes[i + 1, 0].plot(source_time_axis, source.numpy())
            axes[i + 1, 0].set_title(f"Separated Source {i + 1} (Waveform)")
            axes[i + 1, 0].set_xlabel("Time (s)")
            axes[i + 1, 0].set_ylabel("Amplitude")
            axes[i + 1, 0].grid(True)
            
            # Spectrogram
            source_stft = compute_stft(source)
            source_mag = torch.abs(source_stft)
            
            axes[i + 1, 1].imshow(
                torch.log(source_mag + 1e-8).numpy(),
                aspect='auto',
                origin='lower',
                cmap='viridis'
            )
            axes[i + 1, 1].set_title(f"Separated Source {i + 1} (Spectrogram)")
            axes[i + 1, 1].set_xlabel("Time Frames")
            axes[i + 1, 1].set_ylabel("Frequency Bins")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.save_dir / "audio_separation.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def visualize_face_detection(
        self,
        frames: List[np.ndarray],
        face_boxes: List[List[Tuple[int, int, int, int]]],
        save_path: Optional[str] = None
    ) -> None:
        """Visualize face detection results.
        
        Args:
            frames: List of video frames.
            face_boxes: List of face bounding boxes for each frame.
            save_path: Path to save the visualization.
        """
        num_frames = min(len(frames), 8)  # Show max 8 frames
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(num_frames):
            frame = frames[i]
            faces = face_boxes[i] if i < len(face_boxes) else []
            
            # Convert BGR to RGB if needed
            if frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            axes[i].imshow(frame_rgb)
            axes[i].set_title(f"Frame {i + 1}")
            axes[i].axis('off')
            
            # Draw face bounding boxes
            for j, (x, y, w, h) in enumerate(faces):
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none'
                )
                axes[i].add_patch(rect)
                
                # Add face number
                axes[i].text(x, y - 5, f"Face {j + 1}", 
                           color='red', fontsize=10, fontweight='bold')
        
        # Hide unused subplots
        for i in range(num_frames, 8):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.save_dir / "face_detection.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def visualize_attention_maps(
        self,
        attention_weights: torch.Tensor,
        frames: List[np.ndarray],
        save_path: Optional[str] = None
    ) -> None:
        """Visualize attention maps.
        
        Args:
            attention_weights: Attention weights [num_frames, num_faces].
            frames: List of video frames.
            save_path: Path to save the visualization.
        """
        num_frames = min(len(frames), 6)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(num_frames):
            frame = frames[i]
            
            # Convert BGR to RGB if needed
            if frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Resize frame to match attention map
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            
            # Get attention weights for this frame
            if i < attention_weights.shape[0]:
                attention = attention_weights[i].numpy()
                
                # Create attention map
                attention_map = np.zeros((224, 224))
                
                # Place attention weights (simplified)
                for j, weight in enumerate(attention):
                    if j < 2:  # Assume max 2 faces
                        y_start = 50 + j * 100
                        y_end = y_start + 80
                        x_start = 50 + j * 100
                        x_end = x_start + 80
                        
                        attention_map[y_start:y_end, x_start:x_end] = weight
            
                # Overlay attention map
                axes[i].imshow(frame_resized)
                im = axes[i].imshow(attention_map, alpha=0.5, cmap='hot')
                axes[i].set_title(f"Frame {i + 1} - Attention Map")
            else:
                axes[i].imshow(frame_resized)
                axes[i].set_title(f"Frame {i + 1}")
            
            axes[i].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes, shrink=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.save_dir / "attention_maps.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def visualize_metrics_comparison(
        self,
        metrics: Dict[str, List[float]],
        model_names: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """Visualize metrics comparison across models.
        
        Args:
            metrics: Dictionary of metrics with lists of values.
            model_names: List of model names.
            save_path: Path to save the visualization.
        """
        num_metrics = len(metrics)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        metric_names = list(metrics.keys())
        
        for i, metric_name in enumerate(metric_names[:4]):  # Show max 4 metrics
            metric_values = metrics[metric_name]
            
            # Create bar plot
            bars = axes[i].bar(model_names, metric_values)
            axes[i].set_title(f"{metric_name.replace('_', ' ').title()}")
            axes[i].set_ylabel("Score")
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            # Rotate x-axis labels if needed
            if len(max(model_names, key=len)) > 8:
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.save_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def visualize_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_metrics: Optional[Dict[str, List[float]]] = None,
        val_metrics: Optional[Dict[str, List[float]]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Visualize training curves.
        
        Args:
            train_losses: Training losses.
            val_losses: Validation losses.
            train_metrics: Training metrics.
            val_metrics: Validation metrics.
            save_path: Path to save the visualization.
        """
        num_plots = 1
        if train_metrics:
            num_plots += len(train_metrics)
        
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))
        if num_plots == 1:
            axes = [axes]
        
        epochs = range(1, len(train_losses) + 1)
        
        # Plot losses
        axes[0].plot(epochs, train_losses, label='Training Loss', color='blue')
        axes[0].plot(epochs, val_losses, label='Validation Loss', color='red')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot metrics
        if train_metrics and val_metrics:
            for i, (metric_name, train_values) in enumerate(train_metrics.items()):
                if i + 1 < len(axes):
                    val_values = val_metrics.get(metric_name, [])
                    
                    axes[i + 1].plot(epochs, train_values, label=f'Training {metric_name}', color='blue')
                    if val_values:
                        axes[i + 1].plot(epochs, val_values, label=f'Validation {metric_name}', color='red')
                    
                    axes[i + 1].set_title(f'{metric_name.replace("_", " ").title()}')
                    axes[i + 1].set_xlabel('Epoch')
                    axes[i + 1].set_ylabel('Score')
                    axes[i + 1].legend()
                    axes[i + 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.save_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def create_summary_visualization(
        self,
        mixed_audio: torch.Tensor,
        separated_sources: torch.Tensor,
        frames: List[np.ndarray],
        face_boxes: List[List[Tuple[int, int, int, int]]],
        metrics: Dict[str, float],
        save_path: Optional[str] = None
    ) -> None:
        """Create a comprehensive summary visualization.
        
        Args:
            mixed_audio: Mixed audio waveform.
            separated_sources: Separated sources.
            frames: Video frames.
            face_boxes: Face bounding boxes.
            metrics: Evaluation metrics.
            save_path: Path to save the visualization.
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Audio waveforms
        ax1 = fig.add_subplot(gs[0, :2])
        time_axis = torch.linspace(0, len(mixed_audio) / self.sample_rate, len(mixed_audio))
        ax1.plot(time_axis, mixed_audio.numpy(), label='Mixed Audio', alpha=0.7)
        ax1.set_title('Audio Waveforms')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True)
        
        # Separated sources
        for i in range(separated_sources.shape[0]):
            source = separated_sources[i]
            source_time_axis = torch.linspace(0, len(source) / self.sample_rate, len(source))
            ax1.plot(source_time_axis, source.numpy(), 
                    label=f'Source {i+1}', alpha=0.8)
        ax1.legend()
        
        # Metrics
        ax2 = fig.add_subplot(gs[0, 2:])
        metric_names = list(metrics.keys())[:6]  # Show top 6 metrics
        metric_values = [metrics[name] for name in metric_names]
        
        bars = ax2.bar(range(len(metric_names)), metric_values)
        ax2.set_title('Evaluation Metrics')
        ax2.set_xticks(range(len(metric_names)))
        ax2.set_xticklabels([name.replace('_', ' ').title() for name in metric_names], 
                           rotation=45, ha='right')
        ax2.set_ylabel('Score')
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Video frames with face detection
        for i in range(min(4, len(frames))):
            ax = fig.add_subplot(gs[1, i])
            
            frame = frames[i]
            faces = face_boxes[i] if i < len(face_boxes) else []
            
            # Convert BGR to RGB if needed
            if frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            ax.imshow(frame_rgb)
            ax.set_title(f'Frame {i+1}')
            ax.axis('off')
            
            # Draw face bounding boxes
            for j, (x, y, w, h) in enumerate(faces):
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x, y - 5, f'F{j+1}', color='red', fontsize=8, fontweight='bold')
        
        # Spectrograms
        ax3 = fig.add_subplot(gs[2, :2])
        mixed_stft = compute_stft(mixed_audio)
        mixed_mag = torch.abs(mixed_stft)
        
        im1 = ax3.imshow(
            torch.log(mixed_mag + 1e-8).numpy(),
            aspect='auto',
            origin='lower',
            cmap='viridis'
        )
        ax3.set_title('Mixed Audio Spectrogram')
        ax3.set_xlabel('Time Frames')
        ax3.set_ylabel('Frequency Bins')
        
        # Separated source spectrograms
        ax4 = fig.add_subplot(gs[2, 2:])
        source_stft = compute_stft(separated_sources[0])
        source_mag = torch.abs(source_stft)
        
        im2 = ax4.imshow(
            torch.log(source_mag + 1e-8).numpy(),
            aspect='auto',
            origin='lower',
            cmap='viridis'
        )
        ax4.set_title('Separated Source 1 Spectrogram')
        ax4.set_xlabel('Time Frames')
        ax4.set_ylabel('Frequency Bins')
        
        # Add colorbars
        plt.colorbar(im1, ax=ax3, shrink=0.8)
        plt.colorbar(im2, ax=ax4, shrink=0.8)
        
        plt.suptitle('Audio-Visual Source Separation Results', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.save_dir / "summary_visualization.png", dpi=300, bbox_inches='tight')
        
        plt.close()
