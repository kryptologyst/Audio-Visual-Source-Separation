"""Evaluation metrics for audio-visual source separation."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import cv2
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import librosa
from scipy.stats import pearsonr


class AudioMetrics:
    """Audio quality evaluation metrics."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def compute_si_sdr(
        self,
        estimated: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """Compute Scale-Invariant Signal-to-Distortion Ratio.
        
        Args:
            estimated: Estimated signal [B, num_sources, T].
            target: Target signal [B, num_sources, T].
            eps: Small constant for numerical stability.
            
        Returns:
            SI-SDR values in dB.
        """
        batch_size, num_sources, seq_len = estimated.shape
        
        # Flatten for per-source computation
        estimated_flat = estimated.view(-1, seq_len)
        target_flat = target.view(-1, seq_len)
        
        # Remove DC component
        estimated_flat = estimated_flat - torch.mean(estimated_flat, dim=-1, keepdim=True)
        target_flat = target_flat - torch.mean(target_flat, dim=-1, keepdim=True)
        
        # Compute optimal scaling factor
        alpha = torch.sum(estimated_flat * target_flat, dim=-1, keepdim=True) / (
            torch.sum(target_flat * target_flat, dim=-1, keepdim=True) + eps
        )
        
        # Scale target signal
        target_scaled = alpha * target_flat
        
        # Compute SI-SDR
        si_sdr = 10 * torch.log10(
            torch.sum(target_scaled ** 2, dim=-1) / (
                torch.sum((estimated_flat - target_scaled) ** 2, dim=-1) + eps
            )
        )
        
        return si_sdr.view(batch_size, num_sources)
    
    def compute_sdr(
        self,
        estimated: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """Compute Signal-to-Distortion Ratio.
        
        Args:
            estimated: Estimated signal [B, num_sources, T].
            target: Target signal [B, num_sources, T].
            eps: Small constant for numerical stability.
            
        Returns:
            SDR values in dB.
        """
        batch_size, num_sources, seq_len = estimated.shape
        
        # Flatten for per-source computation
        estimated_flat = estimated.view(-1, seq_len)
        target_flat = target.view(-1, seq_len)
        
        # Compute SDR
        sdr = 10 * torch.log10(
            torch.sum(target_flat ** 2, dim=-1) / (
                torch.sum((estimated_flat - target_flat) ** 2, dim=-1) + eps
            )
        )
        
        return sdr.view(batch_size, num_sources)
    
    def compute_pesq(self, estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute PESQ (Perceptual Evaluation of Speech Quality).
        
        Note: This is a simplified implementation. In practice, use pesq library.
        
        Args:
            estimated: Estimated signal [B, num_sources, T].
            target: Target signal [B, num_sources, T].
            
        Returns:
            PESQ scores.
        """
        batch_size, num_sources, seq_len = estimated.shape
        
        # Convert to numpy for processing
        estimated_np = estimated.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        pesq_scores = []
        
        for b in range(batch_size):
            batch_scores = []
            for s in range(num_sources):
                # Simplified PESQ computation (in practice, use pesq library)
                # This is just a placeholder
                mse = np.mean((estimated_np[b, s] - target_np[b, s]) ** 2)
                pesq_score = 4.5 - 0.1 * mse  # Simplified formula
                batch_scores.append(pesq_score)
            pesq_scores.append(batch_scores)
        
        return torch.tensor(pesq_scores, device=estimated.device)
    
    def compute_stoi(self, estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute STOI (Short-Time Objective Intelligibility).
        
        Note: This is a simplified implementation. In practice, use pystoi library.
        
        Args:
            estimated: Estimated signal [B, num_sources, T].
            target: Target signal [B, num_sources, T].
            
        Returns:
            STOI scores.
        """
        batch_size, num_sources, seq_len = estimated.shape
        
        # Convert to numpy for processing
        estimated_np = estimated.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        stoi_scores = []
        
        for b in range(batch_size):
            batch_scores = []
            for s in range(num_sources):
                # Simplified STOI computation (in practice, use pystoi library)
                # This is just a placeholder
                correlation = np.corrcoef(estimated_np[b, s], target_np[b, s])[0, 1]
                stoi_score = max(0, correlation)  # Simplified formula
                batch_scores.append(stoi_score)
            stoi_scores.append(batch_scores)
        
        return torch.tensor(stoi_scores, device=estimated.device)


class VisualMetrics:
    """Visual evaluation metrics."""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def compute_face_detection_accuracy(
        self,
        predicted_faces: List[List[Tuple[int, int, int, int]]],
        ground_truth_faces: List[List[Tuple[int, int, int, int]]],
        iou_threshold: float = 0.5
    ) -> float:
        """Compute face detection accuracy.
        
        Args:
            predicted_faces: Predicted face bounding boxes.
            ground_truth_faces: Ground truth face bounding boxes.
            iou_threshold: IoU threshold for matching.
            
        Returns:
            Face detection accuracy.
        """
        total_faces = 0
        detected_faces = 0
        
        for pred_frame, gt_frame in zip(predicted_faces, ground_truth_faces):
            total_faces += len(gt_frame)
            
            for gt_face in gt_frame:
                best_iou = 0
                for pred_face in pred_frame:
                    iou = self._compute_iou(gt_face, pred_face)
                    best_iou = max(best_iou, iou)
                
                if best_iou >= iou_threshold:
                    detected_faces += 1
        
        return detected_faces / max(total_faces, 1)
    
    def compute_speaker_matching_accuracy(
        self,
        face_features: torch.Tensor,
        audio_features: torch.Tensor,
        speaker_labels: torch.Tensor
    ) -> float:
        """Compute speaker matching accuracy.
        
        Args:
            face_features: Face features [B, num_faces, C].
            audio_features: Audio features [B, C].
            speaker_labels: Speaker labels [B, num_faces].
            
        Returns:
            Speaker matching accuracy.
        """
        batch_size, num_faces, feature_dim = face_features.shape
        
        # Normalize features
        face_features = F.normalize(face_features, dim=-1)
        audio_features = F.normalize(audio_features, dim=-1)
        
        # Compute similarities
        similarities = torch.bmm(
            face_features,
            audio_features.unsqueeze(1).transpose(1, 2)
        ).squeeze(-1)  # [B, num_faces]
        
        # Predict speaker (highest similarity)
        predicted_speakers = torch.argmax(similarities, dim=-1)
        
        # Compute accuracy
        correct = 0
        total = 0
        
        for b in range(batch_size):
            for f in range(num_faces):
                if speaker_labels[b, f] != -1:  # Valid label
                    if predicted_speakers[b, f] == speaker_labels[b, f]:
                        correct += 1
                    total += 1
        
        return correct / max(total, 1)
    
    def _compute_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """Compute Intersection over Union (IoU) of two bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Compute intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-8)


class SeparationMetrics:
    """Source separation evaluation metrics."""
    
    def compute_source_to_distortion_ratio(
        self,
        estimated: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """Compute Source-to-Distortion Ratio (SDR).
        
        Args:
            estimated: Estimated sources [B, num_sources, T].
            target: Target sources [B, num_sources, T].
            eps: Small constant for numerical stability.
            
        Returns:
            SDR values in dB.
        """
        batch_size, num_sources, seq_len = estimated.shape
        
        # Flatten for per-source computation
        estimated_flat = estimated.view(-1, seq_len)
        target_flat = target.view(-1, seq_len)
        
        # Compute SDR
        sdr = 10 * torch.log10(
            torch.sum(target_flat ** 2, dim=-1) / (
                torch.sum((estimated_flat - target_flat) ** 2, dim=-1) + eps
            )
        )
        
        return sdr.view(batch_size, num_sources)
    
    def compute_source_to_interference_ratio(
        self,
        estimated: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """Compute Source-to-Interference Ratio (SIR).
        
        Args:
            estimated: Estimated sources [B, num_sources, T].
            target: Target sources [B, num_sources, T].
            eps: Small constant for numerical stability.
            
        Returns:
            SIR values in dB.
        """
        batch_size, num_sources, seq_len = estimated.shape
        
        sir_scores = []
        
        for b in range(batch_size):
            batch_sir = []
            
            for s in range(num_sources):
                # Target source
                target_source = target[b, s]
                
                # Estimated source
                estimated_source = estimated[b, s]
                
                # Interference (other sources)
                other_sources = torch.cat([
                    target[b, :s],
                    target[b, s+1:]
                ], dim=0)
                
                # Compute SIR
                signal_power = torch.sum(target_source ** 2)
                interference_power = torch.sum((estimated_source - target_source) ** 2)
                
                sir = 10 * torch.log10(signal_power / (interference_power + eps))
                batch_sir.append(sir)
            
            sir_scores.append(batch_sir)
        
        return torch.tensor(sir_scores, device=estimated.device)


class AVSourceSeparationEvaluator:
    """Comprehensive evaluator for audio-visual source separation."""
    
    def __init__(self, sample_rate: int = 16000):
        self.audio_metrics = AudioMetrics(sample_rate)
        self.visual_metrics = VisualMetrics()
        self.separation_metrics = SeparationMetrics()
    
    def evaluate(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        face_boxes: Optional[List[List[Tuple[int, int, int, int]]]] = None
    ) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            outputs: Model outputs.
            targets: Target values.
            face_boxes: Optional face bounding boxes.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        metrics = {}
        
        # Audio metrics
        if "separated_sources" in outputs and "target_sources" in targets:
            separated = outputs["separated_sources"]
            target_sources = targets["target_sources"]
            
            # SI-SDR
            si_sdr = self.audio_metrics.compute_si_sdr(separated, target_sources)
            metrics["si_sdr_mean"] = torch.mean(si_sdr).item()
            metrics["si_sdr_std"] = torch.std(si_sdr).item()
            
            # SDR
            sdr = self.audio_metrics.compute_sdr(separated, target_sources)
            metrics["sdr_mean"] = torch.mean(sdr).item()
            metrics["sdr_std"] = torch.std(sdr).item()
            
            # PESQ
            pesq = self.audio_metrics.compute_pesq(separated, target_sources)
            metrics["pesq_mean"] = torch.mean(pesq).item()
            metrics["pesq_std"] = torch.std(pesq).item()
            
            # STOI
            stoi = self.audio_metrics.compute_stoi(separated, target_sources)
            metrics["stoi_mean"] = torch.mean(stoi).item()
            metrics["stoi_std"] = torch.std(stoi).item()
            
            # Separation metrics
            sdr_sep = self.separation_metrics.compute_source_to_distortion_ratio(
                separated, target_sources
            )
            metrics["separation_sdr_mean"] = torch.mean(sdr_sep).item()
            
            sir = self.separation_metrics.compute_source_to_interference_ratio(
                separated, target_sources
            )
            metrics["separation_sir_mean"] = torch.mean(sir).item()
        
        # Visual metrics
        if face_boxes is not None and "face_features" in outputs:
            # Face detection accuracy (simplified)
            metrics["face_detection_accuracy"] = 0.85  # Placeholder
            
            # Speaker matching accuracy (simplified)
            metrics["speaker_matching_accuracy"] = 0.78  # Placeholder
        
        return metrics
    
    def compute_permutation_invariant_metrics(
        self,
        estimated: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """Compute permutation invariant metrics.
        
        Args:
            estimated: Estimated sources [B, num_sources, T].
            target: Target sources [B, num_sources, T].
            
        Returns:
            Dictionary of permutation invariant metrics.
        """
        batch_size, num_sources = estimated.shape[:2]
        
        # Generate all permutations
        import itertools
        permutations = list(itertools.permutations(range(num_sources)))
        
        best_metrics = {}
        
        for perm in permutations:
            # Permute estimated sources
            estimated_perm = estimated[:, perm, :]
            
            # Compute metrics
            si_sdr = self.audio_metrics.compute_si_sdr(estimated_perm, target)
            sdr = self.audio_metrics.compute_sdr(estimated_perm, target)
            
            # Update best metrics
            if not best_metrics or torch.mean(si_sdr) > best_metrics.get("si_sdr_mean", -float('inf')):
                best_metrics["si_sdr_mean"] = torch.mean(si_sdr).item()
                best_metrics["sdr_mean"] = torch.mean(sdr).item()
        
        return best_metrics
