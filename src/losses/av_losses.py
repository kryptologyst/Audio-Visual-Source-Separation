"""Loss functions for audio-visual source separation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class SISDRLoss(nn.Module):
    """Scale-Invariant Signal-to-Distortion Ratio loss."""
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        estimated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute SI-SDR loss.
        
        Args:
            estimated: Estimated signal [B, num_sources, T].
            target: Target signal [B, num_sources, T].
            
        Returns:
            SI-SDR loss (negative SI-SDR).
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
            torch.sum(target_flat * target_flat, dim=-1, keepdim=True) + self.eps
        )
        
        # Scale target signal
        target_scaled = alpha * target_flat
        
        # Compute SI-SDR
        si_sdr = 10 * torch.log10(
            torch.sum(target_scaled ** 2, dim=-1) / (
                torch.sum((estimated_flat - target_scaled) ** 2, dim=-1) + self.eps
            )
        )
        
        # Return negative SI-SDR as loss
        return -torch.mean(si_sdr)


class SpectralLoss(nn.Module):
    """Spectral loss for audio quality."""
    
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
    
    def forward(
        self,
        estimated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute spectral loss.
        
        Args:
            estimated: Estimated signal [B, num_sources, T].
            target: Target signal [B, num_sources, T].
            
        Returns:
            Spectral loss.
        """
        batch_size, num_sources, seq_len = estimated.shape
        
        # Flatten for processing
        estimated_flat = estimated.view(-1, seq_len)
        target_flat = target.view(-1, seq_len)
        
        # Compute STFT
        estimated_stft = torch.stft(
            estimated_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True
        )
        
        target_stft = torch.stft(
            target_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True
        )
        
        # Compute magnitude
        estimated_mag = torch.abs(estimated_stft)
        target_mag = torch.abs(target_stft)
        
        # L1 loss on magnitude
        magnitude_loss = F.l1_loss(estimated_mag, target_mag)
        
        # Phase loss
        estimated_phase = torch.angle(estimated_stft)
        target_phase = torch.angle(target_stft)
        phase_loss = F.mse_loss(estimated_phase, target_phase)
        
        return magnitude_loss + 0.1 * phase_loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for face-speaker matching."""
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self,
        face_features: torch.Tensor,
        audio_features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss.
        
        Args:
            face_features: Face features [B, num_faces, C].
            audio_features: Audio features [B, C].
            labels: Speaker labels [B, num_faces].
            
        Returns:
            Contrastive loss.
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
        
        # Apply temperature
        similarities = similarities / self.temperature
        
        # Compute contrastive loss
        loss = 0.0
        for b in range(batch_size):
            for f in range(num_faces):
                if labels[b, f] == 1:  # Positive pair
                    loss += -torch.log(torch.sigmoid(similarities[b, f]))
                else:  # Negative pair
                    loss += -torch.log(torch.sigmoid(-similarities[b, f]))
        
        return loss / (batch_size * num_faces)


class MultiTaskLoss(nn.Module):
    """Multi-task loss combining separation and matching objectives."""
    
    def __init__(
        self,
        separation_weight: float = 1.0,
        matching_weight: float = 0.1,
        spectral_weight: float = 0.5
    ):
        super().__init__()
        
        self.separation_loss = SISDRLoss()
        self.spectral_loss = SpectralLoss()
        self.matching_loss = ContrastiveLoss()
        
        self.separation_weight = separation_weight
        self.matching_weight = matching_weight
        self.spectral_weight = spectral_weight
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss.
        
        Args:
            outputs: Model outputs.
            targets: Target values.
            
        Returns:
            Dictionary of losses.
        """
        losses = {}
        
        # Separation loss
        if "separated_sources" in outputs and "target_sources" in targets:
            separated = outputs["separated_sources"]
            target_sources = targets["target_sources"]
            
            losses["separation"] = self.separation_loss(separated, target_sources)
            losses["spectral"] = self.spectral_loss(separated, target_sources)
        
        # Matching loss
        if "face_features" in outputs and "audio_features" in outputs:
            face_features = outputs["face_features"]
            audio_features = outputs["audio_features"]
            
            # Create dummy labels for now (in practice, these would come from ground truth)
            batch_size, num_frames, num_faces, feature_dim = face_features.shape
            labels = torch.ones(batch_size, num_faces, device=face_features.device)
            
            # Average audio features across time
            audio_avg = audio_features.mean(dim=-1)  # [B, C]
            
            # Average face features across time
            face_avg = face_features.mean(dim=1)  # [B, num_faces, C]
            
            losses["matching"] = self.matching_loss(face_avg, audio_avg, labels)
        
        # Total loss
        total_loss = (
            self.separation_weight * losses.get("separation", 0) +
            self.spectral_weight * losses.get("spectral", 0) +
            self.matching_weight * losses.get("matching", 0)
        )
        
        losses["total"] = total_loss
        
        return losses


class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained audio features."""
    
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Simple feature extractor (in practice, use pre-trained model)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, 15, 4),
            nn.ReLU(),
            nn.Conv1d(64, 128, 15, 4),
            nn.ReLU(),
            nn.Conv1d(128, 256, 15, 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(feature_dim // 256)
        )
    
    def forward(
        self,
        estimated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute perceptual loss.
        
        Args:
            estimated: Estimated signal [B, num_sources, T].
            target: Target signal [B, num_sources, T].
            
        Returns:
            Perceptual loss.
        """
        batch_size, num_sources, seq_len = estimated.shape
        
        # Flatten for processing
        estimated_flat = estimated.view(-1, seq_len).unsqueeze(1)
        target_flat = target.view(-1, seq_len).unsqueeze(1)
        
        # Extract features
        estimated_features = self.feature_extractor(estimated_flat)
        target_features = self.feature_extractor(target_flat)
        
        # Compute L2 loss
        return F.mse_loss(estimated_features, target_features)


class ConsistencyLoss(nn.Module):
    """Consistency loss for temporal coherence."""
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute temporal consistency loss.
        
        Args:
            features: Temporal features [B, T, C].
            
        Returns:
            Consistency loss.
        """
        if features.shape[1] < 2:
            return torch.tensor(0.0, device=features.device)
        
        # Compute temporal differences
        temporal_diff = torch.diff(features, dim=1)
        
        # Compute consistency loss (minimize temporal changes)
        consistency_loss = torch.mean(temporal_diff ** 2)
        
        return self.weight * consistency_loss


def compute_permutation_invariant_loss(
    estimated: torch.Tensor,
    target: torch.Tensor,
    loss_fn: nn.Module
) -> torch.Tensor:
    """Compute permutation invariant loss for source separation.
    
    Args:
        estimated: Estimated sources [B, num_sources, T].
        target: Target sources [B, num_sources, T].
        loss_fn: Loss function to apply.
        
    Returns:
        Permutation invariant loss.
    """
    batch_size, num_sources = estimated.shape[:2]
    
    # Generate all permutations
    import itertools
    permutations = list(itertools.permutations(range(num_sources)))
    
    min_loss = float('inf')
    
    for perm in permutations:
        # Permute estimated sources
        estimated_perm = estimated[:, perm, :]
        
        # Compute loss
        loss = loss_fn(estimated_perm, target)
        
        if loss < min_loss:
            min_loss = loss
    
    return min_loss
