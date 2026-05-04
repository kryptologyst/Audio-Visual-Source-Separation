"""Audio processing utilities for source separation."""

import torch
import torchaudio
import librosa
import numpy as np
from typing import Tuple, Optional, Union
from scipy.signal import stft, istft


def load_audio(
    path: str, 
    sample_rate: int = 16000, 
    mono: bool = True
) -> Tuple[torch.Tensor, int]:
    """Load audio file and convert to tensor.
    
    Args:
        path: Path to audio file.
        sample_rate: Target sample rate.
        mono: Whether to convert to mono.
        
    Returns:
        Tuple of (audio_tensor, sample_rate).
    """
    waveform, sr = torchaudio.load(path)
    
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    return waveform.squeeze(0), sample_rate


def compute_stft(
    waveform: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: Optional[int] = None
) -> torch.Tensor:
    """Compute Short-Time Fourier Transform.
    
    Args:
        waveform: Input audio waveform.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        win_length: Window length for STFT.
        
    Returns:
        Complex STFT tensor.
    """
    if win_length is None:
        win_length = n_fft
    
    stft_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=None,  # Return complex values
        normalized=False
    )
    
    return stft_transform(waveform)


def compute_istft(
    stft: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: Optional[int] = None
) -> torch.Tensor:
    """Compute Inverse Short-Time Fourier Transform.
    
    Args:
        stft: Complex STFT tensor.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        win_length: Window length for STFT.
        
    Returns:
        Reconstructed waveform.
    """
    if win_length is None:
        win_length = n_fft
    
    istft_transform = torchaudio.transforms.InverseSpectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )
    
    return istft_transform(stft)


def compute_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: Optional[float] = None
) -> torch.Tensor:
    """Compute mel spectrogram.
    
    Args:
        waveform: Input audio waveform.
        sample_rate: Sample rate of audio.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        n_mels: Number of mel bins.
        f_min: Minimum frequency.
        f_max: Maximum frequency.
        
    Returns:
        Mel spectrogram tensor.
    """
    if f_max is None:
        f_max = sample_rate // 2
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max
    )
    
    return mel_transform(waveform)


def compute_mfcc(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80
) -> torch.Tensor:
    """Compute MFCC features.
    
    Args:
        waveform: Input audio waveform.
        sample_rate: Sample rate of audio.
        n_mfcc: Number of MFCC coefficients.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        n_mels: Number of mel bins.
        
    Returns:
        MFCC tensor.
    """
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": n_mels
        }
    )
    
    return mfcc_transform(waveform)


def normalize_audio(waveform: torch.Tensor, method: str = "peak") -> torch.Tensor:
    """Normalize audio waveform.
    
    Args:
        waveform: Input audio waveform.
        method: Normalization method ('peak', 'rms', 'l2').
        
    Returns:
        Normalized waveform.
    """
    if method == "peak":
        return waveform / (torch.max(torch.abs(waveform)) + 1e-8)
    elif method == "rms":
        rms = torch.sqrt(torch.mean(waveform ** 2))
        return waveform / (rms + 1e-8)
    elif method == "l2":
        return waveform / (torch.norm(waveform) + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def add_noise(
    waveform: torch.Tensor, 
    noise_level: float = 0.1,
    noise_type: str = "gaussian"
) -> torch.Tensor:
    """Add noise to audio waveform.
    
    Args:
        waveform: Input audio waveform.
        noise_level: Noise level (0-1).
        noise_type: Type of noise ('gaussian', 'uniform').
        
    Returns:
        Noisy waveform.
    """
    if noise_type == "gaussian":
        noise = torch.randn_like(waveform) * noise_level
    elif noise_type == "uniform":
        noise = (torch.rand_like(waveform) - 0.5) * 2 * noise_level
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return waveform + noise


def compute_si_sdr(
    estimated: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """Compute Scale-Invariant Signal-to-Distortion Ratio.
    
    Args:
        estimated: Estimated signal.
        target: Target signal.
        eps: Small constant for numerical stability.
        
    Returns:
        SI-SDR value in dB.
    """
    # Remove DC component
    estimated = estimated - torch.mean(estimated, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)
    
    # Compute optimal scaling factor
    alpha = torch.sum(estimated * target, dim=-1, keepdim=True) / (
        torch.sum(target * target, dim=-1, keepdim=True) + eps
    )
    
    # Scale target signal
    target_scaled = alpha * target
    
    # Compute SI-SDR
    si_sdr = 10 * torch.log10(
        torch.sum(target_scaled ** 2, dim=-1) / (
            torch.sum((estimated - target_scaled) ** 2, dim=-1) + eps
        )
    )
    
    return si_sdr


def compute_sdr(
    estimated: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """Compute Signal-to-Distortion Ratio.
    
    Args:
        estimated: Estimated signal.
        target: Target signal.
        eps: Small constant for numerical stability.
        
    Returns:
        SDR value in dB.
    """
    sdr = 10 * torch.log10(
        torch.sum(target ** 2, dim=-1) / (
            torch.sum((estimated - target) ** 2, dim=-1) + eps
        )
    )
    
    return sdr
