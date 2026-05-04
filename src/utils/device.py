"""Utility functions for device management and reproducibility."""

import os
import random
from typing import Optional, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the best available device for computation.
    
    Args:
        device: Device preference. If None, auto-detect best available device.
        
    Returns:
        torch.device: The selected device.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    return torch.device(device)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
        deterministic: Whether to use deterministic algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # Enable deterministic algorithms
        torch.use_deterministic_algorithms(True)
        cudnn.deterministic = True
        cudnn.benchmark = False
        
        # Set environment variables for reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        # Enable optimizations for better performance
        cudnn.benchmark = True


def get_model_size(model: torch.nn.Module) -> dict:
    """Calculate model size and parameter count.
    
    Args:
        model: PyTorch model.
        
    Returns:
        dict: Model size information.
    """
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    return {
        "total_params": param_count,
        "trainable_params": trainable_params,
        "model_size_mb": model_size_mb,
    }


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds.
        
    Returns:
        str: Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours}h {minutes}m {seconds:.2f}s"


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path.
    """
    os.makedirs(path, exist_ok=True)
