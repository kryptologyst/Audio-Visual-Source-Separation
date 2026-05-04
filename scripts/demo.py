#!/usr/bin/env python3
"""Simple demo script for Audio-Visual Source Separation."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.models.av_separation import AVSourceSeparator
from src.utils.device import get_device, set_seed
from src.utils.audio import normalize_audio
from src.utils.video import detect_faces, preprocess_frame
from src.viz.visualizer import AVVisualizer


def create_dummy_data():
    """Create dummy audio and video data for demonstration."""
    # Create dummy audio (two different frequency components)
    sample_rate = 16000
    duration = 5.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    
    # Speaker 1: 440 Hz (A4 note)
    speaker1 = 0.5 * torch.sin(2 * np.pi * 440 * t)
    
    # Speaker 2: 880 Hz (A5 note)  
    speaker2 = 0.5 * torch.sin(2 * np.pi * 880 * t)
    
    # Mix speakers with different time segments
    mixed = torch.zeros_like(t)
    mid_point = len(t) // 2
    
    mixed[:mid_point] = speaker1[:mid_point]
    mixed[mid_point:] = speaker2[mid_point:]
    
    # Add some noise
    mixed += 0.1 * torch.randn_like(mixed)
    mixed = normalize_audio(mixed)
    
    # Create dummy video frames
    num_frames = int(duration * 25)  # 25 FPS
    frames = []
    face_boxes = []
    
    for i in range(num_frames):
        # Create a simple frame with two colored rectangles (simulating faces)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Face 1 (left side)
        cv2.rectangle(frame, (100, 100), (180, 180), (255, 0, 0), -1)
        
        # Face 2 (right side)
        cv2.rectangle(frame, (400, 150), (480, 230), (0, 255, 0), -1)
        
        frames.append(frame)
        
        # Detect faces (should find the rectangles)
        faces = detect_faces(frame)
        face_boxes.append(faces)
    
    # Convert frames to tensors
    frame_tensors = []
    for frame in frames:
        frame_tensor = preprocess_frame(frame)
        frame_tensors.append(frame_tensor)
    
    frames_tensor = torch.stack(frame_tensors).unsqueeze(0)  # Add batch dimension
    audio_tensor = mixed.unsqueeze(0)  # Add batch dimension
    
    return audio_tensor, frames_tensor, face_boxes, mixed


def main():
    """Main demo function."""
    print("🎵 Audio-Visual Source Separation Demo")
    print("=" * 50)
    
    # Set up
    set_seed(42)
    device = get_device("auto")
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = AVSourceSeparator(
        visual_backbone="resnet18",  # Use smaller model for demo
        audio_encoder_type="conv1d",
        hidden_dim=256,
        num_sources=2,
        fusion_type="cross_attention"
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy data
    print("Creating dummy data...")
    audio, frames, face_boxes, mixed_audio = create_dummy_data()
    
    audio = audio.to(device)
    frames = frames.to(device)
    
    print(f"Audio shape: {audio.shape}")
    print(f"Video frames shape: {frames.shape}")
    print(f"Face boxes: {len(face_boxes)} frames with face detection")
    
    # Run inference
    print("Running inference...")
    model.eval()
    
    with torch.no_grad():
        outputs = model(audio, frames, [face_boxes])
    
    separated_sources = outputs["separated_sources"][0].cpu()  # Remove batch dimension
    
    print(f"Separated sources shape: {separated_sources.shape}")
    
    # Create visualizations
    print("Creating visualizations...")
    visualizer = AVVisualizer(save_dir="assets/demo_output")
    
    # Convert frames back to numpy for visualization
    frames_np = []
    for i in range(frames.shape[1]):
        frame = frames[0, i].permute(1, 2, 0).cpu().numpy()
        # Denormalize
        frame = frame * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        frame = np.clip(frame, 0, 1)
        frames_np.append((frame * 255).astype(np.uint8))
    
    # Create target sources for visualization
    target_sources = torch.stack([
        separated_sources[0],  # Use separated as target for demo
        separated_sources[1]
    ])
    
    # Visualize results
    visualizer.visualize_audio_separation(
        mixed_audio,
        separated_sources,
        target_sources,
        save_path="assets/demo_output/audio_separation_demo.png"
    )
    
    visualizer.visualize_face_detection(
        frames_np,
        face_boxes,
        save_path="assets/demo_output/face_detection_demo.png"
    )
    
    # Create summary visualization
    dummy_metrics = {
        "si_sdr_mean": 15.2,
        "sdr_mean": 18.5,
        "pesq_mean": 3.2,
        "stoi_mean": 0.85
    }
    
    visualizer.create_summary_visualization(
        mixed_audio,
        separated_sources,
        frames_np,
        face_boxes,
        dummy_metrics,
        save_path="assets/demo_output/summary_demo.png"
    )
    
    print("Demo completed!")
    print("Visualizations saved to assets/demo_output/")
    print("\nFiles created:")
    print("- audio_separation_demo.png")
    print("- face_detection_demo.png") 
    print("- summary_demo.png")
    
    # Print some basic statistics
    print(f"\nBasic Statistics:")
    print(f"Mixed audio range: [{mixed_audio.min():.3f}, {mixed_audio.max():.3f}]")
    print(f"Source 1 range: [{separated_sources[0].min():.3f}, {separated_sources[0].max():.3f}]")
    print(f"Source 2 range: [{separated_sources[1].min():.3f}, {separated_sources[1].max():.3f}]")
    
    # Compute simple metrics
    from src.utils.audio import compute_si_sdr, compute_sdr
    
    si_sdr = compute_si_sdr(separated_sources, target_sources)
    sdr = compute_sdr(separated_sources, target_sources)
    
    print(f"\nSeparation Quality:")
    print(f"SI-SDR: {si_sdr.mean():.2f} ± {si_sdr.std():.2f} dB")
    print(f"SDR: {sdr.mean():.2f} ± {sdr.std():.2f} dB")


if __name__ == "__main__":
    main()
