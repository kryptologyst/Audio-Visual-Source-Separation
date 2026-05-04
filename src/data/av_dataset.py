"""Data loading and preprocessing for audio-visual source separation."""

import os
import json
import torch
import torchaudio
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random

from ..utils.audio import load_audio, compute_mel_spectrogram, normalize_audio
from ..utils.video import load_video_frames, detect_faces, preprocess_frame


class AVSourceSeparationDataset(Dataset):
    """Dataset for audio-visual source separation."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        sample_rate: int = 16000,
        video_fps: int = 25,
        max_audio_length: float = 10.0,
        max_frames: int = 250,
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.video_fps = video_fps
        self.max_audio_length = max_audio_length
        self.max_frames = max_frames
        self.augment = augment and split == "train"
        
        # Load dataset metadata
        self.metadata = self._load_metadata()
        self.samples = self._get_samples()
        
        # Audio transforms
        self.audio_transforms = self._get_audio_transforms()
        
        # Video transforms
        self.video_transforms = self._get_video_transforms()
    
    def _load_metadata(self) -> Dict:
        """Load dataset metadata."""
        metadata_path = self.data_dir / "annotations.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            # Create dummy metadata for demo
            return self._create_dummy_metadata()
    
    def _create_dummy_metadata(self) -> Dict:
        """Create dummy metadata for demonstration."""
        return {
            "samples": [
                {
                    "id": f"sample_{i:03d}",
                    "audio_path": f"audio/sample_{i:03d}.wav",
                    "video_path": f"video/sample_{i:03d}.mp4",
                    "duration": 5.0,
                    "num_speakers": 2,
                    "speaker_labels": [0, 1],
                    "face_boxes": [
                        [[100, 100, 80, 80], [200, 150, 90, 90]],  # Frame 0
                        [[105, 105, 80, 80], [205, 155, 90, 90]],  # Frame 1
                        # ... more frames
                    ]
                }
                for i in range(10)
            ],
            "speakers": ["speaker_0", "speaker_1"],
            "total_duration": 50.0
        }
    
    def _get_samples(self) -> List[Dict]:
        """Get samples for current split."""
        all_samples = self.metadata["samples"]
        
        # Simple split (in practice, use proper train/val/test splits)
        if self.split == "train":
            return all_samples[:8]
        elif self.split == "val":
            return all_samples[8:9]
        else:  # test
            return all_samples[9:]
    
    def _get_audio_transforms(self) -> List:
        """Get audio augmentation transforms."""
        if not self.augment:
            return []
        
        return [
            # Add noise
            lambda x: x + torch.randn_like(x) * 0.01,
            # Time stretching (simplified)
            lambda x: torch.nn.functional.interpolate(
                x.unsqueeze(0).unsqueeze(0),
                size=int(x.shape[0] * random.uniform(0.9, 1.1)),
                mode='linear',
                align_corners=False
            ).squeeze(0).squeeze(0),
            # Pitch shifting (simplified)
            lambda x: x * random.uniform(0.95, 1.05)
        ]
    
    def _get_video_transforms(self) -> List:
        """Get video augmentation transforms."""
        if not self.augment:
            return []
        
        return [
            # Random horizontal flip
            lambda x: torch.flip(x, dims=[3]) if random.random() < 0.5 else x,
            # Random brightness adjustment
            lambda x: torch.clamp(x * random.uniform(0.8, 1.2), 0, 1),
            # Random contrast adjustment
            lambda x: torch.clamp((x - 0.5) * random.uniform(0.8, 1.2) + 0.5, 0, 1)
        ]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        
        # Load audio
        audio_path = self.data_dir / sample["audio_path"]
        if audio_path.exists():
            audio, sr = load_audio(str(audio_path), self.sample_rate)
        else:
            # Create dummy audio
            audio = self._create_dummy_audio(sample["duration"])
        
        # Load video
        video_path = self.data_dir / sample["video_path"]
        if video_path.exists():
            frames = load_video_frames(str(video_path), self.video_fps, self.max_frames)
        else:
            # Create dummy video
            frames = self._create_dummy_video(sample["duration"])
        
        # Process audio
        audio = self._process_audio(audio)
        
        # Process video
        frames_tensor, face_boxes = self._process_video(frames)
        
        # Create target sources (in practice, these would be pre-separated)
        target_sources = self._create_target_sources(audio, sample["num_speakers"])
        
        return {
            "audio": audio,
            "frames": frames_tensor,
            "face_boxes": face_boxes,
            "target_sources": target_sources,
            "sample_id": sample["id"],
            "duration": sample["duration"],
            "num_speakers": sample["num_speakers"]
        }
    
    def _create_dummy_audio(self, duration: float) -> torch.Tensor:
        """Create dummy audio for demonstration."""
        num_samples = int(duration * self.sample_rate)
        
        # Create two different frequency components (simulating two speakers)
        t = torch.linspace(0, duration, num_samples)
        
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
        
        return mixed
    
    def _create_dummy_video(self, duration: float) -> List[np.ndarray]:
        """Create dummy video frames for demonstration."""
        num_frames = int(duration * self.video_fps)
        frames = []
        
        for i in range(num_frames):
            # Create a simple frame with two colored rectangles (simulating faces)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Face 1 (left side)
            cv2.rectangle(frame, (100, 100), (180, 180), (255, 0, 0), -1)
            
            # Face 2 (right side)
            cv2.rectangle(frame, (400, 150), (480, 230), (0, 255, 0), -1)
            
            frames.append(frame)
        
        return frames
    
    def _process_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Process audio waveform."""
        # Truncate or pad to max length
        max_samples = int(self.max_audio_length * self.sample_rate)
        
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        else:
            audio = torch.nn.functional.pad(audio, (0, max_samples - len(audio)))
        
        # Normalize
        audio = normalize_audio(audio)
        
        # Apply augmentations
        for transform in self.audio_transforms:
            if random.random() < 0.5:  # 50% chance for each augmentation
                audio = transform(audio)
        
        return audio
    
    def _process_video(self, frames: List[np.ndarray]) -> Tuple[torch.Tensor, List[List[Tuple[int, int, int, int]]]]:
        """Process video frames."""
        # Limit number of frames
        if len(frames) > self.max_frames:
            # Sample frames uniformly
            indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
            frames = [frames[i] for i in indices]
        
        # Detect faces in each frame
        face_boxes = []
        processed_frames = []
        
        for frame in frames:
            # Detect faces
            faces = detect_faces(frame)
            face_boxes.append(faces)
            
            # Preprocess frame
            frame_tensor = preprocess_frame(frame)
            processed_frames.append(frame_tensor)
        
        # Stack frames
        frames_tensor = torch.stack(processed_frames)
        
        # Apply augmentations
        for transform in self.video_transforms:
            if random.random() < 0.5:  # 50% chance for each augmentation
                frames_tensor = transform(frames_tensor)
        
        return frames_tensor, face_boxes
    
    def _create_target_sources(self, mixed_audio: torch.Tensor, num_speakers: int) -> torch.Tensor:
        """Create target separated sources (simplified)."""
        # In practice, these would be pre-separated sources
        # For demo, we'll create simple separated sources
        
        length = len(mixed_audio)
        mid_point = length // 2
        
        # Create two sources
        source1 = torch.zeros_like(mixed_audio)
        source2 = torch.zeros_like(mixed_audio)
        
        source1[:mid_point] = mixed_audio[:mid_point]
        source2[mid_point:] = mixed_audio[mid_point:]
        
        # Stack sources
        target_sources = torch.stack([source1, source2])
        
        return target_sources


def create_dataloader(
    dataset: AVSourceSeparationDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """Create a DataLoader for the dataset."""
    
    def collate_fn(batch):
        """Custom collate function for handling variable-length sequences."""
        # Pad sequences to the same length
        audio_lengths = [item["audio"].shape[0] for item in batch]
        max_audio_length = max(audio_lengths)
        
        frame_counts = [item["frames"].shape[0] for item in batch]
        max_frame_count = max(frame_counts)
        
        # Pad audio
        padded_audio = []
        for item in batch:
            audio = item["audio"]
            if len(audio) < max_audio_length:
                audio = torch.nn.functional.pad(audio, (0, max_audio_length - len(audio)))
            padded_audio.append(audio)
        
        # Pad frames
        padded_frames = []
        for item in batch:
            frames = item["frames"]
            if frames.shape[0] < max_frame_count:
                padding = torch.zeros(
                    max_frame_count - frames.shape[0],
                    *frames.shape[1:]
                )
                frames = torch.cat([frames, padding], dim=0)
            padded_frames.append(frames)
        
        return {
            "audio": torch.stack(padded_audio),
            "frames": torch.stack(padded_frames),
            "face_boxes": [item["face_boxes"] for item in batch],
            "target_sources": torch.stack([item["target_sources"] for item in batch]),
            "sample_ids": [item["sample_id"] for item in batch],
            "durations": torch.tensor([item["duration"] for item in batch]),
            "num_speakers": torch.tensor([item["num_speakers"] for item in batch])
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


def create_datasets(
    data_dir: str,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    **kwargs
) -> Tuple[AVSourceSeparationDataset, AVSourceSeparationDataset, AVSourceSeparationDataset]:
    """Create train, validation, and test datasets."""
    
    train_dataset = AVSourceSeparationDataset(
        data_dir=data_dir,
        split="train",
        **kwargs
    )
    
    val_dataset = AVSourceSeparationDataset(
        data_dir=data_dir,
        split="val",
        augment=False,  # No augmentation for validation
        **kwargs
    )
    
    test_dataset = AVSourceSeparationDataset(
        data_dir=data_dir,
        split="test",
        augment=False,  # No augmentation for test
        **kwargs
    )
    
    return train_dataset, val_dataset, test_dataset
