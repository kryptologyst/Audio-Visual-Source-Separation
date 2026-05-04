"""Tests for audio-visual source separation models."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.av_separation import AVSourceSeparator, VisualEncoder, AudioEncoder, CrossModalFusion
from src.losses.av_losses import SISDRLoss, SpectralLoss, MultiTaskLoss
from src.utils.device import get_device, set_seed
from src.utils.audio import compute_si_sdr, compute_sdr
from src.utils.video import detect_faces, preprocess_frame


class TestAVSourceSeparator:
    """Test cases for AVSourceSeparator model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        set_seed(42)
        self.device = get_device("cpu")
        
        self.model = AVSourceSeparator(
            visual_backbone="resnet18",  # Use smaller model for testing
            audio_encoder_type="conv1d",
            hidden_dim=128,
            num_sources=2,
            fusion_type="cross_attention"
        ).to(self.device)
        
        self.batch_size = 2
        self.audio_length = 16000  # 1 second at 16kHz
        self.num_frames = 10
        self.frame_size = (224, 224)
        
        # Create dummy data
        self.audio = torch.randn(self.batch_size, self.audio_length).to(self.device)
        self.frames = torch.randn(self.batch_size, self.num_frames, 3, *self.frame_size).to(self.device)
        self.face_boxes = [
            [[(100, 100, 80, 80), (200, 150, 90, 90)] for _ in range(self.num_frames)]
            for _ in range(self.batch_size)
        ]
    
    def test_model_forward(self):
        """Test model forward pass."""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(self.audio, self.frames, self.face_boxes)
        
        # Check output structure
        assert "separated_sources" in outputs
        assert "audio_features" in outputs
        assert "visual_features" in outputs
        assert "fused_features" in outputs
        
        # Check output shapes
        assert outputs["separated_sources"].shape == (self.batch_size, 2, self.audio_length)
        assert outputs["audio_features"].shape[0] == self.batch_size
        assert outputs["visual_features"].shape == (self.batch_size, self.num_frames, 128)
        assert outputs["fused_features"].shape[0] == self.batch_size
    
    def test_model_gradient_flow(self):
        """Test gradient flow through the model."""
        self.model.train()
        
        outputs = self.model(self.audio, self.frames, self.face_boxes)
        
        # Compute a simple loss
        loss = torch.mean(outputs["separated_sources"])
        loss.backward()
        
        # Check that gradients are computed
        for param in self.model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
    
    def test_different_fusion_types(self):
        """Test different fusion types."""
        fusion_types = ["cross_attention", "late_fusion"]
        
        for fusion_type in fusion_types:
            model = AVSourceSeparator(
                visual_backbone="resnet18",
                audio_encoder_type="conv1d",
                hidden_dim=128,
                num_sources=2,
                fusion_type=fusion_type
            ).to(self.device)
            
            model.eval()
            with torch.no_grad():
                outputs = model(self.audio, self.frames, self.face_boxes)
            
            assert outputs["separated_sources"].shape == (self.batch_size, 2, self.audio_length)


class TestVisualEncoder:
    """Test cases for VisualEncoder."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = get_device("cpu")
        
        self.encoder = VisualEncoder(
            backbone="resnet18",
            pretrained=False,
            output_dim=256,
            num_faces=2
        ).to(self.device)
        
        self.batch_size = 2
        self.num_frames = 5
        self.frame_size = (224, 224)
        
        self.frames = torch.randn(self.batch_size, self.num_frames, 3, *self.frame_size).to(self.device)
        self.face_boxes = [
            [[(100, 100, 80, 80), (200, 150, 90, 90)] for _ in range(self.num_frames)]
            for _ in range(self.batch_size)
        ]
    
    def test_visual_encoder_forward(self):
        """Test visual encoder forward pass."""
        self.encoder.eval()
        
        with torch.no_grad():
            outputs = self.encoder(self.frames, self.face_boxes)
        
        assert "global_features" in outputs
        assert "face_features" in outputs
        assert "temporal_features" in outputs
        
        assert outputs["global_features"].shape == (self.batch_size, self.num_frames, 256)
        assert outputs["face_features"].shape == (self.batch_size, self.num_frames, 2, 256)
        assert outputs["temporal_features"].shape == (self.batch_size, 512)  # 2 * 256
    
    def test_no_faces(self):
        """Test visual encoder without face boxes."""
        self.encoder.eval()
        
        with torch.no_grad():
            outputs = self.encoder(self.frames, None)
        
        assert outputs["face_features"] is None
        assert outputs["global_features"].shape == (self.batch_size, self.num_frames, 256)


class TestAudioEncoder:
    """Test cases for AudioEncoder."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = get_device("cpu")
        
        self.encoder = AudioEncoder(
            input_channels=1,
            hidden_dim=256,
            num_layers=4,
            encoder_type="conv1d"
        ).to(self.device)
        
        self.batch_size = 2
        self.audio_length = 16000
        
        self.audio = torch.randn(self.batch_size, self.audio_length).to(self.device)
    
    def test_audio_encoder_forward(self):
        """Test audio encoder forward pass."""
        self.encoder.eval()
        
        with torch.no_grad():
            features = self.encoder(self.audio)
        
        assert features.shape[0] == self.batch_size
        assert features.shape[1] == 256  # hidden_dim
        assert features.shape[2] <= self.audio_length  # May be downsampled
    
    def test_transformer_encoder(self):
        """Test transformer audio encoder."""
        encoder = AudioEncoder(
            input_channels=1,
            hidden_dim=256,
            num_layers=2,
            encoder_type="transformer"
        ).to(self.device)
        
        encoder.eval()
        
        with torch.no_grad():
            features = encoder(self.audio)
        
        assert features.shape[0] == self.batch_size
        assert features.shape[1] == 256


class TestLossFunctions:
    """Test cases for loss functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = get_device("cpu")
        
        self.batch_size = 2
        self.num_sources = 2
        self.audio_length = 16000
        
        self.estimated = torch.randn(self.batch_size, self.num_sources, self.audio_length).to(self.device)
        self.target = torch.randn(self.batch_size, self.num_sources, self.audio_length).to(self.device)
    
    def test_si_sdr_loss(self):
        """Test SI-SDR loss function."""
        loss_fn = SISDRLoss()
        
        loss = loss_fn(self.estimated, self.target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert not torch.isnan(loss)
    
    def test_spectral_loss(self):
        """Test spectral loss function."""
        loss_fn = SpectralLoss()
        
        loss = loss_fn(self.estimated, self.target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert not torch.isnan(loss)
    
    def test_multi_task_loss(self):
        """Test multi-task loss function."""
        loss_fn = MultiTaskLoss()
        
        outputs = {
            "separated_sources": self.estimated,
            "face_features": torch.randn(self.batch_size, 5, 2, 256).to(self.device),
            "audio_features": torch.randn(self.batch_size, 256, 1000).to(self.device)
        }
        
        targets = {"target_sources": self.target}
        
        losses = loss_fn(outputs, targets)
        
        assert "total" in losses
        assert isinstance(losses["total"], torch.Tensor)
        assert not torch.isnan(losses["total"])


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_compute_si_sdr(self):
        """Test SI-SDR computation."""
        estimated = torch.randn(2, 16000)
        target = torch.randn(2, 16000)
        
        si_sdr = compute_si_sdr(estimated, target)
        
        assert isinstance(si_sdr, torch.Tensor)
        assert si_sdr.shape == (2,)
        assert not torch.isnan(si_sdr).any()
    
    def test_compute_sdr(self):
        """Test SDR computation."""
        estimated = torch.randn(2, 16000)
        target = torch.randn(2, 16000)
        
        sdr = compute_sdr(estimated, target)
        
        assert isinstance(sdr, torch.Tensor)
        assert sdr.shape == (2,)
        assert not torch.isnan(sdr).any()
    
    def test_face_detection(self):
        """Test face detection."""
        # Create a simple test image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add a simple rectangle (simulating a face)
        cv2.rectangle(frame, (100, 100), (180, 180), (255, 255, 255), -1)
        
        faces = detect_faces(frame)
        
        # Should detect at least one face (or none if detection fails)
        assert isinstance(faces, list)
        assert all(isinstance(face, tuple) and len(face) == 4 for face in faces)
    
    def test_preprocess_frame(self):
        """Test frame preprocessing."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        processed = preprocess_frame(frame)
        
        assert isinstance(processed, torch.Tensor)
        assert processed.shape == (3, 224, 224)
        assert processed.min() >= 0
        assert processed.max() <= 1


if __name__ == "__main__":
    pytest.main([__file__])
