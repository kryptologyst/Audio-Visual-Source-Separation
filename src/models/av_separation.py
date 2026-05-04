"""Audio-Visual Source Separation model architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import torchvision.models as models
from transformers import AutoModel


class VisualEncoder(nn.Module):
    """Visual encoder for processing video frames and face detection."""
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: int = 2048,
        num_faces: int = 2
    ):
        super().__init__()
        
        self.num_faces = num_faces
        self.output_dim = output_dim
        
        # Load backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone.fc = nn.Identity()  # Remove final classification layer
            backbone_dim = 2048
        elif backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            self.backbone.fc = nn.Identity()
            backbone_dim = 512
        elif backbone == "vit_base":
            self.backbone = models.vit_b_16(pretrained=pretrained)
            backbone_dim = 768
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection layer
        if backbone_dim != output_dim:
            self.projection = nn.Linear(backbone_dim, output_dim)
        else:
            self.projection = nn.Identity()
        
        # Face-specific processing
        self.face_processor = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim // 2, output_dim)
        )
    
    def forward(self, frames: torch.Tensor, face_boxes: Optional[List[List[Tuple[int, int, int, int]]]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through visual encoder.
        
        Args:
            frames: Video frames tensor [B, T, C, H, W].
            face_boxes: Optional face bounding boxes for each frame.
            
        Returns:
            Dictionary containing visual features.
        """
        batch_size, num_frames = frames.shape[:2]
        
        # Reshape for batch processing
        frames_flat = frames.view(-1, *frames.shape[2:])
        
        # Extract features
        visual_features = self.backbone(frames_flat)
        visual_features = self.projection(visual_features)
        
        # Reshape back to temporal dimension
        visual_features = visual_features.view(batch_size, num_frames, -1)
        
        # Process face-specific features if face boxes provided
        face_features = None
        if face_boxes is not None:
            face_features = self._process_face_features(frames, face_boxes)
        
        return {
            "global_features": visual_features,
            "face_features": face_features,
            "temporal_features": self._compute_temporal_features(visual_features)
        }
    
    def _process_face_features(self, frames: torch.Tensor, face_boxes: List[List[Tuple[int, int, int, int]]]) -> torch.Tensor:
        """Process face-specific features."""
        batch_size, num_frames = frames.shape[:2]
        face_features_list = []
        
        for b in range(batch_size):
            batch_face_features = []
            for t in range(num_frames):
                frame_faces = face_boxes[b][t] if t < len(face_boxes[b]) else []
                
                if frame_faces:
                    # Extract face regions and process
                    face_feats = []
                    for face_box in frame_faces[:self.num_faces]:  # Limit to max faces
                        x, y, w, h = face_box
                        face_region = frames[b, t, :, y:y+h, x:x+w]
                        
                        # Resize face region
                        face_region = F.interpolate(
                            face_region.unsqueeze(0),
                            size=(224, 224),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                        
                        # Extract features
                        face_feat = self.backbone(face_region)
                        face_feat = self.projection(face_feat)
                        face_feat = self.face_processor(face_feat)
                        face_feats.append(face_feat)
                    
                    # Pad or truncate to fixed number of faces
                    while len(face_feats) < self.num_faces:
                        face_feats.append(torch.zeros_like(face_feats[0]))
                    
                    batch_face_features.append(torch.stack(face_feats[:self.num_faces]))
                else:
                    # No faces detected
                    batch_face_features.append(torch.zeros(self.num_faces, self.output_dim))
            
            face_features_list.append(torch.stack(batch_face_features))
        
        return torch.stack(face_features_list)
    
    def _compute_temporal_features(self, visual_features: torch.Tensor) -> torch.Tensor:
        """Compute temporal features from visual features."""
        # Compute temporal differences
        temporal_diff = torch.diff(visual_features, dim=1)
        
        # Compute temporal statistics
        temporal_mean = torch.mean(visual_features, dim=1)
        temporal_std = torch.std(visual_features, dim=1)
        
        return torch.cat([temporal_mean, temporal_std], dim=-1)


class AudioEncoder(nn.Module):
    """Audio encoder for processing audio waveforms."""
    
    def __init__(
        self,
        input_channels: int = 1,
        hidden_dim: int = 512,
        num_layers: int = 6,
        kernel_size: int = 3,
        stride: int = 2,
        encoder_type: str = "conv1d"
    ):
        super().__init__()
        
        self.encoder_type = encoder_type
        
        if encoder_type == "conv1d":
            self.encoder = self._build_conv1d_encoder(
                input_channels, hidden_dim, num_layers, kernel_size, stride
            )
        elif encoder_type == "transformer":
            self.encoder = self._build_transformer_encoder(
                input_channels, hidden_dim, num_layers
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    def _build_conv1d_encoder(
        self,
        input_channels: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int,
        stride: int
    ) -> nn.Module:
        """Build 1D convolutional encoder."""
        layers = []
        in_channels = input_channels
        
        for i in range(num_layers):
            out_channels = hidden_dim * (2 ** i)
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _build_transformer_encoder(
        self,
        input_channels: int,
        hidden_dim: int,
        num_layers: int
    ) -> nn.Module:
        """Build transformer encoder."""
        # Input projection
        self.input_projection = nn.Linear(input_channels, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1000, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass through audio encoder.
        
        Args:
            audio: Audio waveform tensor [B, T] or [B, C, T].
            
        Returns:
            Audio features tensor.
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # Add channel dimension
        
        if self.encoder_type == "conv1d":
            return self.encoder(audio)
        elif self.encoder_type == "transformer":
            # Reshape for transformer
            B, C, T = audio.shape
            audio_flat = audio.transpose(1, 2)  # [B, T, C]
            
            # Project to hidden dimension
            audio_proj = self.input_projection(audio_flat)
            
            # Add positional encoding
            seq_len = audio_proj.shape[1]
            pos_enc = self.pos_encoding[:seq_len].unsqueeze(0).expand(B, -1, -1)
            audio_proj = audio_proj + pos_enc
            
            # Apply transformer
            return self.encoder(audio_proj).transpose(1, 2)  # Back to [B, C, T]


class CrossModalFusion(nn.Module):
    """Cross-modal fusion module using attention mechanisms."""
    
    def __init__(
        self,
        audio_dim: int,
        visual_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        self.visual_projection = nn.Linear(visual_dim, hidden_dim)
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through cross-modal fusion.
        
        Args:
            audio_features: Audio features [B, C, T].
            visual_features: Visual features [B, T, C].
            
        Returns:
            Fused features.
        """
        # Project to common dimension
        audio_proj = self.audio_projection(audio_features.transpose(1, 2))  # [B, T, C]
        visual_proj = self.visual_projection(visual_features)
        
        # Apply cross-attention layers
        fused_features = audio_proj
        
        for attention, layer_norm, ffn in zip(
            self.cross_attention_layers,
            self.layer_norms,
            self.ffns
        ):
            # Cross-attention: audio queries, visual keys/values
            attn_output, _ = attention(
                query=fused_features,
                key=visual_proj,
                value=visual_proj
            )
            
            # Residual connection and layer norm
            fused_features = layer_norm(fused_features + self.dropout(attn_output))
            
            # Feed-forward network
            ffn_output = ffn(fused_features)
            fused_features = layer_norm(fused_features + self.dropout(ffn_output))
        
        return fused_features.transpose(1, 2)  # Back to [B, C, T]


class SourceSeparationDecoder(nn.Module):
    """Decoder for source separation."""
    
    def __init__(
        self,
        input_dim: int,
        output_channels: int = 2,
        kernel_size: int = 3,
        stride: int = 2,
        num_layers: int = 4
    ):
        super().__init__()
        
        self.output_channels = output_channels
        
        # Build decoder layers
        layers = []
        in_channels = input_dim
        
        for i in range(num_layers):
            out_channels = input_dim // (2 ** (i + 1))
            if i == num_layers - 1:
                out_channels = output_channels
            
            layers.extend([
                nn.ConvTranspose1d(
                    in_channels, out_channels, kernel_size, stride,
                    padding=kernel_size//2, output_padding=stride-1
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU() if i < num_layers - 1 else nn.Tanh()
            ])
            
            in_channels = out_channels
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder.
        
        Args:
            fused_features: Fused audio-visual features [B, C, T].
            
        Returns:
            Separated audio sources [B, num_sources, T].
        """
        return self.decoder(fused_features)


class AVSourceSeparator(nn.Module):
    """Main Audio-Visual Source Separation model."""
    
    def __init__(
        self,
        visual_backbone: str = "resnet50",
        audio_encoder_type: str = "conv1d",
        hidden_dim: int = 512,
        num_sources: int = 2,
        fusion_type: str = "cross_attention"
    ):
        super().__init__()
        
        self.num_sources = num_sources
        
        # Visual encoder
        self.visual_encoder = VisualEncoder(
            backbone=visual_backbone,
            output_dim=hidden_dim
        )
        
        # Audio encoder
        self.audio_encoder = AudioEncoder(
            hidden_dim=hidden_dim,
            encoder_type=audio_encoder_type
        )
        
        # Get actual audio encoder output dimension
        dummy_audio = torch.randn(1, 1000)
        with torch.no_grad():
            audio_feat = self.audio_encoder(dummy_audio)
        self.audio_dim = audio_feat.shape[1]
        
        # Fusion module
        if fusion_type == "cross_attention":
            self.fusion = CrossModalFusion(
                audio_dim=self.audio_dim,
                visual_dim=hidden_dim,
                hidden_dim=hidden_dim
            )
        elif fusion_type == "late_fusion":
            self.fusion = nn.Sequential(
                nn.Linear(self.audio_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        self.fusion_type = fusion_type
        
        # Decoder
        self.decoder = SourceSeparationDecoder(
            input_dim=hidden_dim,
            output_channels=num_sources
        )
    
    def forward(
        self,
        audio: torch.Tensor,
        frames: torch.Tensor,
        face_boxes: Optional[List[List[Tuple[int, int, int, int]]]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            audio: Audio waveform [B, T].
            frames: Video frames [B, T, C, H, W].
            face_boxes: Optional face bounding boxes.
            
        Returns:
            Dictionary containing separated sources and intermediate features.
        """
        # Encode audio and visual features
        audio_features = self.audio_encoder(audio)
        visual_outputs = self.visual_encoder(frames, face_boxes)
        visual_features = visual_outputs["global_features"]
        
        # Fuse audio and visual features
        if self.fusion_type == "cross_attention":
            fused_features = self.fusion(audio_features, visual_features)
        elif self.fusion_type == "late_fusion":
            # Simple concatenation and projection
            audio_flat = audio_features.mean(dim=-1)  # [B, C]
            visual_flat = visual_features.mean(dim=1)  # [B, C]
            fused_flat = torch.cat([audio_flat, visual_flat], dim=-1)
            fused_features = self.fusion(fused_flat).unsqueeze(-1).expand(-1, -1, audio_features.shape[-1])
        
        # Decode to separated sources
        separated_sources = self.decoder(fused_features)
        
        return {
            "separated_sources": separated_sources,
            "audio_features": audio_features,
            "visual_features": visual_features,
            "fused_features": fused_features,
            "face_features": visual_outputs.get("face_features")
        }
    
    @classmethod
    def from_config(cls, config: Dict) -> "AVSourceSeparator":
        """Create model from configuration dictionary."""
        return cls(
            visual_backbone=config.get("visual_backbone", "resnet50"),
            audio_encoder_type=config.get("audio_encoder_type", "conv1d"),
            hidden_dim=config.get("hidden_dim", 512),
            num_sources=config.get("num_sources", 2),
            fusion_type=config.get("fusion_type", "cross_attention")
        )
