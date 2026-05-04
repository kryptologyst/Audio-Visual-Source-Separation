"""Interactive demo for Audio-Visual Source Separation using Streamlit."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import torch
import numpy as np
import cv2
import tempfile
import yaml
from typing import Dict, Any, Optional, Tuple
import io
import base64

from src.models.av_separation import AVSourceSeparator
from src.utils.device import get_device, set_seed
from src.utils.audio import load_audio, normalize_audio, compute_si_sdr
from src.utils.video import load_video_frames, detect_faces, preprocess_frame
from src.eval.metrics import AVSourceSeparationEvaluator
from src.viz.visualizer import AVVisualizer


class AVDemo:
    """Interactive demo for Audio-Visual Source Separation."""
    
    def __init__(self):
        self.device = get_device("auto")
        self.model = None
        self.evaluator = AVSourceSeparationEvaluator()
        self.visualizer = AVVisualizer()
        
        # Load model configuration
        self.config = self._load_config()
        
        # Initialize model
        self._load_model()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration."""
        config_path = Path("configs/model.yaml")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                "model": {
                    "visual_backbone": "resnet50",
                    "audio_encoder_type": "conv1d",
                    "hidden_dim": 512,
                    "num_sources": 2,
                    "fusion_type": "cross_attention"
                },
                "sample_rate": 16000,
                "video_fps": 25,
                "max_audio_length": 10.0,
                "max_frames": 250
            }
    
    def _load_model(self) -> None:
        """Load the trained model."""
        try:
            # Try to load from checkpoint
            checkpoint_path = Path("checkpoints/best_model.pt")
            
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                self.model = AVSourceSeparator(
                    visual_backbone=self.config["model"]["visual_backbone"],
                    audio_encoder_type=self.config["model"]["audio_encoder_type"],
                    hidden_dim=self.config["model"]["hidden_dim"],
                    num_sources=self.config["model"]["num_sources"],
                    fusion_type=self.config["model"]["fusion_type"]
                )
                
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.to(self.device)
                self.model.eval()
                
                st.success("Model loaded successfully!")
            else:
                # Create untrained model for demo
                self.model = AVSourceSeparator(
                    visual_backbone=self.config["model"]["visual_backbone"],
                    audio_encoder_type=self.config["model"]["audio_encoder_type"],
                    hidden_dim=self.config["model"]["hidden_dim"],
                    num_sources=self.config["model"]["num_sources"],
                    fusion_type=self.config["model"]["fusion_type"]
                )
                
                self.model.to(self.device)
                self.model.eval()
                
                st.warning("No trained model found. Using untrained model for demo.")
                
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()
    
    def process_audio_video(
        self,
        audio_file,
        video_file,
        num_sources: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor, List[np.ndarray], List[List[Tuple[int, int, int, int]]], Dict[str, float]]:
        """Process uploaded audio and video files."""
        
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_tmp:
            audio_tmp.write(audio_file.read())
            audio_path = audio_tmp.name
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_tmp:
            video_tmp.write(video_file.read())
            video_path = video_tmp.name
        
        try:
            # Load audio
            audio, sr = load_audio(audio_path, self.config["sample_rate"])
            
            # Truncate or pad audio
            max_samples = int(self.config["max_audio_length"] * self.config["sample_rate"])
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            else:
                audio = torch.nn.functional.pad(audio, (0, max_samples - len(audio)))
            
            audio = normalize_audio(audio)
            
            # Load video frames
            frames = load_video_frames(
                video_path,
                self.config["video_fps"],
                self.config["max_frames"]
            )
            
            # Detect faces in frames
            face_boxes = []
            processed_frames = []
            
            for frame in frames:
                faces = detect_faces(frame)
                face_boxes.append(faces)
                
                # Preprocess frame
                frame_tensor = preprocess_frame(frame)
                processed_frames.append(frame_tensor)
            
            # Stack frames
            frames_tensor = torch.stack(processed_frames).unsqueeze(0)  # Add batch dimension
            audio_tensor = audio.unsqueeze(0)  # Add batch dimension
            
            # Move to device
            audio_tensor = audio_tensor.to(self.device)
            frames_tensor = frames_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(audio_tensor, frames_tensor, [face_boxes])
            
            # Get separated sources
            separated_sources = outputs["separated_sources"][0].cpu()  # Remove batch dimension
            
            # Create dummy target sources for evaluation
            target_sources = self._create_dummy_targets(audio, num_sources)
            
            # Compute metrics
            targets = {"target_sources": target_sources.unsqueeze(0)}
            outputs_eval = {"separated_sources": separated_sources.unsqueeze(0)}
            metrics = self.evaluator.evaluate(outputs_eval, targets, [face_boxes])
            
            return audio, separated_sources, frames, face_boxes, metrics
            
        finally:
            # Clean up temporary files
            Path(audio_path).unlink(missing_ok=True)
            Path(video_path).unlink(missing_ok=True)
    
    def _create_dummy_targets(self, mixed_audio: torch.Tensor, num_sources: int) -> torch.Tensor:
        """Create dummy target sources for evaluation."""
        # Simple separation for demo purposes
        length = len(mixed_audio)
        mid_point = length // 2
        
        sources = []
        for i in range(num_sources):
            source = torch.zeros_like(mixed_audio)
            start = i * length // num_sources
            end = (i + 1) * length // num_sources
            source[start:end] = mixed_audio[start:end]
            sources.append(source)
        
        return torch.stack(sources)
    
    def create_audio_player(self, audio: torch.Tensor, sample_rate: int = 16000) -> str:
        """Create audio player widget."""
        # Convert to numpy
        audio_np = audio.numpy()
        
        # Normalize to 16-bit range
        audio_np = (audio_np * 32767).astype(np.int16)
        
        # Convert to bytes
        audio_bytes = audio_np.tobytes()
        
        # Create base64 encoded audio
        audio_b64 = base64.b64encode(audio_bytes).decode()
        
        return f"""
        <audio controls>
            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
        </audio>
        """
    
    def run(self):
        """Run the Streamlit demo."""
        st.set_page_config(
            page_title="Audio-Visual Source Separation Demo",
            page_icon="🎵",
            layout="wide"
        )
        
        st.title("🎵 Audio-Visual Source Separation Demo")
        st.markdown("""
        This demo showcases audio-visual source separation using deep learning techniques.
        Upload an audio file and a video file to separate different sound sources based on visual cues.
        """)
        
        # Safety disclaimer
        st.warning("""
        **Disclaimer**: This is a research demonstration. The model may not work reliably in all scenarios.
        Ensure you have proper consent for any audio/video processing.
        """)
        
        # Sidebar controls
        st.sidebar.header("Controls")
        
        num_sources = st.sidebar.slider(
            "Number of sources to separate",
            min_value=2,
            max_value=4,
            value=2,
            help="Number of audio sources to separate"
        )
        
        show_visualizations = st.sidebar.checkbox(
            "Show detailed visualizations",
            value=True,
            help="Display spectrograms and attention maps"
        )
        
        # File upload
        st.header("📁 Upload Files")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Audio File")
            audio_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'flac'],
                help="Upload an audio file (WAV, MP3, or FLAC)"
            )
        
        with col2:
            st.subheader("Video File")
            video_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov'],
                help="Upload a video file (MP4, AVI, or MOV)"
            )
        
        # Process files
        if audio_file is not None and video_file is not None:
            st.header("🔄 Processing")
            
            with st.spinner("Processing audio and video..."):
                try:
                    audio, separated_sources, frames, face_boxes, metrics = self.process_audio_video(
                        audio_file, video_file, num_sources
                    )
                    
                    st.success("Processing completed!")
                    
                    # Display results
                    st.header("📊 Results")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("SI-SDR", f"{metrics.get('si_sdr_mean', 0):.2f} dB")
                    
                    with col2:
                        st.metric("SDR", f"{metrics.get('sdr_mean', 0):.2f} dB")
                    
                    with col3:
                        st.metric("PESQ", f"{metrics.get('pesq_mean', 0):.2f}")
                    
                    with col4:
                        st.metric("STOI", f"{metrics.get('stoi_mean', 0):.2f}")
                    
                    # Audio players
                    st.header("🎧 Separated Audio Sources")
                    
                    st.subheader("Original Mixed Audio")
                    st.markdown(self.create_audio_player(audio), unsafe_allow_html=True)
                    
                    for i in range(min(num_sources, separated_sources.shape[0])):
                        st.subheader(f"Separated Source {i + 1}")
                        st.markdown(self.create_audio_player(separated_sources[i]), unsafe_allow_html=True)
                    
                    # Visualizations
                    if show_visualizations:
                        st.header("📈 Visualizations")
                        
                        # Create visualizations
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_path = Path(temp_dir)
                            
                            # Audio separation visualization
                            self.visualizer.visualize_audio_separation(
                                audio,
                                separated_sources,
                                save_path=temp_path / "audio_separation.png"
                            )
                            
                            # Face detection visualization
                            self.visualizer.visualize_face_detection(
                                frames,
                                face_boxes,
                                save_path=temp_path / "face_detection.png"
                            )
                            
                            # Summary visualization
                            self.visualizer.create_summary_visualization(
                                audio,
                                separated_sources,
                                frames,
                                face_boxes,
                                metrics,
                                save_path=temp_path / "summary.png"
                            )
                            
                            # Display images
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.image(str(temp_path / "audio_separation.png"), 
                                        caption="Audio Separation Results")
                                st.image(str(temp_path / "face_detection.png"), 
                                        caption="Face Detection Results")
                            
                            with col2:
                                st.image(str(temp_path / "summary.png"), 
                                        caption="Summary Visualization")
                    
                    # Face detection results
                    st.header("👥 Face Detection Results")
                    
                    total_faces = sum(len(faces) for faces in face_boxes)
                    st.info(f"Detected {total_faces} faces across {len(frames)} frames")
                    
                    # Show first few frames with face detection
                    if frames:
                        st.subheader("Sample Frames with Face Detection")
                        
                        # Display first 4 frames
                        cols = st.columns(4)
                        for i, frame in enumerate(frames[:4]):
                            with cols[i]:
                                # Draw face bounding boxes
                                frame_with_faces = frame.copy()
                                faces = face_boxes[i] if i < len(face_boxes) else []
                                
                                for j, (x, y, w, h) in enumerate(faces):
                                    cv2.rectangle(frame_with_faces, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                    cv2.putText(frame_with_faces, f"Face {j + 1}", (x, y - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                
                                st.image(frame_with_faces, caption=f"Frame {i + 1}")
                
                except Exception as e:
                    st.error(f"Error processing files: {str(e)}")
                    st.exception(e)
        
        else:
            st.info("Please upload both audio and video files to start the demo.")
        
        # Model information
        st.sidebar.header("Model Information")
        st.sidebar.markdown(f"""
        **Architecture**: {self.config['model']['fusion_type'].replace('_', ' ').title()} Fusion
        **Visual Backbone**: {self.config['model']['visual_backbone'].upper()}
        **Audio Encoder**: {self.config['model']['audio_encoder_type'].upper()}
        **Hidden Dimension**: {self.config['model']['hidden_dim']}
        **Max Sources**: {self.config['model']['num_sources']}
        """)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center'>
            <p>Audio-Visual Source Separation Demo | 
            <a href='https://github.com/kryptologyst'>github.com/kryptologyst</a></p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main function to run the demo."""
    demo = AVDemo()
    demo.run()


if __name__ == "__main__":
    main()
