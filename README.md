# Audio-Visual Source Separation

A research-ready implementation of audio-visual source separation using deep learning techniques.

## Overview

This project implements state-of-the-art audio-visual source separation methods that leverage both audio and visual cues to isolate individual sound sources from mixed audio streams. The system can separate multiple speakers in videos, isolate specific instruments in musical performances, or extract individual sound sources from complex audio-visual scenes.

## Features

- **Advanced AV Fusion**: Late, early, and cross-attention fusion strategies
- **Multiple Model Architectures**: Transformer-based, CNN-based, and hybrid approaches
- **Comprehensive Evaluation**: SI-SDR, SDR, PESQ, STOI metrics with visualizations
- **Interactive Demo**: Streamlit/Gradio interface for real-time testing
- **Production Ready**: Type hints, comprehensive testing, and modular design

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from src.models.av_separation import AVSourceSeparator
from src.data.loaders import AVDataLoader

# Load model
model = AVSourceSeparator.from_pretrained("configs/models/av_separator.yaml")

# Load data
loader = AVDataLoader("data/")
audio, video = loader.load_sample("sample_001")

# Separate sources
separated_sources = model.separate(audio, video)
```

### Demo

```bash
streamlit run demo/app.py
```

## Project Structure

```
src/
├── data/           # Data loading and preprocessing
├── models/         # Model architectures
├── losses/         # Loss functions
├── eval/           # Evaluation metrics
├── viz/            # Visualization tools
└── utils/          # Utility functions

configs/            # Configuration files
data/               # Dataset storage
scripts/            # Training and evaluation scripts
tests/              # Unit tests
assets/             # Generated outputs
demo/               # Interactive demos
```

## Models

### AVSourceSeparator
Main model implementing audio-visual source separation with:
- Visual encoder (ResNet/ViT) for face/speaker detection
- Audio encoder (Conv1D/Transformer) for audio feature extraction
- Cross-modal fusion layers
- Source separation decoder

### Supported Architectures
- **Late Fusion**: Separate audio/visual encoders with fusion at the end
- **Early Fusion**: Joint audio-visual feature extraction
- **Cross-Attention**: Attention-based fusion mechanisms

## Evaluation

The project includes comprehensive evaluation metrics:

- **Audio Quality**: SI-SDR, SDR, PESQ, STOI
- **Visual Alignment**: Face-speaker matching accuracy
- **Separation Quality**: Source-to-distortion ratio
- **Computational Efficiency**: Inference time and memory usage

## Safety and Limitations

**IMPORTANT DISCLAIMER**: This project is for research and educational purposes only. 

- Not intended for production use without proper validation
- Audio-visual source separation may not work reliably in all scenarios
- Privacy considerations: Ensure proper consent for audio/video processing
- The model may have biases based on training data demographics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{av_source_separation,
  title={Audio-Visual Source Separation},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Audio-Visual-Source-Separation}
}
```
# Audio-Visual-Source-Separation
