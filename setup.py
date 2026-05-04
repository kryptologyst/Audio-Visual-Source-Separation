#!/usr/bin/env python3
"""Setup script for Audio-Visual Source Separation project."""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def create_directories():
    """Create necessary directories."""
    print("📁 Creating project directories...")
    
    directories = [
        "data/audio",
        "data/video", 
        "data/annotations",
        "checkpoints",
        "logs",
        "assets/visualizations",
        "assets/evaluation",
        "assets/demo_output",
        "notebooks",
        "scripts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {directory}")
    
    print("✅ Directory structure created")


def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    # Install dependencies
    success = run_command(
        "pip install -r requirements.txt",
        "Installing Python dependencies"
    )
    
    return success


def setup_pre_commit():
    """Setup pre-commit hooks."""
    print("🔧 Setting up pre-commit hooks...")
    
    # Install pre-commit
    success = run_command(
        "pip install pre-commit",
        "Installing pre-commit"
    )
    
    if not success:
        return False
    
    # Install pre-commit hooks
    success = run_command(
        "pre-commit install",
        "Installing pre-commit hooks"
    )
    
    return success


def create_sample_data():
    """Create sample data for demonstration."""
    print("🎵 Creating sample data...")
    
    # Create a simple sample data structure
    sample_data = {
        "samples": [
            {
                "id": "sample_001",
                "audio_path": "audio/sample_001.wav",
                "video_path": "video/sample_001.mp4",
                "duration": 5.0,
                "num_speakers": 2,
                "speaker_labels": [0, 1],
                "face_boxes": [
                    [[100, 100, 80, 80], [200, 150, 90, 90]],
                    [[105, 105, 80, 80], [205, 155, 90, 90]],
                    [[110, 110, 80, 80], [210, 160, 90, 90]]
                ]
            }
        ],
        "speakers": ["speaker_0", "speaker_1"],
        "total_duration": 5.0
    }
    
    import json
    
    with open("data/annotations.json", "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print("✅ Sample data created")


def run_tests():
    """Run basic tests."""
    print("🧪 Running tests...")
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("⚠️  pytest not installed, skipping tests")
        return True
    
    # Run tests
    success = run_command(
        "python -m pytest tests/ -v",
        "Running unit tests"
    )
    
    return success


def create_demo_output():
    """Create demo output."""
    print("🎬 Running demo...")
    
    success = run_command(
        "python scripts/demo.py",
        "Running demo script"
    )
    
    return success


def main():
    """Main setup function."""
    print("🚀 Setting up Audio-Visual Source Separation project")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Setup pre-commit (optional)
    print("\n🔧 Setting up development tools...")
    setup_pre_commit()
    
    # Create sample data
    create_sample_data()
    
    # Run tests
    print("\n🧪 Testing installation...")
    if not run_tests():
        print("⚠️  Some tests failed, but continuing...")
    
    # Run demo
    print("\n🎬 Testing demo...")
    if not create_demo_output():
        print("⚠️  Demo failed, but setup completed")
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the interactive demo: streamlit run demo/app.py")
    print("2. Train a model: python scripts/train.py --config configs/train.yaml")
    print("3. Evaluate a model: python scripts/evaluate.py --checkpoint checkpoints/best_model.pt")
    print("4. Check out the notebooks/ directory for examples")
    print("\nFor more information, see the README.md file")


if __name__ == "__main__":
    main()
