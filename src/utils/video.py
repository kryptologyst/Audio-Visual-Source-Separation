"""Video processing utilities for source separation."""

import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import Tuple, List, Optional, Union
from PIL import Image


def load_video_frames(
    video_path: str,
    target_fps: int = 25,
    max_frames: Optional[int] = None
) -> List[np.ndarray]:
    """Load video frames from file.
    
    Args:
        video_path: Path to video file.
        target_fps: Target frame rate.
        max_frames: Maximum number of frames to load.
        
    Returns:
        List of video frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(fps / target_fps))
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
            if max_frames and len(frames) >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    return frames


def detect_faces(
    frame: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (30, 30)
) -> List[Tuple[int, int, int, int]]:
    """Detect faces in a video frame.
    
    Args:
        frame: Input video frame.
        scale_factor: Scale factor for face detection.
        min_neighbors: Minimum neighbors for face detection.
        min_size: Minimum face size.
        
    Returns:
        List of face bounding boxes (x, y, w, h).
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )
    
    return faces.tolist() if len(faces) > 0 else []


def extract_face_regions(
    frame: np.ndarray,
    face_boxes: List[Tuple[int, int, int, int]],
    target_size: Tuple[int, int] = (224, 224)
) -> List[np.ndarray]:
    """Extract face regions from frame.
    
    Args:
        frame: Input video frame.
        face_boxes: List of face bounding boxes.
        target_size: Target size for face crops.
        
    Returns:
        List of cropped face regions.
    """
    face_regions = []
    
    for x, y, w, h in face_boxes:
        # Extract face region
        face = frame[y:y+h, x:x+w]
        
        # Resize to target size
        face_resized = cv2.resize(face, target_size)
        face_regions.append(face_resized)
    
    return face_regions


def preprocess_frame(
    frame: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> torch.Tensor:
    """Preprocess video frame for model input.
    
    Args:
        frame: Input video frame.
        target_size: Target frame size.
        normalize: Whether to normalize pixel values.
        
    Returns:
        Preprocessed frame tensor.
    """
    # Convert to PIL Image
    pil_image = Image.fromarray(frame)
    
    # Define transforms
    transform_list = [
        transforms.Resize(target_size),
        transforms.ToTensor()
    ]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    transform = transforms.Compose(transform_list)
    
    return transform(pil_image)


def compute_optical_flow(
    frame1: np.ndarray,
    frame2: np.ndarray,
    method: str = "farneback"
) -> np.ndarray:
    """Compute optical flow between two frames.
    
    Args:
        frame1: First frame.
        frame2: Second frame.
        method: Optical flow method ('farneback', 'lucas_kanade').
        
    Returns:
        Optical flow field.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    
    if method == "farneback":
        flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None)
    elif method == "lucas_kanade":
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
    else:
        raise ValueError(f"Unknown optical flow method: {method}")
    
    return flow


def track_faces_across_frames(
    frames: List[np.ndarray],
    initial_faces: List[Tuple[int, int, int, int]]
) -> List[List[Tuple[int, int, int, int]]]:
    """Track faces across video frames.
    
    Args:
        frames: List of video frames.
        initial_faces: Initial face detections.
        
    Returns:
        List of face tracks across frames.
    """
    if not frames or not initial_faces:
        return []
    
    # Initialize tracker
    tracker = cv2.MultiTracker_create()
    
    # Add initial faces to tracker
    for face_box in initial_faces:
        tracker.add(cv2.TrackerKCF_create(), frames[0], face_box)
    
    face_tracks = [initial_faces]
    
    # Track faces in subsequent frames
    for frame in frames[1:]:
        success, boxes = tracker.update(frame)
        
        if success:
            face_tracks.append([tuple(box) for box in boxes])
        else:
            # Re-detect faces if tracking fails
            detected_faces = detect_faces(frame)
            face_tracks.append(detected_faces)
            
            # Re-initialize tracker
            tracker = cv2.MultiTracker_create()
            for face_box in detected_faces:
                tracker.add(cv2.TrackerKCF_create(), frame, face_box)
    
    return face_tracks


def compute_temporal_features(
    frames: List[torch.Tensor],
    window_size: int = 5
) -> torch.Tensor:
    """Compute temporal features from video frames.
    
    Args:
        frames: List of frame tensors.
        window_size: Temporal window size.
        
    Returns:
        Temporal features tensor.
    """
    if len(frames) < window_size:
        # Pad with last frame if not enough frames
        frames = frames + [frames[-1]] * (window_size - len(frames))
    
    # Stack frames
    frame_stack = torch.stack(frames[:window_size])
    
    # Compute temporal differences
    temporal_diff = torch.diff(frame_stack, dim=0)
    
    # Compute temporal statistics
    temporal_mean = torch.mean(frame_stack, dim=0)
    temporal_std = torch.std(frame_stack, dim=0)
    
    # Concatenate features
    temporal_features = torch.cat([
        temporal_mean,
        temporal_std,
        temporal_diff.mean(dim=0)
    ], dim=0)
    
    return temporal_features


def align_audio_video_timestamps(
    audio_duration: float,
    video_frames: int,
    video_fps: float
) -> Tuple[List[float], List[int]]:
    """Align audio and video timestamps.
    
    Args:
        audio_duration: Audio duration in seconds.
        video_frames: Number of video frames.
        video_fps: Video frame rate.
        
    Returns:
        Tuple of (audio_timestamps, video_frame_indices).
    """
    video_duration = video_frames / video_fps
    
    # Use shorter duration
    duration = min(audio_duration, video_duration)
    
    # Generate timestamps
    audio_timestamps = np.linspace(0, duration, int(audio_duration * 100))  # 100 Hz
    video_frame_indices = np.linspace(0, video_frames - 1, int(duration * video_fps))
    
    return audio_timestamps.tolist(), video_frame_indices.astype(int).tolist()
