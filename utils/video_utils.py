"""
Utility functions for video processing and handling.
"""

import cv2
import os
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import logging


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    frame_rate: Optional[float] = None,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    max_frames: Optional[int] = None
) -> List[str]:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        frame_rate: Target frame rate (if None, use original)
        start_time: Start time in seconds
        end_time: End time in seconds (if None, use full video)
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of extracted frame filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logging.info(f"Video FPS: {original_fps}, Total frames: {total_frames}")
    
    # Calculate frame indices to extract
    start_frame = int(start_time * original_fps)
    end_frame = int(end_time * original_fps) if end_time else total_frames
    
    if frame_rate:
        # Calculate frame step for target frame rate
        frame_step = max(1, int(original_fps / frame_rate))
    else:
        frame_step = 1
    
    extracted_files = []
    frame_count = 0
    
    for frame_idx in range(start_frame, end_frame, frame_step):
        if max_frames and len(extracted_files) >= max_frames:
            break
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save frame
        filename = f"frame_{frame_idx:06d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        extracted_files.append(filename)
        frame_count += 1
    
    cap.release()
    
    logging.info(f"Extracted {len(extracted_files)} frames to {output_dir}")
    return extracted_files


def create_video_from_frames(
    frames_dir: str,
    output_path: str,
    fps: float = 30.0,
    codec: str = 'mp4v',
    frame_pattern: str = "*.png"
) -> str:
    """
    Create a video from a sequence of frames.
    
    Args:
        frames_dir: Directory containing frame images
        output_path: Output video file path
        fps: Frame rate for output video
        codec: Video codec to use
        frame_pattern: Pattern to match frame files
        
    Returns:
        Path to created video file
    """
    frame_files = sorted(list(Path(frames_dir).glob(frame_pattern)))
    
    if not frame_files:
        raise ValueError(f"No frames found in {frames_dir} with pattern {frame_pattern}")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    height, width, _ = first_frame.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        if frame is not None:
            out.write(frame)
    
    out.release()
    
    logging.info(f"Created video: {output_path}")
    return output_path


def resize_frames(
    input_dir: str,
    output_dir: str,
    target_size: Tuple[int, int],
    maintain_aspect_ratio: bool = True
) -> List[str]:
    """
    Resize all frames in a directory.
    
    Args:
        input_dir: Directory containing input frames
        output_dir: Directory to save resized frames
        target_size: Target size as (width, height)
        maintain_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        List of resized frame filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    frame_files = []
    for ext in ['.png', '.jpg', '.jpeg']:
        frame_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
    
    frame_files.sort()
    resized_files = []
    
    for frame_file in frame_files:
        input_path = os.path.join(input_dir, frame_file)
        output_path = os.path.join(output_dir, frame_file)
        
        # Read and resize frame
        frame = cv2.imread(input_path)
        if frame is None:
            continue
        
        if maintain_aspect_ratio:
            # Calculate new size maintaining aspect ratio
            h, w = frame.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scale factor
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize and pad if necessary
            resized = cv2.resize(frame, (new_w, new_h))
            
            # Create padded image
            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            cv2.imwrite(output_path, padded)
        else:
            # Direct resize without maintaining aspect ratio
            resized = cv2.resize(frame, target_size)
            cv2.imwrite(output_path, resized)
        
        resized_files.append(frame_file)
    
    logging.info(f"Resized {len(resized_files)} frames to {target_size}")
    return resized_files


def validate_frame_sequence(frame_dir: str) -> Tuple[bool, List[str]]:
    """
    Validate that a directory contains a proper frame sequence.
    
    Args:
        frame_dir: Directory to validate
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if not os.path.exists(frame_dir):
        return False, [f"Directory does not exist: {frame_dir}"]
    
    # Get frame files
    extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    frame_files = []
    for ext in extensions:
        frame_files.extend([f for f in os.listdir(frame_dir) if f.endswith(ext)])
    
    if not frame_files:
        return False, ["No image files found in directory"]
    
    # Check if files can be sorted numerically
    def _extract_frame_number(filename):
        """Extract frame number from filename, handling various naming patterns."""
        basename = os.path.splitext(filename)[0]
        # Try to extract numbers from the filename
        import re
        numbers = re.findall(r'\d+', basename)
        if numbers:
            # Use the last number found (common pattern: frame_12345.png)
            return int(numbers[-1])
        else:
            # Fallback: try to convert entire basename to int
            try:
                return int(basename)
            except ValueError:
                # If no numbers found, sort alphabetically by returning a hash
                return hash(basename)
    
    try:
        frame_files.sort(key=_extract_frame_number)
    except Exception as e:
        issues.append(f"Frame files cannot be sorted: {str(e)}")
        frame_files.sort()  # Fallback to alphabetical sort
    
    # Check for consistent image dimensions
    first_frame_path = os.path.join(frame_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        return False, [f"Cannot read first frame: {first_frame_path}"]
    
    expected_shape = first_frame.shape
    
    # Sample check a few frames for consistency
    check_indices = [0, len(frame_files)//2, len(frame_files)-1]
    for idx in check_indices:
        if idx < len(frame_files):
            frame_path = os.path.join(frame_dir, frame_files[idx])
            frame = cv2.imread(frame_path)
            if frame is None:
                issues.append(f"Cannot read frame: {frame_path}")
            elif frame.shape != expected_shape:
                issues.append(f"Inconsistent frame dimensions: {frame_path}")
    
    # Check for reasonable frame count
    if len(frame_files) < 2:
        issues.append("Very few frames found (less than 2)")
    elif len(frame_files) > 10000:
        issues.append("Very large number of frames (more than 10000)")
    
    return len(issues) == 0, issues


def get_frame_info(frame_dir: str) -> dict:
    """
    Get information about a frame sequence.
    
    Args:
        frame_dir: Directory containing frames
        
    Returns:
        Dictionary with frame sequence information
    """
    info = {
        'frame_count': 0,
        'dimensions': None,
        'file_sizes': [],
        'extensions': set(),
        'total_size_mb': 0.0
    }
    
    if not os.path.exists(frame_dir):
        return info
    
    extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    frame_files = []
    for ext in extensions:
        frame_files.extend([f for f in os.listdir(frame_dir) if f.endswith(ext)])
    
    info['frame_count'] = len(frame_files)
    
    if frame_files:
        # Get dimensions from first frame
        first_frame_path = os.path.join(frame_dir, frame_files[0])
        first_frame = cv2.imread(first_frame_path)
        if first_frame is not None:
            info['dimensions'] = first_frame.shape
        
        # Collect file information
        total_size = 0
        for frame_file in frame_files:
            frame_path = os.path.join(frame_dir, frame_file)
            file_size = os.path.getsize(frame_path)
            info['file_sizes'].append(file_size)
            info['extensions'].add(os.path.splitext(frame_file)[1].lower())
            total_size += file_size
        
        info['total_size_mb'] = total_size / (1024 * 1024)
        info['extensions'] = list(info['extensions'])
    
    return info
