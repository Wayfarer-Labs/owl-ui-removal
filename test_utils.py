"""
Test utilities for UI tracker.
"""

import os
import numpy as np
import cv2
from typing import List, Tuple
import random


def create_synthetic_ui_frames(
    output_dir: str,
    num_frames: int = 20,
    frame_size: Tuple[int, int] = (800, 600),
    num_ui_elements: int = 5
) -> List[str]:
    """
    Create synthetic UI frames for testing.
    
    Args:
        output_dir: Directory to save frames
        num_frames: Number of frames to generate
        frame_size: Frame dimensions (width, height)
        num_ui_elements: Number of UI elements per frame
        
    Returns:
        List of created frame filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    width, height = frame_size
    frame_files = []
    
    # Define UI element colors and types
    ui_colors = [
        (100, 150, 255),  # Button blue
        (50, 200, 50),    # Menu green
        (255, 100, 100),  # Alert red
        (255, 200, 50),   # Icon yellow
        (150, 100, 255),  # Link purple
    ]
    
    for frame_idx in range(num_frames):
        # Create base frame (gradient background)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(height):
            intensity = int(30 + (y / height) * 50)
            frame[y, :] = [intensity, intensity + 10, intensity + 20]
        
        # Add UI elements
        for elem_idx in range(num_ui_elements):
            # Random position and size
            x = random.randint(50, width - 150)
            y = random.randint(50, height - 100)
            w = random.randint(60, 120)
            h = random.randint(30, 60)
            
            # Add slight movement over time
            x += int(5 * np.sin(frame_idx * 0.1 + elem_idx))
            y += int(3 * np.cos(frame_idx * 0.15 + elem_idx))
            
            # Ensure bounds
            x = max(10, min(x, width - w - 10))
            y = max(10, min(y, height - h - 10))
            
            color = ui_colors[elem_idx % len(ui_colors)]
            
            # Draw UI element (rectangle with border)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            
            # Add text label
            label = f"UI_{elem_idx + 1}"
            font_scale = 0.6
            thickness = 1
            
            # Get text size
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # Center text in rectangle
            text_x = x + (w - text_w) // 2
            text_y = y + (h + text_h) // 2
            
            cv2.putText(
                frame, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
            )
        
        # Add some noise
        noise = np.random.randint(0, 20, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Save frame
        filename = f"frame_{frame_idx:06d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        frame_files.append(filename)
    
    return frame_files


def create_test_video(
    output_path: str,
    duration_seconds: float = 5.0,
    fps: float = 30.0,
    frame_size: Tuple[int, int] = (800, 600)
) -> str:
    """
    Create a test video with synthetic UI elements.
    
    Args:
        output_path: Path for output video
        duration_seconds: Video duration
        fps: Frames per second
        frame_size: Frame dimensions
        
    Returns:
        Path to created video
    """
    width, height = frame_size
    num_frames = int(duration_seconds * fps)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_idx in range(num_frames):
        # Create frame similar to create_synthetic_ui_frames
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background gradient
        for y in range(height):
            intensity = int(40 + (y / height) * 60)
            frame[y, :] = [intensity, intensity + 15, intensity + 25]
        
        # Moving UI elements
        num_elements = 4
        for elem_idx in range(num_elements):
            # Animated position
            time = frame_idx / fps
            x = int(100 + 200 * np.sin(time * 0.5 + elem_idx * 1.5))
            y = int(100 + 150 * np.cos(time * 0.3 + elem_idx * 1.2))
            
            w, h = 80, 40
            
            # Ensure bounds
            x = max(10, min(x, width - w - 10))
            y = max(10, min(y, height - h - 10))
            
            # Color based on element type
            colors = [(0, 100, 255), (0, 255, 100), (255, 100, 0), (255, 0, 100)]
            color = colors[elem_idx]
            
            # Draw element
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            
            # Label
            label = f"BTN{elem_idx + 1}"
            cv2.putText(
                frame, label, (x + 10, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        out.write(frame)
    
    out.release()
    return output_path


def validate_tracking_results(results_dir: str) -> dict:
    """
    Validate tracking results directory structure and contents.
    
    Args:
        results_dir: Directory containing tracking results
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'valid': True,
        'issues': [],
        'statistics': {}
    }
    
    # Check directory structure
    expected_dirs = ['masks', 'annotations']
    for dir_name in expected_dirs:
        dir_path = os.path.join(results_dir, dir_name)
        if not os.path.exists(dir_path):
            validation['valid'] = False
            validation['issues'].append(f"Missing directory: {dir_name}")
    
    # Check mask files
    masks_dir = os.path.join(results_dir, 'masks')
    if os.path.exists(masks_dir):
        mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.npy')]
        validation['statistics']['mask_files'] = len(mask_files)
        
        # Check if masks can be loaded
        sample_masks = mask_files[:5]  # Check first 5
        for mask_file in sample_masks:
            try:
                mask_path = os.path.join(masks_dir, mask_file)
                mask = np.load(mask_path)
                if mask.dtype not in [np.uint8, np.uint16, np.bool_]:
                    validation['issues'].append(f"Invalid mask dtype: {mask_file}")
            except Exception as e:
                validation['issues'].append(f"Cannot load mask {mask_file}: {e}")
    
    # Check annotation files
    annotations_dir = os.path.join(results_dir, 'annotations')
    if os.path.exists(annotations_dir):
        json_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
        validation['statistics']['annotation_files'] = len(json_files)
        
        # Check if JSON files are valid
        import json
        sample_jsons = json_files[:5]
        for json_file in sample_jsons:
            try:
                json_path = os.path.join(annotations_dir, json_file)
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if 'objects' not in data:
                    validation['issues'].append(f"Missing 'objects' key: {json_file}")
            except Exception as e:
                validation['issues'].append(f"Invalid JSON {json_file}: {e}")
    
    return validation


if __name__ == "__main__":
    print("Creating test data...")
    
    # Create synthetic frames
    print("Creating synthetic UI frames...")
    frame_files = create_synthetic_ui_frames(
        output_dir="./test_frames",
        num_frames=30,
        frame_size=(640, 480),
        num_ui_elements=6
    )
    print(f"Created {len(frame_files)} test frames in ./test_frames")
    
    # Create test video
    print("Creating test video...")
    video_path = create_test_video(
        output_path="./test_video.mp4",
        duration_seconds=3.0,
        fps=10.0,
        frame_size=(640, 480)
    )
    print(f"Created test video: {video_path}")
    
    print("\nTest data creation complete!")
    print("You can now test the UI tracker with:")
    print("  python main.py ./test_frames --output ./test_results")
    print("  python main.py ./test_video.mp4 --output ./test_results")
