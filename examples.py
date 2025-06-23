#!/usr/bin/env python3
"""
Example script showing how to use the UI tracker programmatically.
"""

import os
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ui_tracker.tracker import UIElementTracker
from ui_tracker.config import get_mobile_ui_config, get_web_ui_config
from utils.video_utils import extract_frames_from_video


def example_mobile_app_tracking():
    """Example: Track UI elements in a mobile app screen recording."""
    print("Example: Mobile App UI Tracking")
    print("=" * 40)
    
    # Setup paths (adjust these to your actual files)
    video_path = "mobile_app_demo.mp4"  # Your input video
    frames_dir = "./temp_frames"
    output_dir = "./mobile_results"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        print("Please provide a mobile app screen recording video.")
        return
    
    try:
        # Step 1: Extract frames from video
        print("Extracting frames...")
        extract_frames_from_video(
            video_path=video_path,
            output_dir=frames_dir,
            frame_rate=2.0,  # 2 FPS
            max_frames=50    # Limit for demo
        )
        
        # Step 2: Initialize tracker with mobile preset
        print("Initializing UI tracker...")
        config = get_mobile_ui_config()
        
        tracker = UIElementTracker(
            sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
            sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml",
            device="auto"
        )
        
        # Step 3: Track UI elements
        print("Tracking UI elements...")
        results = tracker.track_ui_elements(
            frame_dir=frames_dir,
            text_prompt="button, icon, tab, menu, search bar, navigation bar",
            output_dir=output_dir,
            step_size=5,
            save_visualizations=True
        )
        
        # Step 4: Print results summary
        total_objects = sum(len(frame_masks) for frame_masks in results.values())
        print(f"\nResults:")
        print(f"- Processed {len(results)} frames")
        print(f"- Detected {total_objects} UI objects")
        print(f"- Results saved to: {output_dir}")
        
        # Clean up temp frames
        import shutil
        shutil.rmtree(frames_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"Error: {e}")
        logging.exception("Tracking failed")


def example_web_ui_tracking():
    """Example: Track UI elements in web browser recording."""
    print("Example: Web UI Tracking")
    print("=" * 40)
    
    # Setup paths
    frames_dir = "./web_frames"  # Directory containing PNG frames
    output_dir = "./web_results"
    
    # Check if frames directory exists
    if not os.path.exists(frames_dir):
        print(f"Frames directory not found: {frames_dir}")
        print("Please provide a directory with PNG frame sequence.")
        return
    
    try:
        # Initialize tracker with web UI preset
        print("Initializing UI tracker...")
        config = get_web_ui_config()
        
        tracker = UIElementTracker(
            sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
            sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml",
            device="auto"
        )
        
        # Track UI elements
        print("Tracking UI elements...")
        results = tracker.track_ui_elements(
            frame_dir=frames_dir,
            text_prompt="button, link, menu, form, header, navigation, sidebar",
            output_dir=output_dir,
            step_size=10,
            save_visualizations=True
        )
        
        # Print results
        total_objects = sum(len(frame_masks) for frame_masks in results.values())
        print(f"\nResults:")
        print(f"- Processed {len(results)} frames")
        print(f"- Detected {total_objects} UI objects")
        print(f"- Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        logging.exception("Tracking failed")


def example_custom_tracking():
    """Example: Custom UI tracking with specific parameters."""
    print("Example: Custom UI Tracking")
    print("=" * 40)
    
    frames_dir = "./custom_frames"
    output_dir = "./custom_results"
    
    if not os.path.exists(frames_dir):
        print(f"Frames directory not found: {frames_dir}")
        return
    
    try:
        # Initialize tracker with custom settings
        tracker = UIElementTracker(
            sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
            sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml",
            grounding_model_id="IDEA-Research/grounding-dino-tiny",
            device="cuda",  # Force CUDA
            use_bfloat16=True
        )
        
        # Custom tracking with specific prompt
        results = tracker.track_ui_elements(
            frame_dir=frames_dir,
            text_prompt="close button, minimize button, maximize button, title bar, menu bar",
            output_dir=output_dir,
            prompt_type="mask",
            box_threshold=0.2,
            text_threshold=0.2,
            step_size=8,
            save_visualizations=True
        )
        
        print(f"Custom tracking completed: {len(results)} frames processed")
        
    except Exception as e:
        print(f"Error: {e}")
        logging.exception("Custom tracking failed")


def main():
    """Run examples based on available input data."""
    logging.basicConfig(level=logging.INFO)
    
    print("UI Tracker Examples")
    print("=" * 50)
    
    print("\nAvailable examples:")
    print("1. Mobile App UI Tracking")
    print("2. Web UI Tracking")
    print("3. Custom UI Tracking")
    
    choice = input("\nSelect example (1-3, or 'all'): ").strip()
    
    if choice == '1' or choice == 'all':
        example_mobile_app_tracking()
        print()
    
    if choice == '2' or choice == 'all':
        example_web_ui_tracking()
        print()
    
    if choice == '3' or choice == 'all':
        example_custom_tracking()
        print()
    
    if choice not in ['1', '2', '3', 'all']:
        print("Invalid choice")


if __name__ == "__main__":
    main()
