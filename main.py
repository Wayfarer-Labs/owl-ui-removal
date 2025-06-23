#!/usr/bin/env python3
"""
UI Mask Tracker using Grounded-SAM-2
Main entry point for tracking UI elements in video frame sequences.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ui_tracker.tracker import UIElementTracker, create_ui_tracker
from ui_tracker.config import (
    UITrackerConfig, load_config, create_default_config,
    get_ui_detection_config, get_mobile_ui_config, 
    get_web_ui_config, get_game_ui_config
)
from utils.video_utils import (
    extract_frames_from_video, create_video_from_frames,
    validate_frame_sequence, get_frame_info
)
from utils.visualization import export_masks_to_coco_format


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def validate_inputs(args):
    """Validate input arguments."""
    if not os.path.exists(args.input):
        raise ValueError(f"Input path does not exist: {args.input}")
    
    # Check if input is video file or frame directory
    if os.path.isfile(args.input):
        # Video file - check if frames directory is specified
        if not args.frames_dir:
            args.frames_dir = os.path.join(os.path.dirname(args.input), "extracted_frames")
            logging.info(f"Frames will be extracted to: {args.frames_dir}")
    elif os.path.isdir(args.input):
        # Frame directory
        is_valid, issues = validate_frame_sequence(args.input)
        if not is_valid:
            logging.warning("Frame sequence validation issues:")
            for issue in issues:
                logging.warning(f"  - {issue}")
            
            if not args.force:
                raise ValueError("Frame sequence validation failed. Use --force to proceed anyway.")
    else:
        raise ValueError(f"Input must be a video file or directory: {args.input}")


def extract_frames_if_needed(args) -> str:
    """Extract frames from video if input is a video file."""
    if os.path.isfile(args.input):
        logging.info(f"Extracting frames from video: {args.input}")
        
        frame_files = extract_frames_from_video(
            video_path=args.input,
            output_dir=args.frames_dir,
            frame_rate=args.extract_fps,
            start_time=args.start_time,
            end_time=args.end_time,
            max_frames=args.max_frames
        )
        
        logging.info(f"Extracted {len(frame_files)} frames")
        return args.frames_dir
    else:
        return args.input


def load_or_create_config(args) -> UITrackerConfig:
    """Load configuration from file or create based on preset."""
    if args.config:
        logging.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
    elif args.preset:
        logging.info(f"Using preset configuration: {args.preset}")
        presets = {
            'default': create_default_config,
            'ui': get_ui_detection_config,
            'mobile': get_mobile_ui_config,
            'web': get_web_ui_config,
            'game': get_game_ui_config
        }
        config = presets[args.preset]()
    else:
        logging.info("Using default configuration")
        config = create_default_config()
    
    # Override config with command line arguments
    if args.output:
        config.output.output_dir = args.output
    if args.text_prompt:
        config.detection.text_prompt = args.text_prompt
    if args.box_threshold is not None:
        config.detection.box_threshold = args.box_threshold
    if args.text_threshold is not None:
        config.detection.text_threshold = args.text_threshold
    if args.step_size is not None:
        config.tracking.step_size = args.step_size
    if args.device:
        config.model.device = args.device
    if args.no_visualizations:
        config.tracking.save_visualizations = False
    if args.export_coco:
        config.output.export_coco = True
    if args.export_video:
        config.output.export_video = True
    
    return config


def run_tracking(frame_dir: str, config: UITrackerConfig, max_frames: Optional[int] = None) -> dict:
    """Run UI element tracking."""
    logging.info("Initializing UI tracker...")
    
    try:
        tracker = UIElementTracker(
            sam2_checkpoint=config.model.sam2_checkpoint,
            sam2_config=config.model.sam2_config,
            grounding_model_id=config.model.grounding_model_id,
            device=config.model.device,
            use_bfloat16=config.model.use_bfloat16
        )
    except Exception as e:
        logging.error(f"Failed to initialize tracker: {e}")
        logging.error("Make sure Grounded-SAM-2 is properly installed and checkpoints are available")
        raise
    
    logging.info("Starting UI element tracking...")
    
    results = tracker.track_ui_elements(
        frame_dir=frame_dir,
        text_prompt=config.detection.text_prompt,
        output_dir=config.output.output_dir,
        prompt_type=config.detection.prompt_type,
        box_threshold=config.detection.box_threshold,
        text_threshold=config.detection.text_threshold,
        save_visualizations=config.tracking.save_visualizations,
        step_size=config.tracking.step_size,
        max_frames=max_frames
    )
    
    return results


def post_process_results(results: dict, frame_dir: str, config: UITrackerConfig):
    """Post-process tracking results."""
    frame_files = []
    for ext in ['.png', '.jpg', '.jpeg']:
        frame_files.extend([f for f in os.listdir(frame_dir) if f.lower().endswith(ext)])
    frame_files.sort()
    
    # Export to COCO format if requested
    if config.output.export_coco:
        logging.info("Exporting results to COCO format...")
        coco_file = os.path.join(config.output.output_dir, "annotations.json")
        
        export_masks_to_coco_format(
            masks=results,
            frame_files=frame_files,
            output_file=coco_file,
            image_dir=frame_dir
        )
        logging.info(f"COCO annotations saved to: {coco_file}")
    
    # Create output video if requested
    if config.output.export_video:
        logging.info("Creating output video...")
        vis_dir = os.path.join(config.output.output_dir, "visualizations")
        
        if os.path.exists(vis_dir):
            output_video = os.path.join(config.output.output_dir, "tracking_result.mp4")
            create_video_from_frames(
                frames_dir=vis_dir,
                output_path=output_video,
                fps=config.output.video_fps,
                frame_pattern="*_tracked.png"
            )
            logging.info(f"Output video saved to: {output_video}")
        else:
            logging.warning("Visualizations not available for video export")


def print_summary(results: dict, frame_dir: str, config: UITrackerConfig):
    """Print summary of tracking results."""
    frame_info = get_frame_info(frame_dir)
    
    total_objects = 0
    tracked_frames = len(results)
    
    for frame_masks in results.values():
        total_objects += len(frame_masks)
    
    print("\n" + "="*60)
    print("TRACKING SUMMARY")
    print("="*60)
    print(f"Input frames: {frame_info['frame_count']}")
    print(f"Processed frames: {tracked_frames}")
    print(f"Total UI objects detected: {total_objects}")
    print(f"Average objects per frame: {total_objects/tracked_frames if tracked_frames > 0 else 0:.1f}")
    print(f"Output directory: {config.output.output_dir}")
    print(f"Frame dimensions: {frame_info.get('dimensions', 'Unknown')}")
    print(f"Detection prompt: {config.detection.text_prompt[:100]}...")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Track UI elements in video sequences using Grounded-SAM-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Track UI elements in video file
  python main.py input_video.mp4 --output ./results

  # Track UI elements in frame directory
  python main.py ./frames --output ./results

  # Use mobile UI preset with custom prompt
  python main.py ./frames --preset mobile --text-prompt "button, icon, menu"

  # Extract frames and track with custom settings
  python main.py video.mp4 --extract-fps 2 --step-size 5 --export-video
        """
    )
    
    # Input/Output arguments
    parser.add_argument('input', help='Input video file or directory containing frame sequence')
    parser.add_argument('-o', '--output', default='./results', help='Output directory (default: ./results)')
    parser.add_argument('--frames-dir', help='Directory to extract/find frames (auto-generated for videos)')
    
    # Configuration arguments
    parser.add_argument('-c', '--config', help='Configuration file (YAML or JSON)')
    parser.add_argument('--preset', choices=['default', 'ui', 'mobile', 'web', 'game'],
                       help='Use predefined configuration preset')
    
    # Detection parameters
    parser.add_argument('--text-prompt', help='Text prompt for UI element detection')
    parser.add_argument('--box-threshold', type=float, help='Box confidence threshold')
    parser.add_argument('--text-threshold', type=float, help='Text confidence threshold')
    
    # Tracking parameters
    parser.add_argument('--step-size', type=int, help='Frame step size for detection')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], help='Device to use')
    
    # Frame extraction (for video inputs)
    parser.add_argument('--extract-fps', type=float, help='FPS for frame extraction')
    parser.add_argument('--start-time', type=float, default=0.0, help='Start time in seconds')
    parser.add_argument('--end-time', type=float, help='End time in seconds')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to extract')
    
    # Output options
    parser.add_argument('--no-visualizations', action='store_true', help='Skip visualization generation')
    parser.add_argument('--export-coco', action='store_true', help='Export annotations in COCO format')
    parser.add_argument('--export-video', action='store_true', help='Create output video with tracking results')
    
    # Other options
    parser.add_argument('--force', action='store_true', help='Force processing despite validation warnings')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Validate inputs
        validate_inputs(args)
        
        # Extract frames if needed
        frame_dir = extract_frames_if_needed(args)
        
        # Load or create configuration
        config = load_or_create_config(args)
        
        # Save configuration for reference
        config_path = os.path.join(config.output.output_dir, "config_used.yaml")
        os.makedirs(config.output.output_dir, exist_ok=True)
        config.save_yaml(config_path)
        logging.info(f"Configuration saved to: {config_path}")
        
        # Run tracking
        results = run_tracking(frame_dir, config, args.max_frames)
        
        # Post-process results
        post_process_results(results, frame_dir, config)
        
        # Print summary
        print_summary(results, frame_dir, config)
        
        logging.info("UI tracking completed successfully!")
        
    except KeyboardInterrupt:
        logging.info("Tracking interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error during tracking: {e}")
        logging.error(f"Exception type: [{type(e).__name__}]")
        import traceback
        logging.error(f"Full traceback:\n{traceback.format_exc()}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
