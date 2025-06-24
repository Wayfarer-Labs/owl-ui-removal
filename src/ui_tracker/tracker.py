"""
UI Mask Tracker using Grounded-SAM-2
Processes directories of PNG sequences for video UI mask prediction and tracking.
"""

import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm

# SAM2 imports (assuming Grounded-SAM-2 is installed)
try:
    from sam2.build_sam import build_sam2_video_predictor, build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
except ImportError as e:
    logging.warning(f"SAM2 dependencies not found: {e}")
    logging.warning("Please install Grounded-SAM-2 following the official instructions")


class UIElementTracker:
    """
    Tracks UI elements across video frames using Grounded-SAM-2.
    
    This class provides functionality to:
    - Detect UI elements using Grounding DINO
    - Track them across video frames using SAM 2
    - Generate masks for UI elements
    - Save tracking results in various formats
    """
    
    def __init__(
        self,
        sam2_checkpoint: str = "./checkpoints/sam2.1_hiera_large.pt",
        sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        grounding_model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "auto",
        use_bfloat16: bool = True,
    ):
        """
        Initialize the UI Element Tracker.
        
        Args:
            sam2_checkpoint: Path to SAM2 checkpoint
            sam2_config: Path to SAM2 config file
            grounding_model_id: Hugging Face model ID for Grounding DINO
            device: Device to use ('cuda', 'cpu', or 'auto')
            use_bfloat16: Whether to use bfloat16 precision
        """
        self.device = self._setup_device(device)
        self.use_bfloat16 = use_bfloat16
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.grounding_model_id = grounding_model_id
        
        self._setup_precision()
        self._load_models()
        
    def _setup_device(self, device: str) -> str:
        """Setup the computation device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _setup_precision(self):
        """Setup precision and optimization settings."""
        if self.use_bfloat16 and self.device == "cuda":
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            
        if self.device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
            # Enable TF32 for Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
    def _load_models(self):
        """Load SAM2 and Grounding DINO models."""
        try:
            # Load SAM2 models
            self.video_predictor = build_sam2_video_predictor(
                self.sam2_config, 
                self.sam2_checkpoint
            )
            self.sam2_image_model = build_sam2(
                self.sam2_config, 
                self.sam2_checkpoint
            )
            self.image_predictor = SAM2ImagePredictor(self.sam2_image_model)
            
            # Load Grounding DINO
            self.grounding_processor = AutoProcessor.from_pretrained(self.grounding_model_id)
            self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.grounding_model_id
            ).to(self.device)
            
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise
    
    def detect_ui_elements(
        self, 
        image: np.ndarray, 
        text_prompt: str = "button, menu, icon, text field, dropdown, checkbox, slider, tab, dialog, popup, toolbar, navigation bar, status bar, scroll bar",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25
    ) -> Tuple[np.ndarray, List[str], List[float]]:
        """
        Detect UI elements in an image using Grounding DINO.
        
        Args:
            image: Input image as numpy array
            text_prompt: Text description of UI elements to detect
            box_threshold: Box confidence threshold
            text_threshold: Text confidence threshold
            
        Returns:
            Tuple of (bounding_boxes, labels, confidence_scores)
        """
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        # Prepare inputs
        inputs = self.grounding_processor(
            images=image_pil,
            text=text_prompt,
            return_tensors="pt"
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        
        # Post-process results
        results = self.grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image_pil.size[::-1]]
        )
        
        if len(results) > 0 and len(results[0]["boxes"]) > 0:
            boxes = results[0]["boxes"].cpu().numpy()
            labels = results[0]["labels"]
            scores = results[0]["scores"].cpu().numpy().tolist()
            return boxes, labels, scores
        else:
            return np.array([]), [], []
    
    def generate_masks_for_boxes(
        self, 
        image: np.ndarray, 
        boxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate masks for detected bounding boxes using SAM2.
        
        Args:
            image: Input image as numpy array
            boxes: Bounding boxes as numpy array
            
        Returns:
            Tuple of (masks, scores, logits)
        """
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Set image for SAM2 predictor
        self.image_predictor.set_image(image)
        
        # Generate masks
        masks, scores, logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
        
        # Ensure masks have correct shape
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        elif masks.ndim == 3 and len(boxes) == 1:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        
        return masks, scores, logits
    
    def track_ui_elements(
        self,
        frame_dir: str,
        text_prompt: str = "button, menu, icon, text field, dropdown, checkbox, slider, tab, dialog, popup, toolbar, navigation bar, status bar, scroll bar",
        output_dir: str = "./tracking_results",
        prompt_type: str = "mask",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        save_visualizations: bool = True,
        step_size: int = 10,
        max_frames: Optional[int] = None
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Track UI elements across video frames.
        
        Args:
            frame_dir: Directory containing PNG frame sequence
            text_prompt: Text description of UI elements to detect
            output_dir: Directory to save results
            prompt_type: Type of prompt for SAM2 ('mask', 'box', 'point')
            box_threshold: Box confidence threshold
            text_threshold: Text confidence threshold  
            save_visualizations: Whether to save visualization images
            step_size: Frame step size for detection (process every N frames)
            max_frames: Maximum number of frames to process
            
        Returns:
            Dictionary mapping frame_idx -> {object_id: mask}
        """
        # Setup output directories
        os.makedirs(output_dir, exist_ok=True)
        mask_dir = os.path.join(output_dir, "masks")
        json_dir = os.path.join(output_dir, "annotations")
        vis_dir = os.path.join(output_dir, "visualizations")
        
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)
        if save_visualizations:
            os.makedirs(vis_dir, exist_ok=True)
        
        # Get frame files
        frame_files = self._get_frame_files(frame_dir)
        if not frame_files:
            raise ValueError(f"No image files found in {frame_dir}")
        
        # Limit frames if max_frames is specified
        if max_frames is not None:
            frame_files = frame_files[:max_frames]
            self.logger.info(f"Limited to {len(frame_files)} frames (max_frames={max_frames})")
        
        self.logger.info(f"Processing {len(frame_files)} frames")
        
        # Get absolute path for frame directory
        abs_frame_dir = os.path.abspath(frame_dir)
        
        # Initialize video predictor with different approaches
        try:
            # Try with absolute path first
            inference_state = self.video_predictor.init_state(
                video_path=abs_frame_dir,
                offload_video_to_cpu=True,
                async_loading_frames=True
            )
            self.logger.info("Video predictor initialized with async loading")
        except Exception as e:
            self.logger.warning(f"Failed to initialize with async loading: {e}")
            try:
                # Fallback: try without async loading
                inference_state = self.video_predictor.init_state(
                    video_path=abs_frame_dir,
                    offload_video_to_cpu=True,
                    async_loading_frames=False
                )
                self.logger.info("Video predictor initialized without async loading")
            except Exception as e2:
                self.logger.warning(f"Failed to initialize normally: {e2}")
                # Fallback: try with minimal options
                try:
                    inference_state = self.video_predictor.init_state(video_path=abs_frame_dir)
                    self.logger.info("Video predictor initialized with minimal options")
                except Exception as e3:
                    # Final fallback: manual approach without video predictor frame loading
                    self.logger.warning(f"All video predictor init methods failed: {e3}")
                    self.logger.info("Using manual frame processing approach")
                    inference_state = self.video_predictor.init_state(video_path=None)
                    # We'll handle frame loading manually
        
        video_segments = {}
        object_count = 0
        
        # Strategy: Initialize objects in first few frames, then track consistently
        detected_objects = {}  # Store consistent object mappings
        object_templates = {}  # Store object templates for matching
        max_objects = 4  # Expected number of UI objects
        
        # Process frames in chunks
        for i, start_idx in enumerate(tqdm(range(0, len(frame_files), step_size), desc="Processing frames")):
            frame_idx = start_idx
            if frame_idx >= len(frame_files):
                break
                
            frame_path = os.path.join(frame_dir, frame_files[frame_idx])
            image = cv2.imread(frame_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect UI elements
            boxes, labels, scores = self.detect_ui_elements(
                image_rgb, text_prompt, box_threshold, text_threshold
            )
            
            if len(boxes) == 0:
                self.logger.info(f"No UI elements detected in frame {frame_idx}")
                # For tracking consistency, still propagate existing objects
                if detected_objects:
                    self._propagate_existing_objects(inference_state, frame_idx, video_segments)
                continue
            
            # If this is early in the sequence and we haven't found all objects yet
            if len(detected_objects) < max_objects and i < 10:  # Search in first 10 detection frames
                self._initialize_or_update_objects(
                    inference_state, frame_idx, image_rgb, boxes, labels, scores,
                    detected_objects, object_templates, prompt_type
                )
            else:
                # We have our core objects, now just track them
                self._track_existing_objects(
                    inference_state, frame_idx, image_rgb, boxes, labels, scores,
                    detected_objects, object_templates, prompt_type
                )
            
            # Propagate through video frames for all detected objects
            chunk_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
                inference_state, 
                max_frame_num_to_track=step_size, 
                start_frame_idx=frame_idx
            ):
                frame_masks = {}
                for i, out_obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                    frame_masks[out_obj_id] = mask[0] if mask.ndim > 2 else mask
                
                chunk_segments[out_frame_idx] = frame_masks
                
            video_segments.update(chunk_segments)
                
            video_segments.update(chunk_segments)
        
        # Apply temporal filtering to improve consistency
        # video_segments = self._temporal_filter_masks(video_segments, min_frames=2)
        
        # Save results
        self._save_tracking_results(
            video_segments, frame_files, mask_dir, json_dir, 
            vis_dir if save_visualizations else None, frame_dir
        )
        
        return video_segments
    
    def _get_frame_files(self, frame_dir: str) -> List[str]:
        """Get sorted list of image files in directory."""
        extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        files = []
        for ext in extensions:
            files.extend([f for f in os.listdir(frame_dir) if f.endswith(ext)])
        
        if not files:
            return files
        
        # Sort by numeric value if possible, otherwise alphabetically
        def extract_number(filename):
            import re
            # Try to extract number from filename (e.g., frame_00001.png -> 1)
            match = re.search(r'(\d+)', filename)
            if match:
                return int(match.group(1))
            return 0
        
        try:
            # Try numeric sorting first
            files.sort(key=extract_number)
        except (ValueError, TypeError):
            # Fallback to alphabetical sorting
            files.sort()
        
        return files
    
    def _save_tracking_results(
        self,
        video_segments: Dict[int, Dict[int, np.ndarray]],
        frame_files: List[str],
        mask_dir: str,
        json_dir: str,
        vis_dir: Optional[str] = None,
        frame_dir: Optional[str] = None
    ):
        """Save tracking results to disk."""
        from utils.visualization import create_multi_mask_overlay
        import cv2
        
        for frame_idx, frame_masks in video_segments.items():
            if frame_idx >= len(frame_files):
                continue
                
            frame_name = os.path.splitext(frame_files[frame_idx])[0]
            
            # Save individual masks
            for obj_id, mask in frame_masks.items():
                mask_path = os.path.join(mask_dir, f"{frame_name}_obj_{obj_id}.npy")
                np.save(mask_path, mask.astype(np.uint8))
            
            # Save combined mask with object IDs
            if frame_masks:
                h, w = next(iter(frame_masks.values())).shape[:2]
                combined_mask = np.zeros((h, w), dtype=np.uint16)
                
                annotations = {
                    "frame": frame_name,
                    "objects": {}
                }
                
                for obj_id, mask in frame_masks.items():
                    # Ensure mask is 2D
                    if mask.ndim > 2:
                        mask = mask.squeeze()
                    
                    combined_mask[mask > 0.5] = obj_id
                    
                    # Calculate bounding box
                    y_coords, x_coords = np.where(mask > 0.5)
                    if len(y_coords) > 0:
                        bbox = [
                            int(x_coords.min()), int(y_coords.min()),
                            int(x_coords.max()), int(y_coords.max())
                        ]
                    else:
                        bbox = [0, 0, 0, 0]
                    
                    annotations["objects"][str(obj_id)] = {
                        "bbox": bbox,
                        "area": int(np.sum(mask > 0.5)),
                        "label": "UI_element"
                    }
                
                # Save combined mask
                combined_path = os.path.join(mask_dir, f"{frame_name}_combined.npy")
                np.save(combined_path, combined_mask)
                
                # Save annotations
                json_path = os.path.join(json_dir, f"{frame_name}.json")
                with open(json_path, 'w') as f:
                    json.dump(annotations, f, indent=2)
                
                # Create and save visualization if vis_dir is provided
                if vis_dir is not None and frame_dir is not None:
                    try:
                        # Load original frame
                        frame_path = os.path.join(frame_dir, frame_files[frame_idx])
                        original_frame = cv2.imread(frame_path)
                        if original_frame is not None:
                            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                            
                            # Create mask overlay visualization
                            vis_image = create_multi_mask_overlay(
                                image=original_frame,
                                masks=frame_masks,
                                alpha=0.4
                            )
                            
                            # Add bounding boxes and labels
                            for obj_id, obj_data in annotations["objects"].items():
                                bbox = obj_data["bbox"]
                                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Valid bbox
                                    # Draw bounding box
                                    cv2.rectangle(vis_image, 
                                                (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                                                (255, 255, 255), 2)
                                    
                                    # Add label
                                    label = f"UI_{obj_id}"
                                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                    cv2.rectangle(vis_image,
                                                (bbox[0], bbox[1] - label_size[1] - 10),
                                                (bbox[0] + label_size[0], bbox[1]),
                                                (255, 255, 255), -1)
                                    cv2.putText(vis_image, label, 
                                              (bbox[0], bbox[1] - 5),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                            
                            # Save visualization
                            vis_path = os.path.join(vis_dir, f"{frame_name}_overlay.png")
                            vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(vis_path, vis_image_bgr)
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to create visualization for frame {frame_idx}: {e}")
        
        self.logger.info(f"Saved tracking results for {len(video_segments)} frames")
    
    def visualize_tracking_results(
        self,
        frame_dir: str,
        results_dir: str,
        output_dir: str = "./visualizations"
    ):
        """
        Create visualization of tracking results.
        
        Args:
            frame_dir: Directory containing original frames
            results_dir: Directory containing tracking results
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        mask_dir = os.path.join(results_dir, "masks")
        json_dir = os.path.join(results_dir, "annotations")
        
        frame_files = self._get_frame_files(frame_dir)
        
        for frame_file in tqdm(frame_files, desc="Creating visualizations"):
            frame_name = os.path.splitext(frame_file)[0]
            
            # Load original frame
            frame_path = os.path.join(frame_dir, frame_file)
            image = cv2.imread(frame_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load combined mask
            mask_path = os.path.join(mask_dir, f"{frame_name}_combined.npy")
            if not os.path.exists(mask_path):
                continue
                
            mask = np.load(mask_path)
            
            # Load annotations
            json_path = os.path.join(json_dir, f"{frame_name}.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    annotations = json.load(f)
            else:
                annotations = {"objects": {}}
            
            # Create visualization
            vis_image = self._create_mask_visualization(image_rgb, mask, annotations)
            
            # Save visualization
            vis_path = os.path.join(output_dir, f"{frame_name}_tracked.png")
            cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    def _create_mask_visualization(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        annotations: Dict
    ) -> np.ndarray:
        """Create a visualization of masks overlaid on the image."""
        vis_image = image.copy()
        
        # Generate colors for each object
        unique_ids = np.unique(mask)
        unique_ids = unique_ids[unique_ids > 0]  # Remove background
        
        colors = np.random.randint(0, 255, (len(unique_ids), 3))
        
        for i, obj_id in enumerate(unique_ids):
            obj_mask = (mask == obj_id).astype(np.uint8)
            color = colors[i]
            
            # Create colored overlay
            colored_mask = np.zeros_like(vis_image)
            colored_mask[obj_mask > 0] = color
            
            # Blend with original image
            vis_image = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)
            
            # Draw bounding box if available
            if str(obj_id) in annotations.get("objects", {}):
                bbox = annotations["objects"][str(obj_id)]["bbox"]
                cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color.tolist(), 2)
                
                # Add label
                label = f"UI_{obj_id}"
                cv2.putText(vis_image, label, (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.tolist(), 2)
        
        return vis_image
    
    def _match_objects_with_tracking(
        self, 
        boxes: np.ndarray, 
        labels: List[str], 
        scores: np.ndarray,
        tracking_objects: Dict[int, Dict],
        current_frame: int,
        iou_threshold: float = 0.5,
        max_frame_gap: int = 30
    ) -> Tuple[Dict[int, Tuple], Dict[int, Tuple]]:
        """
        Match detected boxes with existing tracked objects using IoU overlap.
        
        Args:
            boxes: Detected bounding boxes in current frame
            labels: Detected labels
            scores: Detection scores  
            tracking_objects: Currently tracked objects
            current_frame: Current frame index
            iou_threshold: Minimum IoU for matching
            max_frame_gap: Maximum frames since last detection
            
        Returns:
            Tuple of (matched_objects, new_objects)
        """
        def calculate_iou(box1, box2):
            """Calculate IoU between two boxes [x1, y1, x2, y2]"""
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
                
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        matched_objects = {}
        used_boxes = set();
        
        # First, try to match with recently active objects
        active_objects = {
            obj_id: obj_data for obj_id, obj_data in tracking_objects.items()
            if obj_data['active'] and (current_frame - obj_data['last_frame']) <= max_frame_gap
        }
        
        # Match boxes with active tracked objects
        for obj_id, obj_data in active_objects.items():
            last_box = obj_data['last_box']
            best_iou = 0.0
            best_box_idx = -1
            
            for box_idx, box in enumerate(boxes):
                if box_idx in used_boxes:
                    continue;
                    
                iou = calculate_iou(last_box, box)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_box_idx = box_idx
            
            if best_box_idx >= 0:
                matched_objects[obj_id] = (
                    best_box_idx, boxes[best_box_idx], 
                    labels[best_box_idx], scores[best_box_idx]
                )
                used_boxes.add(best_box_idx)
        
        # Mark unmatched objects as inactive
        for obj_id in active_objects:
            if obj_id not in matched_objects:
                tracking_objects[obj_id]['active'] = False
        
        # Collect unmatched boxes as new objects
        new_objects = {}
        for box_idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            if box_idx not in used_boxes:
                new_objects[box_idx] = (box, label, score)
        
        return matched_objects, new_objects
    
    def _temporal_filter_masks(
        self, 
        video_segments: Dict[int, Dict[int, np.ndarray]], 
        min_frames: int = 3
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Filter out objects that appear in too few frames to reduce noise.
        
        Args:
            video_segments: Raw tracking results
            min_frames: Minimum number of frames an object must appear in
            
        Returns:
            Filtered tracking results
        """
        # Count frames per object
        object_frame_counts = {}
        for frame_idx, frame_masks in video_segments.items():
            for obj_id in frame_masks.keys():
                object_frame_counts[obj_id] = object_frame_counts.get(obj_id, 0) + 1
        
        # Filter objects that appear too infrequently
        valid_objects = {
            obj_id for obj_id, count in object_frame_counts.items() 
            if count >= min_frames
        }
        
        # Remove invalid objects from results
        filtered_segments = {}
        for frame_idx, frame_masks in video_segments.items():
            filtered_masks = {
                obj_id: mask for obj_id, mask in frame_masks.items()
                if obj_id in valid_objects
            }
            if filtered_masks:  # Only include frames with valid objects
                filtered_segments[frame_idx] = filtered_masks
        
        removed_count = len(object_frame_counts) - len(valid_objects)
        if removed_count > 0:
            self.logger.info(f"Filtered out {removed_count} objects that appeared in fewer than {min_frames} frames")
        
        return filtered_segments

    def _calculate_box_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _match_detections_to_objects(self, boxes, detected_objects, iou_threshold=0.3):
        """Match current detections to existing objects."""
        matches = {}
        unmatched_detections = list(range(len(boxes)))
        
        for obj_id, obj_data in detected_objects.items():
            best_match = -1
            best_iou = 0
            
            for det_idx in unmatched_detections:
                iou = self._calculate_box_iou(boxes[det_idx], obj_data['box'])
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_match = det_idx
            
            if best_match >= 0:
                matches[obj_id] = best_match
                unmatched_detections.remove(best_match)
        
        return matches, unmatched_detections
    
    def _initialize_or_update_objects(self, inference_state, frame_idx, image, boxes, labels, scores, 
                                    detected_objects, object_templates, prompt_type):
        """Initialize new objects or update existing ones in early frames."""
        # Generate masks for all detections
        masks, mask_scores, logits = self.generate_masks_for_boxes(image, boxes)
        
        if len(masks) == 0:
            return
        
        # Match to existing objects
        matches, unmatched = self._match_detections_to_objects(boxes, detected_objects)
        
        # Update existing objects
        for obj_id, det_idx in matches.items():
            # Update the object's position but keep the same SAM2 tracking
            detected_objects[obj_id]['box'] = boxes[det_idx]
            detected_objects[obj_id]['last_seen'] = frame_idx
        
        # Add new objects for unmatched detections
        for det_idx in unmatched:
            if len(detected_objects) >= 4:  # Limit to 4 objects as specified
                break
                
            new_obj_id = len(detected_objects) + 1
            mask = masks[det_idx]
            
            # Ensure mask is 2D
            if mask.ndim > 2:
                mask = mask.squeeze()
            if mask.ndim != 2:
                continue
            
            # Add to SAM2 video predictor
            try:
                if prompt_type == "mask":
                    self.video_predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=new_obj_id,
                        mask=mask
                    )
                elif prompt_type == "box":
                    self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=new_obj_id,
                        box=boxes[det_idx]
                    )
                
                # Store object information
                detected_objects[new_obj_id] = {
                    'box': boxes[det_idx],
                    'label': labels[det_idx],
                    'score': scores[det_idx],
                    'first_seen': frame_idx,
                    'last_seen': frame_idx
                }
                
                self.logger.info(f"Initialized new UI object {new_obj_id} at frame {frame_idx}")
                
            except Exception as e:
                self.logger.warning(f"Failed to add new object {new_obj_id}: {e}")
    
    def _track_existing_objects(self, inference_state, frame_idx, image, boxes, labels, scores,
                              detected_objects, object_templates, prompt_type):
        """Track existing objects, updating positions for moving objects."""
        # Match detections to existing objects
        matches, unmatched = self._match_detections_to_objects(boxes, detected_objects)
        
        # Update matched objects (especially important for moving objects)
        for obj_id, det_idx in matches.items():
            old_box = detected_objects[obj_id]['box']
            new_box = boxes[det_idx]
            
            # Check if object moved significantly (could be the moving object)
            iou = self._calculate_box_iou(old_box, new_box)
            if iou < 0.8:  # Object moved significantly
                self.logger.debug(f"Object {obj_id} moved significantly (IoU: {iou:.3f})")
                
                # Generate new mask for moved object
                masks, _, _ = self.generate_masks_for_boxes(image, [new_box])
                if len(masks) > 0:
                    mask = masks[0]
                    if mask.ndim > 2:
                        mask = mask.squeeze()
                    if mask.ndim == 2:
                        try:
                            # Add updated mask position to SAM2
                            self.video_predictor.add_new_mask(
                                inference_state=inference_state,
                                frame_idx=frame_idx,
                                obj_id=obj_id,
                                mask=mask
                            )
                        except Exception as e:
                            self.logger.warning(f"Failed to update object {obj_id}: {e}")
            
            # Update object data
            detected_objects[obj_id]['box'] = new_box
            detected_objects[obj_id]['last_seen'] = frame_idx
    
    def _propagate_existing_objects(self, inference_state, frame_idx, video_segments):
        """Propagate existing objects when no new detections are found."""
        try:
            # Just propagate existing objects without adding new ones
            chunk_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
                inference_state, 
                max_frame_num_to_track=1, 
                start_frame_idx=frame_idx
            ):
                if out_frame_idx == frame_idx:  # Only current frame
                    frame_masks = {}
                    for i, out_obj_id in enumerate(out_obj_ids):
                        mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                        frame_masks[out_obj_id] = mask[0] if mask.ndim > 2 else mask
                    
                    video_segments[out_frame_idx] = frame_masks
                    break
        except Exception as e:
            self.logger.warning(f"Failed to propagate objects at frame {frame_idx}: {e}")

def create_ui_tracker(
    sam2_checkpoint: Optional[str] = None,
    sam2_config: Optional[str] = None,
    **kwargs
) -> UIElementTracker:
    """
    Factory function to create a UI tracker with default settings.
    
    Args:
        sam2_checkpoint: Path to SAM2 checkpoint (auto-download if None)
        sam2_config: Path to SAM2 config (use default if None)
        **kwargs: Additional arguments for UIElementTracker
        
    Returns:
        Configured UIElementTracker instance
    """
    # Set default paths if not provided
    if sam2_checkpoint is None:
        sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    if sam2_config is None:
        sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    return UIElementTracker(
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
        **kwargs
    )
