"""
Utility functions for visualization and mask processing.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import os
from typing import List, Dict, Tuple, Optional, Any
import json


def create_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """
    Create an overlay of mask on image.
    
    Args:
        image: Base image (H, W, 3)
        mask: Binary mask (H, W)
        alpha: Transparency factor
        color: Color for mask overlay (R, G, B)
        
    Returns:
        Image with mask overlay
    """
    overlay = image.copy()
    
    # Ensure mask is binary
    mask_binary = (mask > 0.5).astype(np.uint8)
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask_binary == 1] = color
    
    # Blend with original image
    result = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
    
    return result


def create_multi_mask_overlay(
    image: np.ndarray,
    masks: Dict[int, np.ndarray],
    alpha: float = 0.4,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Create overlay with multiple masks using different colors.
    
    Args:
        image: Base image (H, W, 3)
        masks: Dictionary mapping object_id -> mask
        alpha: Transparency factor
        colors: Optional color mapping for each object_id
        
    Returns:
        Image with multi-mask overlay
    """
    if not masks:
        return image.copy()
    
    overlay = image.copy().astype(np.float32)
    
    # Generate colors if not provided
    if colors is None:
        colors = {}
        color_palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 0)
        ]
        for i, obj_id in enumerate(masks.keys()):
            colors[obj_id] = color_palette[i % len(color_palette)]
    
    # Create a combined mask overlay
    combined_overlay = np.zeros_like(image, dtype=np.float32)
    combined_mask = np.zeros(image.shape[:2], dtype=np.float32)
    
    # Apply each mask to the combined overlay
    for obj_id, mask in masks.items():
        color = colors.get(obj_id, (255, 255, 255))
        
        # Ensure mask is binary and normalize
        mask_binary = (mask > 0.5).astype(np.float32)
        
        if np.any(mask_binary):
            # Add colored mask to combined overlay
            for c in range(3):
                combined_overlay[:, :, c] += mask_binary * color[c]
            combined_mask = np.maximum(combined_mask, mask_binary)
    
    # Normalize the combined overlay where masks overlap
    mask_counts = np.zeros_like(combined_mask)
    for mask in masks.values():
        mask_binary = (mask > 0.5).astype(np.float32)
        mask_counts += mask_binary
    
    # Avoid division by zero
    mask_counts = np.maximum(mask_counts, 1.0)
    
    # Normalize overlapping areas
    for c in range(3):
        combined_overlay[:, :, c] = np.where(
            combined_mask > 0,
            combined_overlay[:, :, c] / mask_counts,
            0
        )
    
    # Blend with original image only once
    result = np.where(
        combined_mask[..., np.newaxis] > 0,
        overlay * (1 - alpha) + combined_overlay * alpha,
        overlay
    )
    
    return np.clip(result, 0, 255).astype(np.uint8)


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: List[List[int]],
    labels: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Input image
        boxes: List of bounding boxes [x1, y1, x2, y2]
        labels: Optional labels for each box
        colors: Optional colors for each box
        thickness: Line thickness
        
    Returns:
        Image with bounding boxes drawn
    """
    result = image.copy()
    
    if colors is None:
        colors = [(0, 255, 0)] * len(boxes)  # Default green
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        color = colors[i % len(colors)]
        
        # Draw rectangle
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label if provided
        if labels and i < len(labels):
            label = labels[i]
            font_scale = 0.6
            font_thickness = 1
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                result,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                result, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness
            )
    
    return result


def create_mask_grid_visualization(
    masks: Dict[int, np.ndarray],
    titles: Optional[Dict[int, str]] = None,
    grid_size: Optional[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Create a grid visualization of multiple masks.
    
    Args:
        masks: Dictionary mapping object_id -> mask
        titles: Optional titles for each mask
        grid_size: Optional grid size (rows, cols)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_masks = len(masks)
    
    if grid_size is None:
        # Calculate grid size
        cols = int(np.ceil(np.sqrt(n_masks)))
        rows = int(np.ceil(n_masks / cols))
    else:
        rows, cols = grid_size
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single subplot case
    if n_masks == 1:
        axes = [axes]
    elif rows * cols > 1:
        axes = axes.flatten()
    
    mask_items = list(masks.items())
    
    for i in range(rows * cols):
        ax = axes[i] if isinstance(axes, list) else axes
        
        if i < len(mask_items):
            obj_id, mask = mask_items[i]
            
            # Display mask
            ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
            
            # Set title
            if titles and obj_id in titles:
                title = titles[obj_id]
            else:
                title = f'Object {obj_id}'
            
            ax.set_title(title)
            ax.axis('off')
        else:
            # Hide empty subplots
            ax.axis('off')
    
    plt.tight_layout()
    return fig


def save_masks_as_images(
    masks: Dict[int, np.ndarray],
    output_dir: str,
    prefix: str = "mask",
    format: str = "png"
) -> List[str]:
    """
    Save individual masks as image files.
    
    Args:
        masks: Dictionary mapping object_id -> mask
        output_dir: Output directory
        prefix: Filename prefix
        format: Image format
        
    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    for obj_id, mask in masks.items():
        # Convert mask to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Save mask
        filename = f"{prefix}_obj_{obj_id}.{format}"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, mask_uint8)
        
        saved_files.append(filepath)
    
    return saved_files


def create_comparison_visualization(
    images: List[np.ndarray],
    titles: List[str],
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Create side-by-side comparison of images.
    
    Args:
        images: List of images to compare
        titles: List of titles for each image
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    if n_images == 1:
        axes = [axes]
    
    for i, (image, title) in enumerate(zip(images, titles)):
        axes[i].imshow(image)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def mask_to_polygon(mask: np.ndarray, tolerance: float = 1.0) -> List[List[Tuple[int, int]]]:
    """
    Convert binary mask to polygon coordinates.
    
    Args:
        mask: Binary mask (H, W)
        tolerance: Tolerance for polygon approximation
        
    Returns:
        List of polygons, each as list of (x, y) coordinates
    """
    # Find contours
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        # Approximate polygon
        epsilon = tolerance * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to list of (x, y) tuples
        polygon = [(int(point[0][0]), int(point[0][1])) for point in approx]
        polygons.append(polygon)
    
    return polygons


def calculate_mask_statistics(masks: Dict[int, np.ndarray]) -> Dict[int, Dict[str, Any]]:
    """
    Calculate statistics for each mask.
    
    Args:
        masks: Dictionary mapping object_id -> mask
        
    Returns:
        Dictionary with statistics for each mask
    """
    statistics = {}
    
    for obj_id, mask in masks.items():
        # Basic statistics
        mask_binary = mask > 0.5
        area = np.sum(mask_binary)
        
        if area > 0:
            # Bounding box
            y_coords, x_coords = np.where(mask_binary)
            bbox = [
                int(x_coords.min()), int(y_coords.min()),
                int(x_coords.max()), int(y_coords.max())
            ]
            
            # Centroid
            centroid = [float(x_coords.mean()), float(y_coords.mean())]
            
            # Compactness (area / perimeter^2)
            contours, _ = cv2.findContours(
                mask_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            perimeter = sum(cv2.arcLength(c, True) for c in contours)
            compactness = area / (perimeter ** 2) if perimeter > 0 else 0
            
        else:
            bbox = [0, 0, 0, 0]
            centroid = [0.0, 0.0]
            compactness = 0.0
        
        statistics[obj_id] = {
            'area': int(area),
            'bbox': bbox,
            'centroid': centroid,
            'compactness': float(compactness),
            'width': bbox[2] - bbox[0],
            'height': bbox[3] - bbox[1]
        }
    
    return statistics


def export_masks_to_coco_format(
    masks: Dict[int, Dict[int, np.ndarray]],  # frame_idx -> {obj_id: mask}
    frame_files: List[str],
    output_file: str,
    image_dir: str,
    categories: Optional[List[Dict]] = None
) -> str:
    """
    Export masks to COCO format annotation file.
    
    Args:
        masks: Nested dictionary with frame and object masks
        frame_files: List of frame filenames
        output_file: Output JSON file path
        image_dir: Directory containing images
        categories: Optional list of category dictionaries
        
    Returns:
        Path to saved annotation file
    """
    if categories is None:
        categories = [{"id": 1, "name": "ui_element", "supercategory": "interface"}]
    
    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": "UI Element Tracking Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "UI Tracker",
            "date_created": "2025-01-01"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    annotation_id = 1
    
    for frame_idx, frame_masks in masks.items():
        if frame_idx >= len(frame_files):
            continue
            
        frame_file = frame_files[frame_idx]
        
        # Get image dimensions
        image_path = os.path.join(image_dir, frame_file)
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
        else:
            # Use first mask dimensions as fallback
            if frame_masks:
                first_mask = next(iter(frame_masks.values()))
                height, width = first_mask.shape[:2]
            else:
                continue
        
        # Add image info
        image_info = {
            "id": frame_idx,
            "width": width,
            "height": height,
            "file_name": frame_file
        }
        coco_data["images"].append(image_info)
        
        # Add annotations for each object
        for obj_id, mask in frame_masks.items():
            # Convert mask to RLE or polygon
            mask_binary = (mask > 0.5).astype(np.uint8)
            area = int(np.sum(mask_binary))
            
            if area == 0:
                continue
            
            # Get bounding box
            y_coords, x_coords = np.where(mask_binary)
            bbox = [
                int(x_coords.min()),
                int(y_coords.min()),
                int(x_coords.max() - x_coords.min()),
                int(y_coords.max() - y_coords.min())
            ]
            
            # Convert to polygon
            polygons = mask_to_polygon(mask_binary)
            
            # Create annotation
            annotation = {
                "id": annotation_id,
                "image_id": frame_idx,
                "category_id": 1,  # UI element category
                "segmentation": [
                    [coord for point in polygon for coord in point]
                    for polygon in polygons
                ],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            }
            
            coco_data["annotations"].append(annotation)
            annotation_id += 1
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    return output_file
