"""
Configuration management for UI tracking.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import yaml
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    sam2_checkpoint: str = "./checkpoints/sam2.1_hiera_large.pt"
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    grounding_model_id: str = "IDEA-Research/grounding-dino-tiny"
    device: str = "auto"
    use_bfloat16: bool = True


@dataclass
class DetectionConfig:
    """Configuration for UI element detection."""
    text_prompt: str = "button, menu, icon, text field, dropdown, checkbox, slider, tab, dialog, popup, toolbar, navigation bar, status bar, scroll bar"
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    prompt_type: str = "mask"  # "mask", "box", or "point"


@dataclass
class TrackingConfig:
    """Configuration for tracking parameters."""
    step_size: int = 10
    max_frames: Optional[int] = None
    save_visualizations: bool = True
    save_individual_masks: bool = True
    save_combined_masks: bool = True
    save_annotations: bool = True


@dataclass
class OutputConfig:
    """Configuration for output settings."""
    output_dir: str = "./results"
    export_coco: bool = False
    export_video: bool = False
    video_fps: float = 30.0
    visualization_alpha: float = 0.4


@dataclass
class UITrackerConfig:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'UITrackerConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, config_path: str) -> 'UITrackerConfig':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UITrackerConfig':
        """Create configuration from dictionary."""
        model_config = ModelConfig(**config_dict.get('model', {}))
        detection_config = DetectionConfig(**config_dict.get('detection', {}))
        tracking_config = TrackingConfig(**config_dict.get('tracking', {}))
        output_config = OutputConfig(**config_dict.get('output', {}))
        
        return cls(
            model=model_config,
            detection=detection_config,
            tracking=tracking_config,
            output=output_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': {
                'sam2_checkpoint': self.model.sam2_checkpoint,
                'sam2_config': self.model.sam2_config,
                'grounding_model_id': self.model.grounding_model_id,
                'device': self.model.device,
                'use_bfloat16': self.model.use_bfloat16
            },
            'detection': {
                'text_prompt': self.detection.text_prompt,
                'box_threshold': self.detection.box_threshold,
                'text_threshold': self.detection.text_threshold,
                'prompt_type': self.detection.prompt_type
            },
            'tracking': {
                'step_size': self.tracking.step_size,
                'max_frames': self.tracking.max_frames,
                'save_visualizations': self.tracking.save_visualizations,
                'save_individual_masks': self.tracking.save_individual_masks,
                'save_combined_masks': self.tracking.save_combined_masks,
                'save_annotations': self.tracking.save_annotations
            },
            'output': {
                'output_dir': self.output.output_dir,
                'export_coco': self.output.export_coco,
                'export_video': self.output.export_video,
                'video_fps': self.output.video_fps,
                'visualization_alpha': self.output.visualization_alpha
            }
        }
    
    def save_yaml(self, output_path: str):
        """Save configuration to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def save_json(self, output_path: str):
        """Save configuration to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def create_default_config() -> UITrackerConfig:
    """Create a default configuration."""
    return UITrackerConfig()


def load_config(config_path: str) -> UITrackerConfig:
    """Load configuration from file (auto-detect format)."""
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
        return UITrackerConfig.from_yaml(config_path)
    elif path.suffix.lower() == '.json':
        return UITrackerConfig.from_json(config_path)
    else:
        raise ValueError(f"Unsupported configuration file format: {path.suffix}")


# Predefined configurations for common use cases
def get_ui_detection_config() -> UITrackerConfig:
    """Configuration optimized for general UI element detection."""
    config = create_default_config()
    config.detection.text_prompt = (
        "button, menu, icon, text field, input box, dropdown, select, checkbox, "
        "radio button, slider, tab, dialog, modal, popup, alert, toolbar, "
        "navigation bar, nav bar, menu bar, status bar, scroll bar, progress bar"
    )
    config.detection.box_threshold = 0.3
    config.detection.text_threshold = 0.3
    return config


def get_mobile_ui_config() -> UITrackerConfig:
    """Configuration optimized for mobile UI elements."""
    config = create_default_config()
    config.detection.text_prompt = (
        "button, tap button, icon, input field, text field, search bar, "
        "navigation bar, tab bar, toolbar, menu, hamburger menu, "
        "switch, toggle, slider, progress bar, alert, dialog, modal, "
        "floating action button, fab, card, tile"
    )
    config.detection.box_threshold = 0.25
    config.detection.text_threshold = 0.25
    config.tracking.step_size = 5  # More frequent tracking for mobile
    return config


def get_web_ui_config() -> UITrackerConfig:
    """Configuration optimized for web UI elements."""
    config = create_default_config()
    config.detection.text_prompt = (
        "button, link, hyperlink, input field, text box, textarea, dropdown, "
        "select box, checkbox, radio button, form, header, navigation, nav, "
        "sidebar, footer, modal, dialog, popup, tooltip, notification, "
        "breadcrumb, pagination, tab, accordion, carousel, slider"
    )
    config.detection.box_threshold = 0.3
    config.detection.text_threshold = 0.3
    config.tracking.step_size = 15  # Less frequent for potentially longer web videos
    return config


def get_game_ui_config() -> UITrackerConfig:
    """Configuration optimized for game UI elements."""
    config = create_default_config()
    config.detection.text_prompt = (
        "button, menu, inventory, health bar, mana bar, energy bar, "
        "minimap, radar, hud element, ui panel, dialog box, "
        "score display, timer, progress bar, skill tree, "
        "weapon selector, item icon, achievement notification"
    )
    config.detection.box_threshold = 0.2  # Lower threshold for game UI
    config.detection.text_threshold = 0.2
    config.tracking.step_size = 8
    return config
