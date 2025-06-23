"""
Setup utilities for Grounded-SAM-2 integration.
"""

import os
import subprocess
import urllib.request
import logging
from pathlib import Path


def setup_grounded_sam2():
    """
    Setup Grounded-SAM-2 repository and dependencies.
    This function helps with initial setup.
    """
    print("Setting up Grounded-SAM-2...")
    
    # Check if Grounded-SAM-2 is already available
    try:
        import sam2
        print("✓ SAM2 is already available")
        return True
    except ImportError:
        pass
    
    print("Grounded-SAM-2 setup is required.")
    print("Please follow these steps:")
    print()
    print("1. Clone Grounded-SAM-2 repository:")
    print("   git clone https://github.com/IDEA-Research/Grounded-SAM-2.git")
    print()
    print("2. Install dependencies:")
    print("   cd Grounded-SAM-2")
    print("   pip install -e .")
    print()
    print("3. Download model checkpoints:")
    print("   mkdir -p checkpoints")
    print("   # Download SAM2 checkpoints from the official repo")
    print("   # Download Grounding DINO checkpoints")
    print()
    print("4. Update paths in your configuration:")
    print("   - Set sam2_checkpoint path")
    print("   - Set sam2_config path")
    print()
    print("For detailed instructions, visit:")
    print("https://github.com/IDEA-Research/Grounded-SAM-2")
    
    return False


def download_sam2_checkpoints(checkpoint_dir: str = "./checkpoints"):
    """
    Download SAM2 model checkpoints.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # SAM2 checkpoint URLs (these are examples - use actual URLs)
    checkpoints = {
        "sam2.1_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "sam2.1_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "sam2.1_hiera_small.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "sam2.1_hiera_tiny.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    }
    
    for filename, url in checkpoints.items():
        filepath = os.path.join(checkpoint_dir, filename)
        
        if os.path.exists(filepath):
            print(f"✓ {filename} already exists")
            continue
        
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")


def check_dependencies():
    """
    Check if all required dependencies are available.
    """
    dependencies = [
        'torch',
        'torchvision', 
        'cv2',
        'PIL',
        'numpy',
        'supervision',
        'transformers',
        'matplotlib',
        'tqdm',
        'hydra',
        'omegaconf'
    ]
    
    missing = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep} (missing)")
            missing.append(dep)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install them with: pip install " + " ".join(missing))
        return False
    
    print("\n✓ All dependencies are available")
    return True


def create_example_config():
    """
    Create an example configuration file.
    """
    config_content = """
# Example configuration for UI tracking
model:
  sam2_checkpoint: "./checkpoints/sam2.1_hiera_large.pt"
  sam2_config: "./configs/sam2.1/sam2.1_hiera_l.yaml"
  grounding_model_id: "IDEA-Research/grounding-dino-tiny"
  device: "auto"
  use_bfloat16: true

detection:
  text_prompt: "button, menu, icon, text field, dropdown, checkbox, slider, tab, dialog, popup, toolbar, navigation bar, status bar, scroll bar"
  box_threshold: 0.25
  text_threshold: 0.25
  prompt_type: "mask"

tracking:
  step_size: 10
  max_frames: null
  save_visualizations: true
  save_individual_masks: true
  save_combined_masks: true
  save_annotations: true

output:
  output_dir: "./results"
  export_coco: false
  export_video: false
  video_fps: 30.0
  visualization_alpha: 0.4
"""
    
    with open("example_config.yaml", "w") as f:
        f.write(config_content)
    
    print("Created example_config.yaml")


if __name__ == "__main__":
    print("Grounded-SAM-2 Setup Utility")
    print("=" * 40)
    
    print("\n1. Checking dependencies...")
    check_dependencies()
    
    print("\n2. Checking Grounded-SAM-2 setup...")
    setup_grounded_sam2()
    
    print("\n3. Creating example configuration...")
    create_example_config()
    
    print("\nSetup complete!")
    print("Run 'python main.py --help' for usage instructions.")
