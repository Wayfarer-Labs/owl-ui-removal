# Owl UI Removal

A comprehensive tool for tracking and predicting UI masks in video sequences using Grounded-SAM-2. This project enables automatic detection and segmentation of user interface elements across video frames, making it ideal for UI/UX analysis, automated testing, and dataset creation.

## Features

- **Advanced UI Detection**: Uses Grounding DINO for text-based UI element detection
- **Robust Tracking**: Leverages SAM-2 for consistent object tracking across video frames
- **Multiple Input Formats**: Supports both video files and PNG frame sequences
- **Flexible Configuration**: Preset configurations for different UI types (mobile, web, game)
- **Rich Output Formats**: Saves masks, annotations, visualizations, and COCO format exports
- **Batch Processing**: Efficiently processes large video sequences

## Installation

### Prerequisites

1. **Python 3.8+** 
2. **uv package manager** ([install instructions](https://docs.astral.sh/uv/getting-started/installation/))
3. **CUDA-capable GPU** (recommended) or CPU
4. **Grounded-SAM-2** repository (auto-installed by setup script)

### Step 1: Clone and Setup

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone this repository
git clone <your-repo-url>
cd owl_ui_removal

# Run the installation script
chmod +x install.sh
./install.sh
```

### Alternative Manual Installation

```bash
# Initialize uv project
uv init --no-readme --no-workspace

# Install dependencies
uv sync

# Install in development mode
uv pip install -e .
```

### Step 2: Setup Grounded-SAM-2

```bash
# Clone Grounded-SAM-2
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git

# Install Grounded-SAM-2
cd Grounded-SAM-2
pip install -e .

# Download model checkpoints
mkdir -p checkpoints
# Download SAM2 checkpoints (see Grounded-SAM-2 documentation)
```

### Step 3: Verify Installation

```bash
python setup_utils.py
```

## Quick Start

### Basic Usage

```bash
# Track UI elements in a video file
python main.py input_video.mp4 --output ./results

# Process a directory of PNG frames
python main.py ./frame_directory --output ./results

# Use mobile UI optimized settings
python main.py input_video.mp4 --preset mobile --output ./results
```

### Advanced Usage

```bash
# Custom detection prompt and parameters
python main.py video.mp4 \
  --text-prompt "button, menu, icon, search bar" \
  --box-threshold 0.3 \
  --step-size 5 \
  --export-video

# Extract frames at specific FPS and time range
python main.py long_video.mp4 \
  --extract-fps 2 \
  --start-time 30 \
  --end-time 90 \
  --max-frames 100
```

## Configuration

### Presets

The tool includes several preset configurations:

- `default`: General UI element detection
- `ui`: Optimized for desktop UI elements
- `mobile`: Mobile app interface elements
- `web`: Web page UI components
- `game`: Game interface elements

### Custom Configuration

Create a YAML configuration file:

```yaml
model:
  sam2_checkpoint: "./checkpoints/sam2.1_hiera_large.pt"
  sam2_config: "configs/sam2.1/sam2.1_hiera_l.yaml"
  device: "cuda"

detection:
  text_prompt: "button, menu, icon, text field"
  box_threshold: 0.25
  text_threshold: 0.25

tracking:
  step_size: 10
  save_visualizations: true

output:
  output_dir: "./results"
  export_coco: true
  export_video: true
```

Use with: `python main.py input.mp4 --config my_config.yaml`

## Output Structure

```
results/
├── masks/                    # Individual and combined masks
│   ├── frame_000001_obj_1.npy
│   ├── frame_000001_combined.npy
│   └── ...
├── annotations/             # JSON annotations with bounding boxes
│   ├── frame_000001.json
│   └── ...
├── visualizations/          # Visual overlays (if enabled)
│   ├── frame_000001_tracked.png
│   └── ...
├── annotations.json         # COCO format (if enabled)
├── tracking_result.mp4      # Output video (if enabled)
└── config_used.yaml        # Configuration used for tracking
```

## API Usage

```python
from src.ui_tracker.tracker import UIElementTracker
from src.ui_tracker.config import get_mobile_ui_config

# Initialize tracker
config = get_mobile_ui_config()
tracker = UIElementTracker(
    sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
    sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml"
)

# Track UI elements
results = tracker.track_ui_elements(
    frame_dir="./video_frames",
    text_prompt="button, menu, icon",
    output_dir="./results"
)

# Results is a dictionary: {frame_idx: {object_id: mask}}
```

## Supported UI Elements

The tool can detect and track various UI elements including:

**General UI Components:**
- Buttons, icons, menus
- Text fields, input boxes
- Dropdowns, checkboxes, sliders
- Tabs, dialogs, popups
- Toolbars, navigation bars

**Mobile UI:**
- Tap buttons, floating action buttons
- Tab bars, navigation bars
- Cards, tiles, switches

**Web UI:**
- Links, forms, headers
- Sidebars, footers, modals
- Breadcrumbs, pagination

**Game UI:**
- Health bars, minimaps
- Inventory panels, HUD elements
- Menu systems, achievement notifications

## Performance Tips

1. **GPU Usage**: Use CUDA-enabled GPU for faster processing
2. **Step Size**: Increase step size for longer videos to reduce processing time
3. **Frame Resolution**: Consider resizing frames if very high resolution
4. **Batch Size**: Adjust based on available memory

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure Grounded-SAM-2 is properly installed
2. **CUDA Out of Memory**: Reduce batch size or use CPU mode
3. **No Objects Detected**: Adjust box_threshold or text_prompt
4. **Slow Processing**: Increase step_size or use smaller model checkpoint

### Debug Mode

Run with verbose logging:
```bash
python main.py input.mp4 --verbose --output ./results
```

## Examples

### Example 1: Mobile App Screen Recording

```bash
python main.py mobile_app_recording.mp4 \
  --preset mobile \
  --text-prompt "button, icon, tab, search bar, menu" \
  --export-video \
  --output ./mobile_results
```

### Example 2: Web Browser Session

```bash
python main.py web_session.mp4 \
  --preset web \
  --step-size 15 \
  --box-threshold 0.3 \
  --export-coco \
  --output ./web_results
```

### Example 3: Game UI Analysis

```bash
python main.py gameplay_video.mp4 \
  --preset game \
  --text-prompt "health bar, minimap, inventory, menu" \
  --start-time 60 \
  --end-time 180 \
  --output ./game_results
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) for the core tracking capabilities
- [Segment Anything 2](https://github.com/facebookresearch/segment-anything-2) for segmentation models
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) for object detection

## Citation

If you use this work in your research, please cite:

```bibtex
@software{owl_ui_removal,
  title={Owl UI Removal: UI Element Tracking with Grounded-SAM-2},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/owl_ui_removal}
}
```
