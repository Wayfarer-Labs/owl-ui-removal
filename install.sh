#!/bin/bash

# Installation script for Owl UI Removal (uv-based)
set -e

echo "Installing Owl UI Removal..."
echo "=========================="

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is required for this project"
    echo "Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "Or visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "✓ uv detected"

# Check Python version with uv
python_version=$(uv python list | grep -E "python3\.[0-9]+" | head -n1 | awk '{print $1}' | cut -d'n' -f2)
if [ -z "$python_version" ]; then
    python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
fi

required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "Error: Python 3.8+ required, found $python_version"
    exit 1
fi

echo "✓ Python version check passed ($python_version)"

# Initialize uv project if not already initialized
# if [ ! -f "uv.lock" ]; then
#     echo "Initializing uv project..."
#     uv init --no-readme --no-workspace
# fi

# Install Python dependencies
echo "Installing Python dependencies..."
uv sync

# Install in development mode
echo "Installing project in development mode..."
uv pip install -e .

echo "✓ Python dependencies installed"

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    echo "Installing CUDA-enabled PyTorch..."
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "! No NVIDIA GPU detected, using CPU-only PyTorch"
fi

# Create directories
echo "Creating directories..."
mkdir -p checkpoints
mkdir -p configs
mkdir -p results
mkdir -p temp

echo "✓ Directories created"

# Setup Grounded-SAM-2
if [ ! -d "Grounded-SAM-2" ]; then
    echo "Cloning Grounded-SAM-2..."
    git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
    
    echo "Installing Grounded-SAM-2..."
    cd Grounded-SAM-2
    uv pip install -e .
    cd ..
    
    echo "✓ Grounded-SAM-2 installed"
else
    echo "✓ Grounded-SAM-2 already exists"
fi

# Check if models need to be downloaded
echo ""
echo "Model Setup"
echo "==========="
echo "You need to download model checkpoints:"
echo ""
echo "1. SAM2 checkpoints:"
echo "   https://github.com/facebookresearch/segment-anything-2/blob/main/INSTALL.md"
echo ""
echo "2. Place checkpoints in ./checkpoints/ directory"
echo ""
echo "3. Update paths in configuration files if needed"
echo ""

# Run setup utility
echo "Running setup utility..."
uv run setup_utils.py

echo ""
echo "Installation complete!"
echo "====================="
echo ""
echo "Next steps:"
echo "1. Download model checkpoints (see links above)"
echo "2. Test installation: python main.py --help"
echo "3. Create test data: python test_utils.py"
echo "4. Run example: python examples.py"
echo ""
echo "For detailed usage, see README.md"
