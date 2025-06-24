#!/usr/bin/env python3
"""
Download SAM2 checkpoints script
"""

from setup_utils import download_sam2_checkpoints

if __name__ == "__main__":
    print("Downloading SAM2 checkpoints...")
    download_sam2_checkpoints()
    print("Download completed!")
