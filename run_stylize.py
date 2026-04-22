#!/usr/bin/env python3
"""
CLI: run Pipeline on a single image (ControlNet Canny + Img2Img + Waldo ckpt).

Examples:
  python run_stylize.py images/level1-scene.png -o output/stylized.png
"""

from run_stylize_c import main

if __name__ == "__main__":
    main()
