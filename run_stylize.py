#!/usr/bin/env python3
"""
CLI: run Pipeline on a single image (ControlNet Canny + Img2Img + Waldo ckpt).

Examples:
  python run_stylize.py images/level1-scene.png -o output/stylized.png
  python run_stylize.py photo.jpg -o out.png --strength 0.5 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stylize.pipeline_a import StylizeConfig, stylize_image


def main() -> None:
    p = argparse.ArgumentParser(description="Where's Waldo stylization (Pipeline A)")
    p.add_argument("input", type=Path, help="Input image path")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output image path")
    p.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Local Waldo merged .ckpt (default: download from Hugging Face)",
    )
    p.add_argument(
        "--strength",
        type=float,
        default=0.55,
        help="Img2Img strength (typical range ~0.45–0.65)",
    )
    p.add_argument(
        "--control-scale",
        type=float,
        default=0.85,
        help="ControlNet conditioning scale",
    )
    p.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    p.add_argument("--guidance", type=float, default=7.5, help="Classifier-free guidance scale")
    p.add_argument(
        "--max-long",
        type=int,
        default=640,
        help="Max long edge in pixels (Waldo model often works best at ≤640)",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override default positive prompt",
    )
    p.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Override default negative prompt",
    )
    args = p.parse_args()

    cfg_kwargs = dict(
        num_inference_steps=args.steps,
        strength=args.strength,
        guidance_scale=args.guidance,
        controlnet_conditioning_scale=args.control_scale,
        max_long_edge=args.max_long,
        seed=args.seed,
    )
    if args.prompt is not None:
        cfg_kwargs["prompt"] = args.prompt
    if args.negative_prompt is not None:
        cfg_kwargs["negative_prompt"] = args.negative_prompt

    config = StylizeConfig(**cfg_kwargs)
    stylize_image(
        args.input,
        args.output,
        config=config,
        ckpt_path=args.ckpt,
    )


if __name__ == "__main__":
    main()
