#!/usr/bin/env python3
"""
CLI: Pipeline C — preprocess → SoftEdge (PidiNet) → ControlNet SoftEdge + Img2Img + Waldo ckpt → post.

Examples:
  python run_stylize_c.py images/scene.png -o output/stylized_c.png
  python run_stylize_c.py photo.jpg -o out.png --strength 0.4 --no-bilateral --no-post
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stylize.pipeline_c import StylizeConfigC, stylize_image


def main() -> None:
    p = argparse.ArgumentParser(description="Where's Waldo stylization — Pipeline C (SoftEdge)")
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
        default=0.28,
        help="Img2Img strength (phase-1 global: ~0.25–0.30 for stable base)",
    )
    p.add_argument(
        "--control-scale",
        type=float,
        default=0.40,
        help="ControlNet conditioning (~0.35–0.45 for global pass)",
    )
    p.add_argument("--steps", type=int, default=30, help="Inference steps")
    p.add_argument("--guidance", type=float, default=7.5, help="CFG scale")
    p.add_argument("--max-long", type=int, default=1280, help="Max long edge (pixels)")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--prompt", type=str, default=None, help="Override positive prompt")
    p.add_argument("--negative-prompt", type=str, default=None, help="Override negative prompt")
    p.add_argument(
        "--no-bilateral",
        action="store_true",
        help="Skip Step 1 bilateral smoothing (resize only)",
    )
    p.add_argument(
        "--no-softedge-safe",
        action="store_true",
        help="PidiNet with safe=False (default uses safe=True)",
    )
    p.add_argument(
        "--softedge-max-resolution",
        type=int,
        default=768,
        metavar="N",
        help="Cap PidiNet max edge (default 768; use 1024 for finer structure, or a huge value to relax cap)",
    )
    p.add_argument(
        "--no-post",
        action="store_true",
        help="Skip Step 4 sharpen/color postprocess",
    )
    p.add_argument("--post-sharpen", type=float, default=1.08, help="ImageEnhance.Sharpness factor")
    p.add_argument("--post-color", type=float, default=1.02, help="ImageEnhance.Color factor")
    args = p.parse_args()

    cfg_kwargs = dict(
        num_inference_steps=args.steps,
        strength=args.strength,
        guidance_scale=args.guidance,
        controlnet_conditioning_scale=args.control_scale,
        max_long_edge=args.max_long,
        seed=args.seed,
        preprocess_use_bilateral=not args.no_bilateral,
        softedge_safe=not args.no_softedge_safe,
        postprocess_enabled=not args.no_post,
        post_sharpen=args.post_sharpen,
        post_color=args.post_color,
    )
    if args.prompt is not None:
        cfg_kwargs["prompt"] = args.prompt
    if args.negative_prompt is not None:
        cfg_kwargs["negative_prompt"] = args.negative_prompt
    cfg_kwargs["softedge_max_resolution"] = args.softedge_max_resolution

    config = StylizeConfigC(**cfg_kwargs)
    stylize_image(
        args.input,
        args.output,
        config=config,
        ckpt_path=args.ckpt,
    )


if __name__ == "__main__":
    main()
