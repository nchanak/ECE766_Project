import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dino_sam_semantic_depth import analyze_scene
from place_waldo import choose_waldo_placement
from stylize.pipeline_a import StylizeConfig, WaldoStylizerPipelineA
from waldo_blending_pipeline import blend_waldo_into_scene
from waldo_blending_pipeline import match_color_local


DEFAULT_PIPELINE_ORDER = "place_blend_stylize"
PIPELINE_ORDERS = {
    "place_blend_stylize",
    "place_stylize",
    "stylize_bg_place_raw_waldo",
    "stylize_bg_place_local_waldo",
    "stylize_place_blend",
    "place_blend",
}


def apply_scene_occlusion(
    composited_rgb: np.ndarray,
    background_rgb: np.ndarray,
    occ_mask: np.ndarray,
) -> np.ndarray:
    result = composited_rgb.copy()
    result[occ_mask] = background_rgb[occ_mask]
    return result


def alpha_place_without_blending(
    background_rgb: np.ndarray,
    waldo_rgba: np.ndarray,
    x: int,
    y: int,
) -> np.ndarray:
    result = background_rgb.copy()
    h_bg, w_bg = result.shape[:2]
    h_waldo, w_waldo = waldo_rgba.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_bg, x + w_waldo)
    y2 = min(h_bg, y + h_waldo)
    if x1 >= x2 or y1 >= y2:
        return result

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    waldo_crop = waldo_rgba[oy1:oy2, ox1:ox2]
    alpha = waldo_crop[..., 3:4].astype(np.float32) / 255.0
    bg_crop = result[y1:y2, x1:x2].astype(np.float32)
    waldo_rgb = waldo_crop[..., :3].astype(np.float32)
    result[y1:y2, x1:x2] = np.clip(alpha * waldo_rgb + (1.0 - alpha) * bg_crop, 0, 255).astype(np.uint8)
    return result


def lightly_color_match_waldo(
    background_rgb: np.ndarray,
    waldo_rgba: np.ndarray,
    x: int,
    y: int,
    *,
    color_match_strength: float,
) -> np.ndarray:
    if color_match_strength <= 0.0:
        return waldo_rgba

    matched_rgb = match_color_local(waldo_rgba[..., :3][:, :, ::-1], background_rgb[:, :, ::-1], x, y)[:, :, ::-1]
    base_rgb = waldo_rgba[..., :3].astype(np.float32)
    matched_rgb = matched_rgb.astype(np.float32)
    mixed_rgb = np.clip(
        (1.0 - color_match_strength) * base_rgb + color_match_strength * matched_rgb,
        0,
        255,
    ).astype(np.uint8)
    return np.dstack([mixed_rgb, waldo_rgba[..., 3]])


def place_alpha_mask(
    canvas_shape: tuple[int, int],
    waldo_rgba: np.ndarray,
    x: int,
    y: int,
) -> np.ndarray:
    mask = np.zeros(canvas_shape, dtype=np.uint8)
    h_canvas, w_canvas = canvas_shape
    h_waldo, w_waldo = waldo_rgba.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_canvas, x + w_waldo)
    y2 = min(h_canvas, y + h_waldo)
    if x1 >= x2 or y1 >= y2:
        return mask

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)
    mask[y1:y2, x1:x2] = waldo_rgba[oy1:oy2, ox1:ox2, 3]
    return mask


def compute_local_crop_bounds(
    scene_shape: tuple[int, int, int],
    waldo_rgba: np.ndarray,
    x: int,
    y: int,
    pad_scale: float,
) -> tuple[int, int, int, int]:
    h_scene, w_scene = scene_shape[:2]
    h_waldo, w_waldo = waldo_rgba.shape[:2]
    pad_x = int(round(w_waldo * pad_scale))
    pad_y = int(round(h_waldo * pad_scale))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_scene, x + w_waldo + pad_x)
    y2 = min(h_scene, y + h_waldo + pad_y)
    return x1, y1, x2, y2


def extract_stylized_waldo_from_crop(
    stylized_crop_rgb: np.ndarray,
    waldo_alpha_mask: np.ndarray,
    waldo_local_box: tuple[int, int, int, int],
) -> np.ndarray:
    wx1, wy1, wx2, wy2 = waldo_local_box
    rgb = stylized_crop_rgb[wy1:wy2, wx1:wx2]
    alpha = waldo_alpha_mask[wy1:wy2, wx1:wx2]
    return np.dstack([rgb, alpha])


def stylize_local_waldo_crop(
    stylizer: WaldoStylizerPipelineA,
    background_rgb: np.ndarray,
    waldo_rgba: np.ndarray,
    x: int,
    y: int,
    config: StylizeConfig | None,
    *,
    crop_pad_scale: float,
) -> dict[str, Any]:
    crop_x1, crop_y1, crop_x2, crop_y2 = compute_local_crop_bounds(
        background_rgb.shape,
        waldo_rgba,
        x,
        y,
        crop_pad_scale,
    )
    crop_bg = background_rgb[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    local_x = x - crop_x1
    local_y = y - crop_y1
    crop_with_waldo = alpha_place_without_blending(crop_bg, waldo_rgba, local_x, local_y)
    stylized_crop = maybe_stylize_image(
        stylizer,
        Image.fromarray(crop_with_waldo),
        config,
        (crop_x2 - crop_x1, crop_y2 - crop_y1),
    )
    stylized_crop_rgb = np.array(stylized_crop.convert("RGB"))
    waldo_alpha_mask = place_alpha_mask((crop_y2 - crop_y1, crop_x2 - crop_x1), waldo_rgba, local_x, local_y)
    stylized_waldo_rgba = extract_stylized_waldo_from_crop(
        stylized_crop_rgb,
        waldo_alpha_mask,
        (local_x, local_y, local_x + waldo_rgba.shape[1], local_y + waldo_rgba.shape[0]),
    )
    return {
        "crop_box": [crop_x1, crop_y1, crop_x2, crop_y2],
        "crop_background_rgb": crop_bg,
        "crop_with_waldo_rgb": crop_with_waldo,
        "stylized_crop_rgb": stylized_crop_rgb,
        "stylized_waldo_rgba": stylized_waldo_rgba,
    }


def save_image(image: Image.Image | np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, np.ndarray):
        Image.fromarray(image).save(path)
    else:
        image.save(path)


def maybe_stylize_image(
    stylizer: WaldoStylizerPipelineA | None,
    image: Image.Image,
    config: StylizeConfig | None,
    target_size: tuple[int, int],
) -> Image.Image:
    if stylizer is None:
        return image
    stylized = stylizer.stylize(image.convert("RGB"), config=config)
    if stylized.size != target_size:
        stylized = stylized.resize(target_size, Image.LANCZOS)
    return stylized


def run_waldo_pipeline(
    image_path: str | Path,
    waldo_path: str | Path = "waldo.png",
    output_dir: str | Path = "layered_output",
    *,
    pipeline_order: str = DEFAULT_PIPELINE_ORDER,
    analyze_max_size: int = 800,
    placement_seed: int = 34,
    stylize: bool = True,
    stylize_ckpt: str | Path | None = None,
    stylize_config: StylizeConfig | None = None,
    apply_color_match: bool = True,
    apply_sharpness_match: bool = True,
    add_noise: bool = True,
    feather: bool = True,
    feather_ksize: int = 5,
    local_waldo_crop_pad_scale: float = 2.0,
    raw_waldo_color_match_strength: float = 0.65,
) -> dict[str, Any]:
    if pipeline_order not in PIPELINE_ORDERS:
        raise ValueError(f"pipeline_order must be one of {sorted(PIPELINE_ORDERS)}")

    image_path = Path(image_path)
    waldo_path = Path(waldo_path)
    output_dir = Path(output_dir)
    stem = image_path.stem

    analysis_meta = analyze_scene(
        image_path,
        output_dir=output_dir,
        clean_output=True,
        max_size=analyze_max_size,
    )
    placement_result = choose_waldo_placement(
        image_path,
        waldo_path=waldo_path,
        output_dir=output_dir,
        seed=placement_seed,
    )

    scene_size = tuple(analysis_meta["image_size"])
    base_scene = placement_result["scene_image"].convert("RGB")
    stylizer = None
    if stylize and pipeline_order != "place_blend":
        stylizer = WaldoStylizerPipelineA(ckpt_path=stylize_ckpt)

    if pipeline_order in {"stylize_place_blend", "stylize_bg_place_local_waldo", "stylize_bg_place_raw_waldo"}:
        working_scene = maybe_stylize_image(stylizer, base_scene, stylize_config, scene_size)
        save_image(working_scene, output_dir / f"{stem}_scene_stylized.png")
    else:
        working_scene = base_scene

    working_scene_rgb = np.array(working_scene.convert("RGB"))
    waldo_rgba = np.array(placement_result["waldo_scaled"].convert("RGBA"))
    place_x, place_y = placement_result["placement"]["top_left"]

    local_waldo_result = None
    if pipeline_order == "place_stylize":
        placed_rgb = alpha_place_without_blending(
            working_scene_rgb,
            waldo_rgba,
            place_x,
            place_y,
        )
        blended_with_occlusion = apply_scene_occlusion(
            placed_rgb,
            working_scene_rgb,
            placement_result["occ_mask"],
        )
        blend_result = None
    elif pipeline_order == "stylize_bg_place_raw_waldo":
        raw_waldo_rgba = lightly_color_match_waldo(
            working_scene_rgb,
            waldo_rgba,
            place_x,
            place_y,
            color_match_strength=raw_waldo_color_match_strength,
        )
        placed_rgb = alpha_place_without_blending(
            working_scene_rgb,
            raw_waldo_rgba,
            place_x,
            place_y,
        )
        blended_with_occlusion = apply_scene_occlusion(
            placed_rgb,
            working_scene_rgb,
            placement_result["occ_mask"],
        )
        blend_result = None
        local_waldo_result = {
            "raw_waldo_rgba": raw_waldo_rgba,
        }
    elif pipeline_order == "stylize_bg_place_local_waldo":
        if stylizer is None:
            raise ValueError("stylize_bg_place_local_waldo requires stylization to be enabled.")
        local_waldo_result = stylize_local_waldo_crop(
            stylizer,
            working_scene_rgb,
            waldo_rgba,
            place_x,
            place_y,
            stylize_config,
            crop_pad_scale=local_waldo_crop_pad_scale,
        )
        blend_result = blend_waldo_into_scene(
            working_scene_rgb,
            local_waldo_result["stylized_waldo_rgba"],
            place_x,
            place_y,
            apply_color_match=apply_color_match,
            apply_sharpness_match=apply_sharpness_match,
            add_noise=add_noise,
            feather=feather,
            feather_ksize=feather_ksize,
        )
        blended_with_occlusion = apply_scene_occlusion(
            blend_result["composited_rgb"],
            working_scene_rgb,
            placement_result["occ_mask"],
        )
    else:
        blend_result = blend_waldo_into_scene(
            working_scene_rgb,
            waldo_rgba,
            place_x,
            place_y,
            apply_color_match=apply_color_match,
            apply_sharpness_match=apply_sharpness_match,
            add_noise=add_noise,
            feather=feather,
            feather_ksize=feather_ksize,
        )
        blended_with_occlusion = apply_scene_occlusion(
            blend_result["composited_rgb"],
            working_scene_rgb,
            placement_result["occ_mask"],
        )
    pre_style_path = output_dir / f"{stem}_pipeline_pre_style.png"
    save_image(blended_with_occlusion, pre_style_path)

    if pipeline_order in {"place_blend_stylize", "place_stylize"} and stylizer is not None:
        final_image = maybe_stylize_image(
            stylizer,
            Image.fromarray(blended_with_occlusion),
            stylize_config,
            scene_size,
        )
    else:
        final_image = Image.fromarray(blended_with_occlusion)

    final_path = output_dir / f"{stem}_pipeline_final.png"
    save_image(final_image, final_path)
    if blend_result is not None:
        save_image(np.array(blend_result["waldo_rgb"]), output_dir / f"{stem}_waldo_matched.png")
        save_image(blend_result["blend_alpha"], output_dir / f"{stem}_waldo_blend_alpha.png")
    if pipeline_order == "stylize_bg_place_local_waldo" and local_waldo_result is not None:
        save_image(local_waldo_result["crop_with_waldo_rgb"], output_dir / f"{stem}_local_waldo_crop_input.png")
        save_image(local_waldo_result["stylized_crop_rgb"], output_dir / f"{stem}_local_waldo_crop_stylized.png")
        save_image(local_waldo_result["stylized_waldo_rgba"], output_dir / f"{stem}_local_waldo_stylized.png")

    manifest = {
        "image_path": str(image_path),
        "waldo_path": str(waldo_path),
        "output_dir": str(output_dir),
        "pipeline_order": pipeline_order,
        "stylize_enabled": stylize and pipeline_order != "place_blend",
        "blending_enabled": pipeline_order not in {"place_stylize", "stylize_bg_place_raw_waldo"},
        "blending": {
            "apply_color_match": apply_color_match,
            "apply_sharpness_match": apply_sharpness_match,
            "add_noise": add_noise,
            "feather": feather,
            "feather_ksize": feather_ksize,
        },
        "local_waldo_stylization": {
            "enabled": pipeline_order == "stylize_bg_place_local_waldo",
            "crop_pad_scale": local_waldo_crop_pad_scale,
        },
        "raw_waldo_placement": {
            "enabled": pipeline_order == "stylize_bg_place_raw_waldo",
            "color_match_strength": raw_waldo_color_match_strength,
        },
        "placement": placement_result["placement"],
        "artifacts": {
            "analysis_meta": str(output_dir / f"{stem}_layers.json"),
            "pre_style_composite": str(pre_style_path),
            "final_image": str(final_path),
        },
    }
    if blend_result is not None:
        manifest["artifacts"]["matched_waldo"] = str(output_dir / f"{stem}_waldo_matched.png")
        manifest["artifacts"]["blend_alpha"] = str(output_dir / f"{stem}_waldo_blend_alpha.png")
    if pipeline_order == "stylize_bg_place_local_waldo" and local_waldo_result is not None:
        manifest["artifacts"]["local_waldo_crop_input"] = str(output_dir / f"{stem}_local_waldo_crop_input.png")
        manifest["artifacts"]["local_waldo_crop_stylized"] = str(output_dir / f"{stem}_local_waldo_crop_stylized.png")
        manifest["artifacts"]["local_waldo_stylized"] = str(output_dir / f"{stem}_local_waldo_stylized.png")
        manifest["local_waldo_stylization"]["crop_box"] = local_waldo_result["crop_box"]
    if pipeline_order == "stylize_bg_place_raw_waldo":
        save_image(local_waldo_result["raw_waldo_rgba"], output_dir / f"{stem}_raw_waldo_light_matched.png")
        manifest["artifacts"]["raw_waldo_light_matched"] = str(output_dir / f"{stem}_raw_waldo_light_matched.png")

    with open(output_dir / f"{stem}_pipeline_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Where's Waldo pipeline")
    parser.add_argument("image", help="Input background image")
    parser.add_argument("--waldo", default="waldo.png", help="Path to Waldo cutout")
    parser.add_argument("--output-dir", default="layered_output", help="Output directory")
    parser.add_argument(
        "--pipeline-order",
        choices=sorted(PIPELINE_ORDERS),
        default=DEFAULT_PIPELINE_ORDER,
        help="Stage ordering to use",
    )
    parser.add_argument("--analyze-max-size", type=int, default=800, help="Resize long edge before scene analysis")
    parser.add_argument("--seed", type=int, default=34, help="Placement seed")
    parser.add_argument("--no-stylize", action="store_true", help="Skip the stylization stage")
    parser.add_argument("--ckpt", type=Path, default=None, help="Optional local stylization checkpoint")
    parser.add_argument("--no-color-match", action="store_true", help="Disable local color matching during Waldo blending")
    parser.add_argument("--no-sharpness-match", action="store_true", help="Disable local sharpness matching during Waldo blending")
    parser.add_argument("--no-noise", action="store_true", help="Disable background-matched noise injection during Waldo blending")
    parser.add_argument("--no-feather", action="store_true", help="Disable alpha feathering during Waldo blending")
    parser.add_argument("--feather-ksize", type=int, default=5, help="Feather kernel size for alpha blending")
    parser.add_argument("--local-waldo-crop-pad-scale", type=float, default=2.0, help="Padding multiplier around Waldo for local crop stylization")
    parser.add_argument("--raw-waldo-color-match-strength", type=float, default=0.75, help="Blend factor for light color matching when placing raw Waldo after background stylization")
    parser.add_argument("--strength", type=float, default=0.85, help="Stylizer img2img strength")
    parser.add_argument("--control-scale", type=float, default=0.85, help="ControlNet conditioning scale")
    parser.add_argument("--steps", type=int, default=24, help="Stylizer inference steps")
    parser.add_argument("--guidance", type=float, default=3.0, help="Stylizer guidance scale")
    parser.add_argument("--max-long", type=int, default=896, help="Stylizer max long edge")
    parser.add_argument("--seed-style", type=int, default=None, help="Optional stylizer seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stylize_config = StylizeConfig(
        num_inference_steps=args.steps,
        strength=args.strength,
        guidance_scale=args.guidance,
        controlnet_conditioning_scale=args.control_scale,
        max_long_edge=args.max_long,
        seed=args.seed_style,
    )
    manifest = run_waldo_pipeline(
        args.image,
        waldo_path=args.waldo,
        output_dir=args.output_dir,
        pipeline_order=args.pipeline_order,
        analyze_max_size=args.analyze_max_size,
        placement_seed=args.seed,
        stylize=not args.no_stylize,
        stylize_ckpt=args.ckpt,
        stylize_config=stylize_config,
        apply_color_match=not args.no_color_match,
        apply_sharpness_match=not args.no_sharpness_match,
        add_noise=not args.no_noise,
        feather=not args.no_feather,
        feather_ksize=args.feather_ksize,
        local_waldo_crop_pad_scale=args.local_waldo_crop_pad_scale,
        raw_waldo_color_match_strength=args.raw_waldo_color_match_strength,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
