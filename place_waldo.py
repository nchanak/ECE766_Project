import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_OUTPUT_DIR = Path("layered_output")
PREFERRED_SUPPORT_CLASSES = {
    "ground", "road", "earth", "grass", "rock", "sign",
    "building", "plant", "vegetation", "mountain", "tree", "cliff",
}
FORBIDDEN_CLASSES = {"sky", "water"}

MIN_SCALE = 0.08
MAX_SCALE = 0.28
MIN_OCCLUDED_FRAC = 0.5
MAX_OCCLUDED_FRAC = 0.8
NUM_POSITION_SAMPLES = 1200
SEED = 34
TOP_K_RANDOM = 25
TEXTURE_PATCH_PAD_FRAC = 0.35
TEXTURE_VARIANCE_WEIGHT = 0.45
TEXTURE_EDGE_WEIGHT = 0.55
TEXTURE_SCORE_WEIGHT = 2.0
ROTATION_ANGLE_CANDIDATES = [0, -20, 20, -35, 35, 145, 160, 180, 200, 215]
ROTATION_TIEBREAK_WEIGHT = 0.08
HEAD_REGION_TOP_FRAC = 0.16
HEAD_REGION_BOTTOM_FRAC = 0.72
HEAD_REGION_SIDE_MARGIN_FRAC = 0.44
MIN_HEAD_VISIBLE_FRAC = 0.8


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def alpha_bbox(alpha: np.ndarray):
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def resize_rgba_keep_aspect(img: Image.Image, target_h: int) -> Image.Image:
    w, h = img.size
    scale = target_h / h
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.LANCZOS)


def is_preferred_occluder(name: str) -> bool:
    name = name.lower()
    keywords = [
        "tree", "rock", "boulder", "fence", "pole", "sign",
        "car", "truck", "bus", "van", "building", "wall", "facade", "cliff", "skyscraper",
    ]
    return any(keyword in name for keyword in keywords)


def compute_support_mask(layer_entries, mask_root: Path, h: int, w: int, stem: str) -> tuple[np.ndarray, bool]:
    preferred = np.zeros((h, w), dtype=bool)
    forbidden = np.zeros((h, w), dtype=bool)

    for entry in layer_entries:
        name = entry["name"]
        mask = None

        mask_file = entry.get("mask_file")
        if mask_file:
            path = mask_root / mask_file
            if path.exists():
                mask = load_mask(path)

        if mask is None:
            semantic_path = mask_root / "out_masks" / f"{stem}_semantic_{name}.png"
            if semantic_path.exists():
                mask = load_mask(semantic_path)

        if mask is None:
            continue

        if name in PREFERRED_SUPPORT_CLASSES:
            preferred |= mask
        if name in FORBIDDEN_CLASSES:
            forbidden |= mask

    preferred_mask = preferred & (~forbidden)
    if preferred_mask.any():
        return preferred_mask, True
    return ~forbidden, False


def choose_scale_from_depth(depth_map_u8: np.ndarray, foot_x: int, foot_y: int, scene_h: int) -> int:
    depth_value = depth_map_u8[foot_y, foot_x] / 255.0
    relative_scale = MIN_SCALE + depth_value * (MAX_SCALE - MIN_SCALE)
    return max(12, int(scene_h * relative_scale))


def paste_rgba(base_rgba: np.ndarray, overlay_rgba: np.ndarray, x: int, y: int):
    h_base, w_base = base_rgba.shape[:2]
    h_overlay, w_overlay = overlay_rgba.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_base, x + w_overlay)
    y2 = min(h_base, y + h_overlay)
    if x1 >= x2 or y1 >= y2:
        return base_rgba

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    base_crop = base_rgba[y1:y2, x1:x2].astype(np.float32)
    over_crop = overlay_rgba[oy1:oy2, ox1:ox2].astype(np.float32)

    alpha = over_crop[..., 3:4] / 255.0
    out_rgb = alpha * over_crop[..., :3] + (1.0 - alpha) * base_crop[..., :3]
    out_a = np.maximum(base_crop[..., 3:4], over_crop[..., 3:4])
    base_rgba[y1:y2, x1:x2] = np.concatenate([out_rgb, out_a], axis=-1).clip(0, 255).astype(np.uint8)
    return base_rgba


def build_front_occlusion_mask(
    waldo_mask_scene: np.ndarray,
    waldo_layer_id: int,
    layer_entries,
    mask_root: Path,
) -> np.ndarray:
    occlusion = np.zeros_like(waldo_mask_scene, dtype=bool)
    for entry in layer_entries:
        if entry["layer_id"] <= waldo_layer_id:
            continue
        if not is_preferred_occluder(entry["name"]):
            continue
        mask_file = entry.get("mask_file")
        if not mask_file:
            continue
        obj_mask = load_mask(mask_root / mask_file)
        occlusion |= obj_mask & waldo_mask_scene
    return occlusion


def estimate_waldo_layer_id(foot_x: int, foot_y: int, layer_map: np.ndarray, layer_entries) -> int:
    local_layer = int(layer_map[foot_y, foot_x])
    if local_layer > 0:
        return local_layer

    h, w = layer_map.shape
    radius = 8
    x1 = max(0, foot_x - radius)
    x2 = min(w, foot_x + radius + 1)
    y1 = max(0, foot_y - radius)
    y2 = min(h, foot_y + radius + 1)
    patch = layer_map[y1:y2, x1:x2]
    vals = patch[patch > 0]
    if vals.size:
        return int(np.median(vals))

    if layer_entries:
        return int(np.median([entry["layer_id"] for entry in layer_entries]))
    return 1


def sample_candidate_points(mask: np.ndarray, n: int):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    idx = np.random.choice(len(xs), size=min(n, len(xs)), replace=False)
    return list(zip(xs[idx], ys[idx]))


def rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    return 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]


def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def compute_texture_score(
    scene_rgb: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    pad_frac: float = TEXTURE_PATCH_PAD_FRAC,
) -> float:
    h, w = scene_rgb.shape[:2]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = max(2, int(round(box_w * pad_frac)))
    pad_y = max(2, int(round(box_h * pad_frac)))
    px1, py1, px2, py2 = clamp_box(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, w, h)

    patch = scene_rgb[py1:py2, px1:px2, :3]
    if patch.size == 0:
        return 0.0

    luma = rgb_to_luma(patch)
    variance_score = float(np.std(luma) / 64.0)
    grad_x = np.abs(np.diff(luma, axis=1))
    grad_y = np.abs(np.diff(luma, axis=0))
    edge_score = float((grad_x.mean() + grad_y.mean()) / 24.0)

    combined = (
        TEXTURE_VARIANCE_WEIGHT * variance_score +
        TEXTURE_EDGE_WEIGHT * edge_score
    )
    return float(np.clip(combined, 0.0, 1.0))


def trim_waldo(waldo_img: Image.Image) -> Image.Image:
    waldo_alpha = np.array(waldo_img)[..., 3]
    waldo_box = alpha_bbox(waldo_alpha)
    if waldo_box is None:
        raise RuntimeError("Waldo image has empty alpha.")
    wx1, wy1, wx2, wy2 = waldo_box
    return waldo_img.crop((wx1, wy1, wx2 + 1, wy2 + 1))


def rotate_waldo_image(img: Image.Image, angle_degrees: int) -> Image.Image:
    if angle_degrees % 360 == 0:
        return img
    return img.rotate(
        angle_degrees,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(0, 0, 0, 0),
    )


def build_head_mask(alpha_mask: np.ndarray) -> np.ndarray:
    h, w = alpha_mask.shape
    x1 = int(round(w * HEAD_REGION_SIDE_MARGIN_FRAC))
    x2 = int(round(w * (1.0 - HEAD_REGION_SIDE_MARGIN_FRAC)))
    y1 = int(round(h * HEAD_REGION_TOP_FRAC))
    y2 = int(round(h * HEAD_REGION_BOTTOM_FRAC))
    head_mask = np.zeros_like(alpha_mask, dtype=bool)
    head_mask[y1:y2, x1:x2] = True
    return head_mask & alpha_mask


def place_local_mask_on_scene(
    local_mask: np.ndarray,
    scene_h: int,
    scene_w: int,
    place_x: int,
    place_y: int,
) -> np.ndarray:
    placed = np.zeros((scene_h, scene_w), dtype=bool)
    h_mask, w_mask = local_mask.shape
    x1 = max(0, place_x)
    y1 = max(0, place_y)
    x2 = min(scene_w, place_x + w_mask)
    y2 = min(scene_h, place_y + h_mask)
    if x1 >= x2 or y1 >= y2:
        return placed

    ox1 = x1 - place_x
    oy1 = y1 - place_y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)
    placed[y1:y2, x1:x2] = local_mask[oy1:oy2, ox1:ox2]
    return placed


def compute_visible_fraction(mask: np.ndarray, occlusion_mask: np.ndarray) -> float:
    total = int(mask.sum())
    if total == 0:
        return 1.0
    visible = int((mask & (~occlusion_mask)).sum())
    return visible / total


def choose_waldo_placement(
    image_path: str | Path,
    waldo_path: str | Path = "waldo.png",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    seed: int = SEED,
    num_position_samples: int = NUM_POSITION_SAMPLES,
    top_k_random: int = TOP_K_RANDOM,
    save_debug: bool = True,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)

    image_path = Path(image_path)
    waldo_path = Path(waldo_path)
    output_dir = Path(output_dir)
    stem = image_path.stem

    with open(output_dir / f"{stem}_layers.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    target_w, target_h = meta["image_size"]
    layer_entries = meta["layers"]

    scene_img = Image.open(image_path).convert("RGBA").resize((target_w, target_h), Image.LANCZOS)
    waldo_img = Image.open(waldo_path).convert("RGBA")
    scene_np = np.array(scene_img)
    h, w = scene_np.shape[:2]

    layer_map = np.array(Image.open(output_dir / f"{stem}_layer_map.png").convert("L"))
    depth_map_u8 = np.array(Image.open(output_dir / f"{stem}_depth_map.png").convert("L"))
    support_mask, used_preferred_support = compute_support_mask(layer_entries, output_dir, h, w, stem)
    candidates = sample_candidate_points(support_mask, num_position_samples)
    if not candidates:
        raise RuntimeError("No valid candidate support pixels found for Waldo placement.")

    waldo_trim = trim_waldo(waldo_img)
    all_sampled_candidates = []
    valid_candidates = []

    for foot_x, foot_y in candidates:
        all_sampled_candidates.append((int(foot_x), int(foot_y)))
        target_height = choose_scale_from_depth(depth_map_u8, foot_x, foot_y, h)
        waldo_scaled = resize_rgba_keep_aspect(waldo_trim, target_height)
        waldo_layer_id = estimate_waldo_layer_id(foot_x, foot_y, layer_map, layer_entries)
        depth_score = depth_map_u8[foot_y, foot_x] / 255.0
        y_score = foot_y / h

        variant_results = []
        for angle_degrees in ROTATION_ANGLE_CANDIDATES:
            waldo_variant_img = rotate_waldo_image(waldo_scaled, angle_degrees)
            waldo_variant_arr = np.array(waldo_variant_img)
            alpha_variant = waldo_variant_arr[..., 3] > 0
            h_waldo, w_waldo = alpha_variant.shape
            place_x = foot_x - w_waldo // 2
            place_y = foot_y - h_waldo + 1
            if place_x < -w_waldo // 3 or place_y < -h_waldo // 3 or place_x >= w or place_y >= h:
                continue

            x1 = max(0, place_x)
            y1 = max(0, place_y)
            x2 = min(w, place_x + w_waldo)
            y2 = min(h, place_y + h_waldo)
            if x1 >= x2 or y1 >= y2:
                continue

            texture_score = compute_texture_score(scene_np[..., :3], x1, y1, x2, y2)
            waldo_scene_mask = place_local_mask_on_scene(alpha_variant, h, w, place_x, place_y)
            total = int(waldo_scene_mask.sum())
            if total == 0:
                continue

            occ_mask = build_front_occlusion_mask(waldo_scene_mask, waldo_layer_id, layer_entries, output_dir)
            occ_frac = occ_mask.sum() / total
            if occ_frac > MAX_OCCLUDED_FRAC:
                continue

            head_mask_local = build_head_mask(alpha_variant)
            head_mask_scene = place_local_mask_on_scene(head_mask_local, h, w, place_x, place_y)
            head_visible_frac = compute_visible_fraction(head_mask_scene, occ_mask)
            occ_penalty = abs(occ_frac - 1.0) if occ_frac < MIN_OCCLUDED_FRAC else 0.0
            head_penalty = max(0.0, MIN_HEAD_VISIBLE_FRAC - head_visible_frac)
            rotation_penalty = ROTATION_TIEBREAK_WEIGHT * min(abs(angle_degrees), abs(abs(angle_degrees) - 180))
            score = (
                0.3 * y_score
                + 1.35 * (1.0 - abs(occ_frac - 0.6))
                + 0.3 * (1.0 - depth_score)
                + TEXTURE_SCORE_WEIGHT * texture_score
                - 1.45 * occ_penalty
                - 1.8 * head_penalty
                - rotation_penalty
            )
            variant_results.append({
                "rotation_degrees": int(angle_degrees),
                "place_x": int(place_x),
                "place_y": int(place_y),
                "waldo_scaled": waldo_variant_img,
                "waldo_scene_mask": waldo_scene_mask,
                "occ_mask": occ_mask,
                "occluded_fraction": float(occ_frac),
                "head_visible_fraction": float(head_visible_frac),
                "texture_score": float(texture_score),
                "score": float(score),
            })

        if not variant_results:
            continue

        chosen_variant = max(
            variant_results,
            key=lambda item: (
                item["head_visible_fraction"] >= MIN_HEAD_VISIBLE_FRAC,
                item["head_visible_fraction"],
                item["score"],
            ),
        )

        valid_candidates.append({
            "foot_x": int(foot_x),
            "foot_y": int(foot_y),
            "place_x": int(chosen_variant["place_x"]),
            "place_y": int(chosen_variant["place_y"]),
            "score": float(chosen_variant["score"]),
            "occluded_fraction": float(chosen_variant["occluded_fraction"]),
            "head_visible_fraction": float(chosen_variant["head_visible_fraction"]),
            "texture_score": float(chosen_variant["texture_score"]),
            "target_h": int(target_height),
            "waldo_layer_id": int(waldo_layer_id),
            "waldo_scaled": chosen_variant["waldo_scaled"],
            "rotation_degrees": int(chosen_variant["rotation_degrees"]),
            "occ_mask": chosen_variant["occ_mask"],
            "waldo_scene_mask": chosen_variant["waldo_scene_mask"],
        })

    if not valid_candidates:
        raise RuntimeError("Failed to find a valid Waldo placement.")

    valid_candidates_sorted = sorted(valid_candidates, key=lambda c: c["score"], reverse=True)
    top_k = valid_candidates_sorted[:min(top_k_random, len(valid_candidates_sorted))]
    best = random.choice(top_k)

    waldo_arr = np.array(best["waldo_scaled"])
    composed = paste_rgba(scene_np.copy(), waldo_arr, best["place_x"], best["place_y"])
    original_scene = scene_np.copy()
    composed[best["occ_mask"]] = original_scene[best["occ_mask"]]

    placement = {
        "foot": [best["foot_x"], best["foot_y"]],
        "top_left": [best["place_x"], best["place_y"]],
        "center": [
            best["place_x"] + waldo_arr.shape[1] / 2.0,
            best["place_y"] + waldo_arr.shape[0] / 2.0,
        ],
        "image_size_px": [int(w), int(h)],
        "waldo_size_px": [int(waldo_arr.shape[1]), int(waldo_arr.shape[0])],
        "rotation_degrees": best["rotation_degrees"],
        "used_preferred_support": used_preferred_support,
        "waldo_layer_id": best["waldo_layer_id"],
        "target_height_px": best["target_h"],
        "occluded_fraction": best["occluded_fraction"],
        "head_visible_fraction": best["head_visible_fraction"],
        "texture_score": best["texture_score"],
        "score": best["score"],
    }

    if save_debug:
        debug = scene_np.copy()
        for fx, fy in all_sampled_candidates:
            radius = 2
            debug[max(0, fy - radius):min(h, fy + radius + 1), max(0, fx - radius):min(w, fx + radius + 1), :3] = [255, 255, 0]
        for candidate in valid_candidates:
            fx, fy = candidate["foot_x"], candidate["foot_y"]
            radius = 3
            debug[max(0, fy - radius):min(h, fy + radius + 1), max(0, fx - radius):min(w, fx + radius + 1), :3] = [0, 255, 255]
        fy, fx = best["foot_y"], best["foot_x"]
        radius = 5
        debug[max(0, fy - radius):min(h, fy + radius + 1), max(0, fx - radius):min(w, fx + radius + 1), :3] = [255, 0, 0]
        Image.fromarray(debug).save(output_dir / f"{stem}_waldo_candidates_debug.png")

    Image.fromarray(composed).save(output_dir / f"{stem}_waldo_composited.png")

    serializable = dict(placement)
    serializable["occ_mask_file"] = f"{stem}_waldo_occ_mask.png"
    serializable["waldo_mask_file"] = f"{stem}_waldo_mask.png"
    Image.fromarray(best["occ_mask"].astype(np.uint8) * 255).save(output_dir / serializable["occ_mask_file"])
    Image.fromarray(best["waldo_scene_mask"].astype(np.uint8) * 255).save(output_dir / serializable["waldo_mask_file"])

    with open(output_dir / f"{stem}_waldo_placement.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    return {
        "placement": placement,
        "scene_image": Image.fromarray(scene_np),
        "composited_image": Image.fromarray(composed),
        "waldo_scaled": best["waldo_scaled"],
        "occ_mask": best["occ_mask"],
        "waldo_scene_mask": best["waldo_scene_mask"],
        "layer_entries": layer_entries,
        "image_size": (w, h),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Choose Waldo placement from layered scene outputs.")
    parser.add_argument("image", nargs="?", default="landscape.jpg", help="Background image path")
    parser.add_argument("--waldo", default="waldo.png", help="Path to Waldo image with alpha")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory containing scene analysis outputs")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = choose_waldo_placement(
        args.image,
        waldo_path=args.waldo,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print("Placement:")
    print(json.dumps(result["placement"], indent=2))


if __name__ == "__main__":
    main()
