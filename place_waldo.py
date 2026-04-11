import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


# hardcoded path rn
IMAGE_PATH = "landscape.jpg"
WALDO_PATH = "waldo.png"
OUTPUT_DIR = Path("layered_output")

# temporary classes for waldo to stand
SUPPORT_CLASSES = {"ground", "road", "earth", "grass", "rock", "plant", "vegetation", "mountain", "tree", "cliff"}

# Waldo ain't jesus
FORBIDDEN_CLASSES = {"sky", "water"}

def resize_image_keep_aspect(image: Image.Image, max_size: int) -> Image.Image:
    w, h = image.size
    scale = max_size / max(w, h)
    if scale < 1:
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image

# good stuff to hide behind
def is_preferred_occluder(name: str) -> bool:
    name = name.lower()
    keywords = [
        "tree", "rock", "boulder", "fence", "pole", "sign",
        "car", "truck", "bus", "van", "building", "wall", "facade", "cliff", "skyscraper"
    ]
    return any(k in name for k in keywords)

# scale waldo based on depth
MIN_SCALE = 0.08
MAX_SCALE = 0.28

# how much to hide waldo
MIN_OCCLUDED_FRAC = 0.4
MAX_OCCLUDED_FRAC = 0.75

# spots to look at
NUM_POSITION_SAMPLES = 500
SEED = 34

# spots to pick from
TOP_K_RANDOM = 25



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


def compute_support_mask(layer_entries, mask_root: Path, H: int, W: int, stem: str) -> np.ndarray:
    support = np.zeros((H, W), dtype=bool)
    forbidden = np.zeros((H, W), dtype=bool)

    for entry in layer_entries:
        name = entry["name"]
        mask = None

        mask_file = entry.get("mask_file")
        if mask_file:
            p = mask_root / mask_file
            if p.exists():
                mask = load_mask(p)

        # Fallback: semantic masks saved by dino_sam_semantic_depth.py
        if mask is None:
            semantic_path = mask_root / "out_masks" / f"{stem}_semantic_{name}.png"
            if semantic_path.exists():
                mask = load_mask(semantic_path)

        if mask is None:
            continue

        if name in SUPPORT_CLASSES:
            support |= mask
        if name in FORBIDDEN_CLASSES:
            forbidden |= mask

    return support & (~forbidden)


def percentile_normalize(arr: np.ndarray) -> np.ndarray:
    lo = np.percentile(arr, 2)
    hi = np.percentile(arr, 98)
    arr = np.clip((arr - lo) / max(1e-8, hi - lo), 0.0, 1.0)
    return arr


def choose_scale_from_depth(depth_map_u8: np.ndarray, foot_x: int, foot_y: int, scene_h: int) -> int:
    d = depth_map_u8[foot_y, foot_x] / 255.0

    # depth 0 = far background, depth 1 = near foreground
    rel = MIN_SCALE + d * (MAX_SCALE - MIN_SCALE)
    return max(12, int(scene_h * rel))


def paste_rgba(base_rgba: np.ndarray, overlay_rgba: np.ndarray, x: int, y: int):
    H, W = base_rgba.shape[:2]
    h, w = overlay_rgba.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)

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

    out = np.concatenate([out_rgb, out_a], axis=-1)
    base_rgba[y1:y2, x1:x2] = np.clip(out, 0, 255).astype(np.uint8)
    return base_rgba


def build_front_occlusion_mask(
    waldo_mask_scene: np.ndarray,
    waldo_layer_id: int,
    layer_entries,
    mask_root: Path,
    H: int,
    W: int,
):
    occ = np.zeros((H, W), dtype=bool)

    for entry in layer_entries:
        if entry["layer_id"] <= waldo_layer_id:
            continue

        name = entry["name"]
        if not is_preferred_occluder(name):
            continue

        mask_file = entry.get("mask_file")
        if not mask_file:
            continue

        obj_mask = load_mask(mask_root / mask_file)
        occ |= (obj_mask & waldo_mask_scene)

    return occ


def estimate_waldo_layer_id(foot_x: int, foot_y: int, layer_map: np.ndarray, layer_entries) -> int:
    local_layer = int(layer_map[foot_y, foot_x])
    if local_layer > 0:
        return local_layer

    H, W = layer_map.shape
    r = 8
    x1 = max(0, foot_x - r)
    x2 = min(W, foot_x + r + 1)
    y1 = max(0, foot_y - r)
    y2 = min(H, foot_y + r + 1)
    patch = layer_map[y1:y2, x1:x2]
    vals = patch[patch > 0]
    if vals.size:
        return int(np.median(vals))

    if layer_entries:
        return int(np.median([e["layer_id"] for e in layer_entries]))
    return 1


def sample_candidate_points(mask: np.ndarray, n: int):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []

    idx = np.random.choice(len(xs), size=min(n, len(xs)), replace=False)
    return list(zip(xs[idx], ys[idx]))



def main():
    random.seed(SEED)
    np.random.seed(SEED)

    image_path = Path(IMAGE_PATH)
    stem = image_path.stem

    scene_img = Image.open(image_path).convert("RGBA")
    waldo_img = Image.open(WALDO_PATH).convert("RGBA")

    # match res of image
    with open(OUTPUT_DIR / f"{stem}_layers.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    target_w, target_h = meta["image_size"]
    scene_img = scene_img.resize((target_w, target_h), Image.LANCZOS)

    scene_np = np.array(scene_img)
    H, W = scene_np.shape[:2]

    layer_entries = meta["layers"]

    with open(OUTPUT_DIR / f"{stem}_layers.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    layer_entries = meta["layers"]

    layer_map = np.array(Image.open(OUTPUT_DIR / f"{stem}_layer_map.png").convert("L"))
    depth_map_u8 = np.array(Image.open(OUTPUT_DIR / f"{stem}_depth_map.png").convert("L"))

    mask_root = OUTPUT_DIR

    support_mask = compute_support_mask(layer_entries, mask_root, H, W, stem)

    candidate_mask = support_mask

    candidates = sample_candidate_points(candidate_mask, NUM_POSITION_SAMPLES)
    if not candidates:
        raise RuntimeError("No valid candidate support pixels found for Waldo placement.")

    all_sampled_candidates = []
    valid_candidates = []
    
    best = None
    best_score = -1e9

    waldo_alpha = np.array(waldo_img)[..., 3]
    waldo_box = alpha_bbox(waldo_alpha)
    if waldo_box is None:
        raise RuntimeError("Waldo image has empty alpha.")
    wx1, wy1, wx2, wy2 = waldo_box
    waldo_trim = waldo_img.crop((wx1, wy1, wx2 + 1, wy2 + 1))

    for foot_x, foot_y in candidates:
        all_sampled_candidates.append((int(foot_x), int(foot_y)))
        target_h = choose_scale_from_depth(depth_map_u8, foot_x, foot_y, H)
        waldo_scaled = resize_rgba_keep_aspect(waldo_trim, target_h)
        waldo_arr = np.array(waldo_scaled)
        alpha = waldo_arr[..., 3] > 0

        h, w = alpha.shape

        # place by feet
        place_x = foot_x - w // 2
        place_y = foot_y - h + 1

        if place_x < -w // 3 or place_y < -h // 3 or place_x >= W or place_y >= H:
            continue

        waldo_scene_mask = np.zeros((H, W), dtype=bool)

        x1 = max(0, place_x)
        y1 = max(0, place_y)
        x2 = min(W, place_x + w)
        y2 = min(H, place_y + h)
        if x1 >= x2 or y1 >= y2:
            continue

        ox1 = x1 - place_x
        oy1 = y1 - place_y
        ox2 = ox1 + (x2 - x1)
        oy2 = oy1 + (y2 - y1)

        waldo_scene_mask[y1:y2, x1:x2] = alpha[oy1:oy2, ox1:ox2]

        forbidden_overlap = 0
        total = waldo_scene_mask.sum()
        if total == 0:
            continue

        waldo_layer_id = estimate_waldo_layer_id(foot_x, foot_y, layer_map, layer_entries)

        occ_mask = build_front_occlusion_mask(
            waldo_scene_mask, waldo_layer_id, layer_entries, mask_root, H, W
        )

        occ_frac = occ_mask.sum() / total

        # don't allow above max occlude
        if occ_frac > MAX_OCCLUDED_FRAC:
            continue

        # soft lower bound
        if occ_frac < MIN_OCCLUDED_FRAC:
            occ_penalty = abs(occ_frac - 1.0)
        else:
            occ_penalty = 0.0

        # prefer 45% occluded, slightly low part of image, far away depth, penalize low occlusion
        depth_score = depth_map_u8[foot_y, foot_x] / 255.0
        y_score = foot_y / H
        score = (
            0.3 * y_score
            + 0.8 * (1.0 - abs(occ_frac - 0.45))
            + 0.3 * (1.0 - depth_score)
            - 1.2 * occ_penalty
        )
        valid_candidates.append({
            "foot_x": int(foot_x),
            "foot_y": int(foot_y),
            "place_x": int(place_x),
            "place_y": int(place_y),
            "score": float(score),
            "occluded_fraction": float(occ_frac),
            "target_h": int(target_h),
            "waldo_layer_id": int(waldo_layer_id),
            "waldo_scaled": waldo_scaled,
            "occ_mask": occ_mask,
            "waldo_scene_mask": waldo_scene_mask,
        })
    if not valid_candidates:
        raise RuntimeError("Failed to find a valid Waldo placement.")

    valid_candidates_sorted = sorted(
        valid_candidates,
        key=lambda c: c["score"],
        reverse=True
    )

    top_k = valid_candidates_sorted[:min(TOP_K_RANDOM, len(valid_candidates_sorted))]
    best = random.choice(top_k)
    best_score = best["score"]

    waldo_scaled = best["waldo_scaled"]
    waldo_arr = np.array(waldo_scaled)
    h, w = waldo_arr.shape[:2]

    # apply walod
    composed = scene_np.copy()
    composed = paste_rgba(composed, waldo_arr, best["place_x"], best["place_y"])

    # Then place occluders over
    occ_mask = best["occ_mask"]
    original_scene = scene_np.copy()
    composed[occ_mask] = original_scene[occ_mask]

    # save
    out_img = Image.fromarray(composed)
    out_path = OUTPUT_DIR / f"{stem}_waldo_composited.png"
    out_img.save(out_path)

    # debug
    debug = scene_np.copy()

    # not valid candidate points yellow
    for fx, fy in all_sampled_candidates:
        r = 2
        debug[max(0, fy-r):min(H, fy+r+1), max(0, fx-r):min(W, fx+r+1), :3] = [255, 255, 0]

    # valid candidates cyab
    for cand in valid_candidates:
        fx, fy = cand["foot_x"], cand["foot_y"]
        r = 3
        debug[max(0, fy-r):min(H, fy+r+1), max(0, fx-r):min(W, fx+r+1), :3] = [0, 255, 255]

    # chosesn candidate red from random Top_K
    fy, fx = best["foot_y"], best["foot_x"]
    r = 5
    debug[max(0, fy-r):min(H, fy+r+1), max(0, fx-r):min(W, fx+r+1), :3] = [255, 0, 0]

    Image.fromarray(debug).save(OUTPUT_DIR / f"{stem}_waldo_candidates_debug.png")

    print("Saved:", out_path)
    print("Placement:")
    print(json.dumps({
        "foot": [best["foot_x"], best["foot_y"]],
        "top_left": [best["place_x"], best["place_y"]],
        "waldo_layer_id": best["waldo_layer_id"],
        "target_height_px": best["target_h"],
        "occluded_fraction": best["occluded_fraction"],
    }, indent=2))


if __name__ == "__main__":
    main()