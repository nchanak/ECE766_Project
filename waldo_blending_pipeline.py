import os
import argparse
import cv2
import numpy as np


# =========================================================
# Helper functions
# =========================================================

def overlay_waldo(background, waldo_rgba, x, y, scale=1.0):
    """
    Simple placement baseline.
    """
    h_w, w_w = waldo_rgba.shape[:2]
    new_w = int(w_w * scale)
    new_h = int(h_w * scale)
    waldo_resized = cv2.resize(waldo_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    waldo_rgb = waldo_resized[:, :, :3]
    alpha = waldo_resized[:, :, 3] / 255.0

    result = background.copy()
    h_b, w_b = background.shape[:2]

    if x >= w_b or y >= h_b:
        raise ValueError("Chosen position is completely outside the background image.")

    x_end = min(x + new_w, w_b)
    y_end = min(y + new_h, h_b)

    waldo_crop_w = x_end - x
    waldo_crop_h = y_end - y

    waldo_rgb = waldo_rgb[:waldo_crop_h, :waldo_crop_w]
    alpha = alpha[:waldo_crop_h, :waldo_crop_w]

    roi = result[y:y_end, x:x_end]

    for c in range(3):
        roi[:, :, c] = alpha * waldo_rgb[:, :, c] + (1 - alpha) * roi[:, :, c]

    result[y:y_end, x:x_end] = roi.astype(np.uint8)
    return result


def prepare_waldo(background, waldo_rgba, x, y, scale=1.0):
    """
    Resizes Waldo and creates:
    - composited result
    - full-scene Waldo mask
    - resized Waldo RGB
    - resized Waldo alpha mask
    """
    h_w, w_w = waldo_rgba.shape[:2]
    new_w = int(w_w * scale)
    new_h = int(h_w * scale)

    waldo_resized = cv2.resize(waldo_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    waldo_rgb = waldo_resized[:, :, :3]
    alpha = waldo_resized[:, :, 3] / 255.0
    alpha_mask = (waldo_resized[:, :, 3] > 0).astype(np.uint8) * 255

    result = background.copy()
    h_b, w_b = background.shape[:2]

    if x >= w_b or y >= h_b:
        raise ValueError("Chosen position is completely outside the background image.")

    x_end = min(x + new_w, w_b)
    y_end = min(y + new_h, h_b)

    waldo_crop_w = x_end - x
    waldo_crop_h = y_end - y

    waldo_rgb = waldo_rgb[:waldo_crop_h, :waldo_crop_w]
    alpha = alpha[:waldo_crop_h, :waldo_crop_w]
    alpha_mask = alpha_mask[:waldo_crop_h, :waldo_crop_w]

    roi = result[y:y_end, x:x_end]

    for c in range(3):
        roi[:, :, c] = (alpha * waldo_rgb[:, :, c] + (1 - alpha) * roi[:, :, c]).astype(np.uint8)

    result[y:y_end, x:x_end] = roi

    full_mask = np.zeros((h_b, w_b), dtype=np.uint8)
    full_mask[y:y_end, x:x_end] = alpha_mask

    return result, full_mask, waldo_rgb, alpha_mask


def prepare_waldo_alpha(waldo_rgba, scale):
    h, w = waldo_rgba.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)

    waldo_resized = cv2.resize(waldo_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
    waldo_rgb = waldo_resized[:, :, :3]
    waldo_alpha = waldo_resized[:, :, 3]

    return waldo_rgb, waldo_alpha


def match_color_local(waldo_rgb, background, x, y):
    """
    Matches Waldo's color to the local background patch.
    """
    h_w, w_w = waldo_rgb.shape[:2]
    h_b, w_b = background.shape[:2]

    x_end = min(x + w_w, w_b)
    y_end = min(y + h_w, h_b)

    bg_patch = background[y:y_end, x:x_end]
    bg_patch = cv2.resize(bg_patch, (w_w, h_w))

    waldo = waldo_rgb.astype(np.float32)
    bg = bg_patch.astype(np.float32)

    for c in range(3):
        waldo_mean, waldo_std = waldo[:, :, c].mean(), waldo[:, :, c].std()
        bg_mean, bg_std = bg[:, :, c].mean(), bg[:, :, c].std()

        if waldo_std < 1e-6:
            waldo_std = 1.0
        if bg_std < 1e-6:
            bg_std = 1.0

        waldo[:, :, c] = (waldo[:, :, c] - waldo_mean) / waldo_std
        waldo[:, :, c] = waldo[:, :, c] * bg_std + bg_mean

    return np.clip(waldo, 0, 255).astype(np.uint8)


def estimate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def match_sharpness_and_noise(waldo_rgb, background, x, y, add_noise=True):
    """
    Match Waldo's sharpness to the local background patch.
    """
    h_w, w_w = waldo_rgb.shape[:2]
    h_b, w_b = background.shape[:2]

    x_end = min(x + w_w, w_b)
    y_end = min(y + h_w, h_b)

    bg_patch = background[y:y_end, x:x_end]
    bg_patch = cv2.resize(bg_patch, (w_w, h_w))

    waldo = waldo_rgb.copy()

    waldo_sharpness = estimate_sharpness(waldo)
    bg_sharpness = estimate_sharpness(bg_patch)

    if waldo_sharpness > bg_sharpness * 1.5:
        ratio = waldo_sharpness / max(bg_sharpness, 1e-6)

        if ratio < 1.5:
            ksize = 3
        elif ratio < 2.5:
            ksize = 5
        else:
            ksize = 7

        waldo = cv2.GaussianBlur(waldo, (ksize, ksize), 0)

    if add_noise:
        bg_gray = cv2.cvtColor(bg_patch, cv2.COLOR_BGR2GRAY)
        bg_noise_level = np.std(
            bg_gray.astype(np.float32)
            - cv2.GaussianBlur(bg_gray, (3, 3), 0).astype(np.float32)
        )

        if bg_noise_level > 3:
            noise = np.random.normal(0, bg_noise_level * 0.15, waldo.shape).astype(np.float32)
            waldo = np.clip(waldo.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return waldo


def feather_mask(mask, blur_ksize=7):
    """
    Softens mask edges for smoother alpha blending.
    """
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    return cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)


def alpha_blend(background, waldo_rgb, waldo_alpha, x, y):
    """
    Paste Waldo onto background using alpha blending.
    """
    result = background.copy()

    h_b, w_b = background.shape[:2]
    h_w, w_w = waldo_rgb.shape[:2]

    if x >= w_b or y >= h_b:
        raise ValueError("Waldo position is outside the background image.")

    x_end = min(x + w_w, w_b)
    y_end = min(y + h_w, h_b)

    crop_w = x_end - x
    crop_h = y_end - y

    if crop_w <= 0 or crop_h <= 0:
        raise ValueError("Waldo is completely outside the background.")

    waldo_rgb = waldo_rgb[:crop_h, :crop_w]
    waldo_alpha = waldo_alpha[:crop_h, :crop_w]

    roi = result[y:y_end, x:x_end]

    alpha = waldo_alpha.astype(np.float32) / 255.0
    alpha = np.expand_dims(alpha, axis=2)

    blended = alpha * waldo_rgb.astype(np.float32) + (1 - alpha) * roi.astype(np.float32)
    result[y:y_end, x:x_end] = np.clip(blended, 0, 255).astype(np.uint8)

    return result


def seamless_clone_waldo(background, waldo_rgb, waldo_mask, x, y, mode="mixed"):
    """
    Paste Waldo into the background using OpenCV seamless cloning.
    mode: 'normal' or 'mixed'
    """
    h_w, w_w = waldo_rgb.shape[:2]
    h_b, w_b = background.shape[:2]

    x_end = min(x + w_w, w_b)
    y_end = min(y + h_w, h_b)

    crop_w = x_end - x
    crop_h = y_end - y

    if crop_w <= 0 or crop_h <= 0:
        raise ValueError("Waldo placement is outside the background.")

    waldo_rgb = waldo_rgb[:crop_h, :crop_w]
    waldo_mask = waldo_mask[:crop_h, :crop_w]

    waldo_mask = (waldo_mask > 127).astype(np.uint8) * 255
    center = (x + crop_w // 2, y + crop_h // 2)

    clone_flag = cv2.MIXED_CLONE if mode.lower() == "mixed" else cv2.NORMAL_CLONE
    result = cv2.seamlessClone(waldo_rgb, background, waldo_mask, center, clone_flag)

    return result


def create_mask_visualizations(background, full_mask):
    mask_vis = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR)
    mask_vis[full_mask > 0] = (0, 255, 0)
    overlay_vis = cv2.addWeighted(background, 0.5, mask_vis, 0.5, 0)
    return mask_vis, overlay_vis


def blend_waldo_into_scene(
    background_rgb: np.ndarray,
    waldo_rgba: np.ndarray,
    x: int,
    y: int,
    *,
    apply_color_match: bool = True,
    apply_sharpness_match: bool = True,
    add_noise: bool = True,
    feather: bool = True,
    feather_ksize: int = 5,
):
    """
    Blend a pre-scaled Waldo RGBA cutout into a scene array.

    Returns a dict with the composited scene and intermediate masks that can be
    combined with scene-layer occlusion logic.
    """
    if background_rgb.ndim != 3 or background_rgb.shape[2] != 3:
        raise ValueError("background_rgb must be an RGB image array.")
    if waldo_rgba.ndim != 3 or waldo_rgba.shape[2] != 4:
        raise ValueError("waldo_rgba must be an RGBA image array.")

    background_bgr = cv2.cvtColor(background_rgb, cv2.COLOR_RGB2BGR)
    waldo_bgra = cv2.cvtColor(waldo_rgba, cv2.COLOR_RGBA2BGRA)
    waldo_rgb = waldo_bgra[:, :, :3]
    waldo_alpha = waldo_bgra[:, :, 3]

    if apply_color_match:
        waldo_rgb = match_color_local(waldo_rgb, background_bgr, x, y)
    if apply_sharpness_match:
        waldo_rgb = match_sharpness_and_noise(waldo_rgb, background_bgr, x, y, add_noise=add_noise)

    blend_alpha = feather_mask(waldo_alpha.copy(), blur_ksize=feather_ksize) if feather else waldo_alpha.copy()
    composited_bgr = alpha_blend(background_bgr, waldo_rgb, blend_alpha, x, y)

    return {
        "background_bgr": background_bgr,
        "waldo_rgb_bgr": waldo_rgb,
        "waldo_alpha": waldo_alpha,
        "blend_alpha": blend_alpha,
        "composited_rgb": cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB),
        "waldo_rgb": cv2.cvtColor(waldo_rgb, cv2.COLOR_BGR2RGB),
    }


# =========================================================
# Pipeline
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_pipeline(
    background_path,
    waldo_path,
    output_dir,
    x,
    y,
    scale,
    save_step1=True,
    save_masks=True,
    apply_color_match=True,
    apply_sharpness_match=True,
    add_noise=True,
    blend_method="alpha",
    feather=True,
    feather_ksize=5,
    poisson_mode="mixed",
):
    ensure_dir(output_dir)

    background = cv2.imread(background_path)
    waldo = cv2.imread(waldo_path, cv2.IMREAD_UNCHANGED)

    if background is None:
        raise FileNotFoundError(f"Could not load background image: {background_path}")
    if waldo is None:
        raise FileNotFoundError(f"Could not load Waldo image: {waldo_path}")
    if waldo.shape[2] != 4:
        raise ValueError("Waldo image must have an alpha channel.")

    step1_result = overlay_waldo(background, waldo, x, y, scale)
    _, full_mask, waldo_rgb, alpha_mask = prepare_waldo(background, waldo, x, y, scale)

    if save_step1:
        cv2.imwrite(os.path.join(output_dir, "step1_waldo_placed.png"), step1_result)

    if save_masks:
        cv2.imwrite(os.path.join(output_dir, "step2_full_mask.png"), full_mask)
        cv2.imwrite(os.path.join(output_dir, "step2_local_mask.png"), alpha_mask)
        _, overlay_vis = create_mask_visualizations(background, full_mask)
        cv2.imwrite(os.path.join(output_dir, "step2_mask_overlay.png"), overlay_vis)

    if apply_color_match:
        waldo_rgb = match_color_local(waldo_rgb, background, x, y)
        cv2.imwrite(os.path.join(output_dir, "step3_color_matched.png"), waldo_rgb)

    if apply_sharpness_match:
        waldo_rgb = match_sharpness_and_noise(
            waldo_rgb, background, x, y, add_noise=add_noise
        )
        cv2.imwrite(os.path.join(output_dir, "step4_sharpness_noise_matched.png"), waldo_rgb)

    if blend_method == "alpha":
        waldo_alpha = alpha_mask.copy()
        if feather:
            waldo_alpha = feather_mask(waldo_alpha, blur_ksize=feather_ksize)
        final_result = alpha_blend(background, waldo_rgb, waldo_alpha, x, y)
        cv2.imwrite(os.path.join(output_dir, "step5_final_alpha.png"), final_result)

    elif blend_method == "poisson":
        final_result = seamless_clone_waldo(
            background, waldo_rgb, alpha_mask, x, y, mode=poisson_mode
        )
        cv2.imwrite(os.path.join(output_dir, "step5_final_poisson.png"), final_result)

    else:
        raise ValueError("blend_method must be 'alpha' or 'poisson'.")

    return final_result


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Waldo blending pipeline")

    parser.add_argument("--background", type=str, required=True, help="Path to background image")
    parser.add_argument("--waldo", type=str, required=True, help="Path to Waldo RGBA cutout")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for outputs")

    parser.add_argument("--x", type=int, required=True, help="Top-left x position")
    parser.add_argument("--y", type=int, required=True, help="Top-left y position")
    parser.add_argument("--scale", type=float, default=0.1, help="Waldo scale factor")

    parser.add_argument("--no_step1", action="store_true", help="Disable saving step 1 output")
    parser.add_argument("--no_masks", action="store_true", help="Disable saving masks")
    parser.add_argument("--no_color_match", action="store_true", help="Disable local color matching")
    parser.add_argument("--no_sharpness_match", action="store_true", help="Disable sharpness/noise matching")
    parser.add_argument("--no_noise", action="store_true", help="Disable noise addition")

    parser.add_argument("--blend_method", choices=["alpha", "poisson"], default="alpha")
    parser.add_argument("--no_feather", action="store_true", help="Disable mask feathering for alpha blend")
    parser.add_argument("--feather_ksize", type=int, default=5, help="Feather kernel size")
    parser.add_argument("--poisson_mode", choices=["normal", "mixed"], default="mixed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_pipeline(
        background_path=args.background,
        waldo_path=args.waldo,
        output_dir=args.output_dir,
        x=args.x,
        y=args.y,
        scale=args.scale,
        save_step1=not args.no_step1,
        save_masks=not args.no_masks,
        apply_color_match=not args.no_color_match,
        apply_sharpness_match=not args.no_sharpness_match,
        add_noise=not args.no_noise,
        blend_method=args.blend_method,
        feather=not args.no_feather,
        feather_ksize=args.feather_ksize,
        poisson_mode=args.poisson_mode,
    )
