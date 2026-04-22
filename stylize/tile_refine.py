"""
Overlapping tile refinement for Pipeline C: feather-blend local passes.

Refine policy: uniform scans all tiles (with optional flat skip). selective keeps only
high-priority tiles (edge energy x variance, optional center bias) to spend budget on busy / mid-ground
regions instead of sky or uniform texture.

"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Callable, Literal, Optional

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from stylize.pipeline_c import StylizeConfigC, WaldoStylizerPipelineC

logger = logging.getLogger(__name__)

RefinePolicy = Literal["uniform", "selective"]


def _tile_origins(dim: int, tile: int, overlap: int) -> list[int]:
    """Start positions along one axis so the last tile touches the far edge."""
    if dim <= 0 or tile <= 0:
        return [0]
    if dim <= tile:
        return [0]
    step = max(1, tile - overlap)
    positions = list(range(0, dim - tile + 1, step))
    last = dim - tile
    if positions[-1] != last:
        positions.append(last)
    out: list[int] = []
    for p in positions:
        if not out or out[-1] != p:
            out.append(p)
    return out


def iter_tiles(
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    """Return list of (x0, y0, x1, y1) boxes; may use smaller than tile_size at right/bottom."""
    xs = _tile_origins(width, tile_size, overlap)
    ys = _tile_origins(height, tile_size, overlap)
    boxes: list[tuple[int, int, int, int]] = []
    for y0 in ys:
        for x0 in xs:
            x1 = min(x0 + tile_size, width)
            y1 = min(y0 + tile_size, height)
            boxes.append((x0, y0, x1, y1))
    return boxes


def hann_weight(h: int, w: int) -> np.ndarray:
    """2D separable Hann window, high in center and zero at edges (feather)."""
    if h < 2 or w < 2:
        return np.ones((h, w), dtype=np.float64)
    wy = np.hanning(h)
    wx = np.hanning(w)
    return np.outer(wy, wx).astype(np.float64)


def _patch_std_grayscale(patch: Image.Image) -> float:
    g = np.array(patch.convert("L"), dtype=np.float64)
    return float(g.std())


def tile_priority_score(
    patch: Image.Image,
    box: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    *,
    center_bias: float = 0.0,
) -> float:
    """Higher ≈ more edge/texture activity (crowd-like) and worth a refine pass; flat sky → ~0."""
    arr = np.asarray(patch.convert("L"), dtype=np.float64)
    if arr.size < 16:
        return 0.0
    sd = float(arr.std())
    if sd < 2.5:
        return 0.0

    gx = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_64F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    edge_mean = float(edge.mean()) / (255.0 + 1e-6)
    score = edge_mean * float(np.log1p(sd)) * 12.0

    if center_bias > 0:
        x0, y0, x1, y1 = box
        cx = (x0 + x1) * 0.5 / max(img_w, 1)
        cy = (y0 + y1) * 0.5 / max(img_h, 1)
        dx, dy = cx - 0.5, cy - 0.5
        d2 = dx * dx + dy * dy
        score *= 1.0 + center_bias * float(np.exp(-d2 * 6.0))

    return float(score)


def _selective_tile_boxes(
    base_image: Image.Image,
    structure_source: Optional[Image.Image],
    all_boxes: list[tuple[int, int, int, int]],
    *,
    flat_std_threshold: float,
    keep_fraction: float,
    min_score: float,
    center_bias: float,
) -> list[tuple[int, int, int, int]]:
    """Drop flat tiles, then keep top ``keep_fraction`` by priority (on structure crop if present)."""
    w, h = base_image.size
    scored: list[tuple[tuple[int, int, int, int], float]] = []
    for box in all_boxes:
        probe = (
            structure_source.crop(box)
            if structure_source is not None
            else base_image.crop(box)
        )
        if _patch_std_grayscale(probe) < flat_std_threshold:
            continue
        sc = tile_priority_score(
            probe,
            box,
            w,
            h,
            center_bias=center_bias,
        )
        if sc < min_score:
            continue
        scored.append((box, sc))

    if not scored:
        logger.warning("Selective mode: no tiles passed flat/min-score filter")
        return []

    scores = np.array([s for _, s in scored], dtype=np.float64)
    cut = float(np.percentile(scores, (1.0 - keep_fraction) * 100.0))
    chosen = [b for b, s in scored if s >= cut]

    if not chosen:
        best = max(scored, key=lambda x: x[1])
        chosen = [best[0]]
        logger.warning("Selective mode: percentile empty — kept single best tile")

    logger.info(
        "Selective refine: %s / %s tiles (from %s candidates)",
        len(chosen),
        len(scored),
        len(all_boxes),
    )
    return chosen


def refine_image_tiled(
    base_image: Image.Image,
    stylizer: "WaldoStylizerPipelineC",
    refine_config: "StylizeConfigC",
    *,
    structure_source: Optional[Image.Image] = None,
    tile_size: int = 256,
    overlap: int = 64,
    refine_policy: RefinePolicy = "selective",
    selective_keep_fraction: float = 0.5,
    selective_center_bias: float = 0.2,
    selective_min_score: float = 0.0,
    skip_flat_tiles: bool = True,
    flat_std_threshold: float = 10.0,
    patch_postprocess: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Image.Image:
    """
    Crop overlapping patches; dual-source uses ``control_source=orig_crop`` when possible.

    ``refine_policy``: ``uniform`` = every tile (minus flat skip); ``selective`` = priority subset.
    """
    base_image = base_image.convert("RGB")
    if structure_source is not None:
        structure_source = structure_source.convert("RGB")
        if structure_source.size != base_image.size:
            raise ValueError(
                f"structure_source size {structure_source.size} must match base_image {base_image.size}"
            )

    w, h = base_image.size
    all_boxes = iter_tiles(w, h, tile_size, overlap)
    flat_in_loop = skip_flat_tiles

    if refine_policy == "selective":
        boxes = _selective_tile_boxes(
            base_image,
            structure_source,
            all_boxes,
            flat_std_threshold=flat_std_threshold,
            keep_fraction=selective_keep_fraction,
            min_score=selective_min_score,
            center_bias=selective_center_bias,
        )
        if len(boxes) == 0:
            logger.warning("Selective produced no tiles — falling back to uniform + flat skip")
            boxes = all_boxes
            flat_in_loop = True
        else:
            flat_in_loop = False
    else:
        boxes = all_boxes

    accum = np.zeros((h, w, 3), dtype=np.float64)
    wsum = np.zeros((h, w), dtype=np.float64)

    n = len(boxes)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        if progress_cb:
            progress_cb(i + 1, n)
        base_crop = base_image.crop((x0, y0, x1, y1))
        orig_crop = (
            structure_source.crop((x0, y0, x1, y1)) if structure_source is not None else None
        )
        flat_probe = orig_crop if orig_crop is not None else base_crop
        if flat_in_loop and _patch_std_grayscale(flat_probe) < flat_std_threshold:
            logger.debug("Skip flat tile (%s,%s)-(%s,%s)", x0, y0, x1, y1)
            continue

        cfg = (
            refine_config
            if patch_postprocess
            else replace(refine_config, postprocess_enabled=False)
        )

        if orig_crop is not None:
            refined = stylizer.stylize(base_crop, config=cfg, control_source=orig_crop)
        else:
            refined = stylizer.stylize(base_crop, config=cfg)
        refined = refined.convert("RGB").resize(base_crop.size, Image.Resampling.LANCZOS)

        ph, pw = y1 - y0, x1 - x0
        weight = hann_weight(ph, pw)

        arr = np.asarray(refined, dtype=np.float64)
        accum[y0:y1, x0:x1] += arr * weight[..., np.newaxis]
        wsum[y0:y1, x0:x1] += weight

    if float(wsum.max()) < 1e-6:
        logger.warning("All tiles skipped — returning base image unchanged")
        return base_image

    wsum = np.maximum(wsum, 1e-6)
    out = (accum / wsum[..., np.newaxis]).clip(0, 255).astype(np.uint8)
    blended = Image.fromarray(out, mode="RGB")

    mask = wsum <= 1e-5
    if mask.any():
        base_arr = np.asarray(base_image, dtype=np.float64)
        out_arr = np.asarray(blended, dtype=np.float64)
        out_arr[mask] = base_arr[mask]
        blended = Image.fromarray(out_arr.clip(0, 255).astype(np.uint8), mode="RGB")

    return blended
