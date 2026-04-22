"""
Pipeline C: preprocess → SoftEdge control map (PidiNet, safe) → ControlNet SoftEdge + Img2Img
→ Waldo merged weights → light post (sharpen / color).

ControlNet: lllyasviel/control_v11p_sd15_softedge (SD1.5)
Soft-edge conditioning map: controlnet_aux PidiNetDetector from lllyasviel/Annotators (see HF README).

Waldo checkpoint: same as pipeline_a (SanDiegoDude/WheresWaldoStyle).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageEnhance

from stylize.pipeline_a import resize_for_waldo

# Illustration-reconstruction wording (Pipeline C); keep photo-stylization prompts out of negatives.
# Previous generic presets (pipeline_a–style) kept for reference:
# DEFAULT_PROMPT_PREV = (
#     "(Wheres Waldo Style:1.1), sharp illustration, clear characters, "
#     "clean lineart, highly detailed crowd, no blur, crisp edges"
# )
DEFAULT_PROMPT_C = (
    "(Wheres Waldo Style:1.1), dense crowd illustration, children's book art, storybook clarity, "
    "clean readable composition, clear small figures, crisp outlines, busy but legible scene, "
    "coherent silhouettes, full illustration rebuild on the scene layout"
)
DEFAULT_NEGATIVE_C = (
    "photograph, photorealistic, camera, noisy, blurry faces, fused bodies, melted crowd, "
    "merged limbs, tangled people, messy edges, smeared details, low quality, watermark, text, "
    "ugly, deformed, oversharpen halos"
)

logger = logging.getLogger(__name__)

WALDO_REPO = "SanDiegoDude/WheresWaldoStyle"
WALDO_CKPT_FILENAME = "Wheres_Waldo_Style_14000step_merged.ckpt"
CONTROLNET_SOFTEDGE_ID = "lllyasviel/control_v11p_sd15_softedge"
ANNOTATORS_REPO = "lllyasviel/Annotators"


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _dtype_for(device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    return torch.float16


def preprocess_step1(
    image: Image.Image,
    max_long_edge: int,
    *,
    bilateral_d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
    use_bilateral: bool = True,
) -> Image.Image:
    """Resize (LANCZOS via resize_for_waldo), then optional light bilateral smoothing."""
    rgb = resize_for_waldo(image, max_long_edge=max_long_edge)
    if not use_bilateral:
        return rgb
    arr = np.asarray(rgb.convert("RGB"))
    # bilateralFilter expects uint8 BGR for cv2 if we use cv2 directly — RGB order works per channel
    smoothed = cv2.bilateralFilter(arr, bilateral_d, sigma_color, sigma_space)
    return Image.fromarray(smoothed)


def _get_pidi_softedge_detector():
    try:
        from controlnet_aux import PidiNetDetector
    except ImportError as e:
        raise ImportError(
            "Pipeline C requires controlnet_aux for SoftEdge maps. "
            "Install with: pip install controlnet_aux"
        ) from e
    return PidiNetDetector.from_pretrained(ANNOTATORS_REPO)


def make_softedge_control_image(
    rgb_image: Image.Image,
    *,
    safe: bool = True,
    detector=None,
    max_resolution: Optional[int] = None,
) -> Image.Image:
    """Soft-edge structure map for control_v11p_sd15_softedge (PidiNet, matches HF README).

    Must match ``init_image`` spatial size for ControlNet: PidiNet defaults (512) would shrink
    the map while Img2Img uses full ``init_image``, causing latent/control shape mismatch.
    We set detect/image resolution from the input size (optionally capped by ``max_resolution``).
    """
    proc = detector or _get_pidi_softedge_detector()
    w, h = rgb_image.size
    res = max(w, h)
    if max_resolution is not None:
        res = min(res, max_resolution)
    return proc(
        rgb_image.convert("RGB"),
        safe=safe,
        detect_resolution=res,
        image_resolution=res,
    )


def align_control_to_init(
    control_image: Image.Image,
    init_image: Image.Image,
) -> Image.Image:
    """Resize control map to exactly match ``init_image`` (fixes off-by-one / rounding)."""
    if control_image.size == init_image.size:
        return control_image
    return control_image.resize(init_image.size, Image.Resampling.LANCZOS)


def postprocess_step4(
    image: Image.Image,
    *,
    sharpen: float = 1.08,
    color: float = 1.02,
) -> Image.Image:
    """Mild sharpen + saturation tweak."""
    out = ImageEnhance.Sharpness(image).enhance(sharpen)
    out = ImageEnhance.Color(out).enhance(color)
    return out


@dataclass
class StylizeConfigC:
    """Phase-1 global defaults: stable layout/color/Waldo mood — not hyper-detailed everywhere."""

    prompt: str = DEFAULT_PROMPT_C
    negative_prompt: str = DEFAULT_NEGATIVE_C
    num_inference_steps: int = 30
    # Global pass: slightly conservative so ControlNet does not over-fit fine clutter
    strength: float = 0.28
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 0.40
    max_long_edge: int = 1280
    seed: Optional[int] = None
    # Step 1
    preprocess_use_bilateral: bool = True
    preprocess_bilateral_d: int = 9
    preprocess_sigma_color: float = 75.0
    preprocess_sigma_space: float = 75.0
    # PidiNet / soft-edge: default cap so distant crowds are not over-etched (768 or 1024 typical)
    softedge_safe: bool = True
    softedge_max_resolution: Optional[int] = 768
    # Step 4 (phase-1; tile phase usually disables per-patch and applies once on the blend)
    postprocess_enabled: bool = True
    post_sharpen: float = 1.06
    post_color: float = 1.02


def refine_stylize_config(phase1: StylizeConfigC) -> StylizeConfigC:
    """Phase-2 tile pass: control from preprocessed *original* (see refine_image_tiled); no per-patch post."""
    return replace(
        phase1,
        # Local semantic zoom + low softedge cap: emphasis on figures, not brick texture
        strength=0.38,
        controlnet_conditioning_scale=0.34,
        softedge_max_resolution=384,
        postprocess_enabled=False,
        post_sharpen=1.0,
        post_color=1.0,
    )


class WaldoStylizerPipelineC:
    """Lazy-loaded pipeline + lazy PidiNet detector."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        ckpt_path: Optional[Union[str, Path]] = None,
    ):
        self.device = device or _pick_device()
        self.dtype = _dtype_for(self.device)
        self._ckpt_path_explicit = Path(ckpt_path) if ckpt_path else None
        self._ckpt_path_resolved: Optional[Path] = None
        self._pipe = None
        self._softedge_detector = None

    def _resolve_ckpt_path(self) -> Path:
        if self._ckpt_path_resolved is not None:
            return self._ckpt_path_resolved
        if self._ckpt_path_explicit is not None:
            self._ckpt_path_resolved = self._ckpt_path_explicit
            return self._ckpt_path_resolved
        self._ckpt_path_resolved = Path(
            hf_hub_download(
                repo_id=WALDO_REPO,
                filename=WALDO_CKPT_FILENAME,
                repo_type="model",
            )
        )
        return self._ckpt_path_resolved

    def _get_detector(self):
        if self._softedge_detector is None:
            logger.info("Loading PidiNet soft-edge detector (Annotators, may take a moment)...")
            self._softedge_detector = _get_pidi_softedge_detector()
        return self._softedge_detector

    def _load_pipeline(self):
        if self._pipe is not None:
            return self._pipe

        from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
        from diffusers import UniPCMultistepScheduler

        logger.info("Loading ControlNet SoftEdge + Waldo checkpoint (may take a minute)...")
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_SOFTEDGE_ID,
            torch_dtype=self.dtype,
        )

        pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
            str(self._resolve_ckpt_path()),
            controlnet=controlnet,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        if self.device.type == "cuda":
            pipe.vae.enable_slicing()
            pipe.enable_model_cpu_offload()
        elif self.device.type == "mps":
            pipe.to("mps")
            pipe.enable_attention_slicing()
        else:
            pipe.to("cpu")

        self._pipe = pipe
        return pipe

    @torch.inference_mode()
    def stylize(
        self,
        image: Image.Image,
        config: Optional[StylizeConfigC] = None,
        *,
        control_source: Optional[Image.Image] = None,
    ) -> Image.Image:
        """Run Img2Img + SoftEdge ControlNet.

        Default: preprocess ``image``, build control from the same preprocessed frame (global pass).

        If ``control_source`` is set (e.g. photo crop aligned with a stylized ``image`` crop): PidiNet /
        ControlNet use **real structure** from ``control_source``, while ``image`` is the Img2Img init
        (e.g. Phase-1 Waldo base crop) — local rebuild instead of refining blur alone.
        """
        cfg = config or StylizeConfigC()

        if control_source is not None:
            init_image = image.convert("RGB")
            control_image = make_softedge_control_image(
                control_source.convert("RGB"),
                safe=cfg.softedge_safe,
                detector=self._get_detector(),
                max_resolution=cfg.softedge_max_resolution,
            )
            control_image = align_control_to_init(control_image, init_image)
        else:
            init_image = preprocess_step1(
                image,
                cfg.max_long_edge,
                bilateral_d=cfg.preprocess_bilateral_d,
                sigma_color=cfg.preprocess_sigma_color,
                sigma_space=cfg.preprocess_sigma_space,
                use_bilateral=cfg.preprocess_use_bilateral,
            )

            control_image = make_softedge_control_image(
                init_image,
                safe=cfg.softedge_safe,
                detector=self._get_detector(),
                max_resolution=cfg.softedge_max_resolution,
            )
            control_image = align_control_to_init(control_image, init_image)

        pipe = self._load_pipeline()
        generator = None
        if cfg.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(cfg.seed)

        result = pipe(
            prompt=cfg.prompt,
            negative_prompt=cfg.negative_prompt,
            image=init_image,
            control_image=control_image,
            num_inference_steps=cfg.num_inference_steps,
            strength=cfg.strength,
            guidance_scale=cfg.guidance_scale,
            controlnet_conditioning_scale=cfg.controlnet_conditioning_scale,
            generator=generator,
        )
        out = result.images[0]
        if cfg.postprocess_enabled:
            out = postprocess_step4(
                out,
                sharpen=cfg.post_sharpen,
                color=cfg.post_color,
            )
        return out


def stylize_image(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    config: Optional[StylizeConfigC] = None,
    ckpt_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Load image from disk, run Pipeline C, save; returns the output path."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path_in = Path(input_path)
    path_out = Path(output_path)
    path_out.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(path_in).convert("RGB")
    stylizer = WaldoStylizerPipelineC(ckpt_path=ckpt_path)
    out = stylizer.stylize(image, config=config)
    out.save(path_out)
    logger.info("Saved: %s", path_out)
    return path_out


def stylize_image_two_phase(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    phase1_config: Optional[StylizeConfigC] = None,
    refine_config: Optional[StylizeConfigC] = None,
    tile_size: int = 256,
    tile_overlap: int = 64,
    refine_policy: str = "selective",
    selective_keep_fraction: float = 0.5,
    selective_center_bias: float = 0.2,
    selective_min_score: float = 0.0,
    skip_flat_tiles: bool = True,
    flat_std_threshold: float = 10.0,
    final_postprocess: bool = True,
    final_sharpen: float = 1.04,
    final_color: float = 1.01,
    ckpt_path: Optional[Union[str, Path]] = None,
    skip_phase2: bool = False,
    phase1_output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Phase 1: global Pipeline C (stable base). Phase 2: tiles use **preprocessed original** crops for
    SoftEdge (structure) and **base** crops for Img2Img (style) — see ``refine_image_tiled``.
    Optional ``phase1_output_path`` saves the phase-1 base for inspection.
    """
    from stylize.tile_refine import refine_image_tiled

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path_in = Path(input_path)
    path_out = Path(output_path)
    path_out.parent.mkdir(parents=True, exist_ok=True)

    p1 = phase1_config or StylizeConfigC()
    image = Image.open(path_in).convert("RGB")
    stylizer = WaldoStylizerPipelineC(ckpt_path=ckpt_path)

    # Same frame as inside stylize(): needed so phase-2 tiles can take structure from *photo*, init from base.
    preprocessed_orig = preprocess_step1(
        image,
        p1.max_long_edge,
        bilateral_d=p1.preprocess_bilateral_d,
        sigma_color=p1.preprocess_sigma_color,
        sigma_space=p1.preprocess_sigma_space,
        use_bilateral=p1.preprocess_use_bilateral,
    )

    base = stylizer.stylize(image, config=p1)
    if preprocessed_orig.size != base.size:
        logger.warning(
            "preprocessed_orig %s != base %s; phase-2 structure/init alignment may be wrong",
            preprocessed_orig.size,
            base.size,
        )
    if phase1_output_path is not None:
        p1p = Path(phase1_output_path)
        p1p.parent.mkdir(parents=True, exist_ok=True)
        base.save(p1p)
        logger.info("Saved phase-1 base: %s", p1p)

    if skip_phase2:
        out = base
    else:
        rcfg = refine_config if refine_config is not None else refine_stylize_config(p1)

        def _progress(done: int, total: int) -> None:
            if done == 1 or done == total or done % max(1, total // 10) == 0:
                logger.info("Tile refine %s / %s", done, total)

        blended = refine_image_tiled(
            base,
            stylizer,
            rcfg,
            structure_source=preprocessed_orig,
            tile_size=tile_size,
            overlap=tile_overlap,
            refine_policy=refine_policy,
            selective_keep_fraction=selective_keep_fraction,
            selective_center_bias=selective_center_bias,
            selective_min_score=selective_min_score,
            skip_flat_tiles=skip_flat_tiles,
            flat_std_threshold=flat_std_threshold,
            patch_postprocess=False,
            progress_cb=_progress,
        )
        out = blended

    if final_postprocess:
        out = postprocess_step4(out, sharpen=final_sharpen, color=final_color)

    out.save(path_out)
    logger.info("Saved: %s", path_out)
    return path_out
