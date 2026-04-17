"""
Pipeline: ControlNet (Canny) + Img2Img + merged Where's Waldo weights.

Hugging Face: SanDiegoDude/WheresWaldoStyle
  - Wheres_Waldo_Style_14000step_merged.ckpt (Dreambooth-merged SD1.5 with Waldo illustration style)
  - ControlNet: lllyasviel/sd-controlnet-canny (edge structure conditioning)

First run downloads ~4GB+ checkpoint from Hugging Face; needs disk space and network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

logger = logging.getLogger(__name__)

WALDO_REPO = "SanDiegoDude/WheresWaldoStyle"
WALDO_CKPT_FILENAME = "Wheres_Waldo_Style_14000step_merged.ckpt"
CONTROLNET_CANNY_ID = "lllyasviel/sd-controlnet-canny"

DEFAULT_PROMPT = (
    "(Wheres Waldo Style:1.0), illustration, busy detailed scene, "
    "(bright primary colors:1.1), pen and ink, children's book illustration"
)
DEFAULT_NEGATIVE = (
    "photograph, photo realistic, blurry, low quality, washed out, monochrome, "
    "text, watermark, ugly, deformed"
)


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


def resize_for_waldo(
    image: Image.Image,
    max_long_edge: int = 640,
) -> Image.Image:
    """Resize so long edge is capped (per model README); avoids huge resolutions that blur."""
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return image.convert("RGB")
    scale = max_long_edge / long_edge
    nw, nh = int(w * scale), int(h * scale)
    return image.convert("RGB").resize((nw, nh), Image.Resampling.LANCZOS)


def make_canny_control_image(
    rgb_image: Image.Image,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> Image.Image:
    """Match diffusers example: Canny edges as 3-channel PIL RGB."""
    np_image = np.array(rgb_image.convert("RGB"))
    edges = cv2.Canny(np_image, low_threshold, high_threshold)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    return Image.fromarray(edges)


@dataclass
class StylizeConfig:
    prompt: str = DEFAULT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE
    num_inference_steps: int = 24
    strength: float = 0.85
    guidance_scale: float = 3.0
    controlnet_conditioning_scale: float = 1.1
    max_long_edge: int = 896
    canny_low: int = 100
    canny_high: int = 200
    seed: Optional[int] = None


class WaldoStylizerPipelineA:
    """Lazy-loaded pipeline so importing the module does not allocate GPU memory."""

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

    def _load_pipeline(self):
        if self._pipe is not None:
            return self._pipe

        from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
        from diffusers import UniPCMultistepScheduler

        logger.info("Loading ControlNet Canny + Waldo checkpoint (may take a minute)...")
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_CANNY_ID,
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
            pipe.enable_vae_slicing()
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
        config: Optional[StylizeConfig] = None,
    ) -> Image.Image:
        cfg = config or StylizeConfig()
        init_image = resize_for_waldo(image, max_long_edge=cfg.max_long_edge)
        control_image = make_canny_control_image(
            init_image,
            low_threshold=cfg.canny_low,
            high_threshold=cfg.canny_high,
        )

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
        return result.images[0]


def stylize_image(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    config: Optional[StylizeConfig] = None,
    ckpt_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Load image from disk, stylize, save; returns the output path."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path_in = Path(input_path)
    path_out = Path(output_path)
    path_out.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(path_in).convert("RGB")
    stylizer = WaldoStylizerPipelineA(ckpt_path=ckpt_path)
    out = stylizer.stylize(image, config=config)
    out.save(path_out)
    logger.info("Saved: %s", path_out)
    return path_out
