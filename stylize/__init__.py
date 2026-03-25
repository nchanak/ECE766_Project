"""Where's Waldo stylization — Pipeline: ControlNet Canny + Img2Img + Waldo checkpoint."""

from .pipeline_a import WaldoStylizerPipelineA, stylize_image

__all__ = ["WaldoStylizerPipelineA", "stylize_image"]
