import gradio as gr
from PIL import Image

from modules.core import constants
from modules.core import shared
from modules import im_backend


def image_to_image(clip_skip: int, positive_prompt: str, negative_prompt: str, seed: int, steps: int, sampler: str, cfg_scale: float, width: int, height: int, reference_image: Image.Image):
    if not im_backend.is_diffuser_loaded():
        raise gr.Error(visible=False, print_exception=False)

    yield None
    try:
        images: list[Image.Image] = shared.diffuser["ref"].generate_image(  # type: ignore
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            clip_skip=clip_skip,
            cfg_scale=cfg_scale,
            min_cfg=0.0,
            init_image=reference_image,
            width=width,
            height=height,
            sample_method=sampler,
            sample_steps=steps,
            seed=seed,
        )
        image: Image.Image = images[0]
        im_backend.save_image(image)
        yield image
    except:
        gr.Warning(constants.WARNING_GENERIC)
        yield None


def on_generate_button_click(image: Image.Image | None, is_diffuser_busy: bool):
    if image is None:
        raise gr.Error("You must provide a reference image.", print_exception=False)

    if is_diffuser_busy:
        raise gr.Error(visible=False, print_exception=False)
