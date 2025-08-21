import gradio as gr
import stable_diffusion_cpp  # type: ignore

from modules.core import constants
from modules.core import shared
from modules import im_backend
from modules.ui import setting_components


class Element:
    def __init__(self) -> None:
        with gr.Sidebar(position="right"):
            gr.HTML(f"""
            <h1 style="text-align: right;">
                CUDIFFUSION
            </h1>
            <h2 style="text-align: right;">
                Version {constants.VERSION}
            </h2>
            """)
            self.image_model_dropdown: gr.Dropdown = gr.Dropdown(
                choices=im_backend.get_image_models(),
                label="Image Model",
                interactive=True,
            )
            self.vae_tiling_checkbox: setting_components.Checkbox = setting_components.Checkbox(
                key="image_model/use_vae_tiling",
                default_value=constants.DEFAULT_SETTINGS["image_model"]["use_vae_tiling"],
                label="VAE Tiling",
                interactive=True,
            )
            self.scheduler_dropdown: setting_components.Dropdown = setting_components.Dropdown(
                key="image_model/scheduler",
                default_value=constants.DEFAULT_SETTINGS["image_model"]["scheduler"],
                choices=(
                    "default",
                    "discrete",
                    "karras",
                    "exponential",
                    "ays",
                    "gits",
                ),
                label="Scheduler",
                interactive=True,
            )
            self.rng_type_dropdown: setting_components.Dropdown = setting_components.Dropdown(
                key="image_model/rng_type",
                default_value=constants.DEFAULT_SETTINGS["image_model"]["rng_type"],
                choices=(
                    "default",
                    "cuda",
                ),
                label="Random Number Generator",
                interactive=True,
            )
            self.load_image_model_button: gr.Button = gr.Button(
                value="Load",
                variant="primary",
                interactive=True,
            )

    def on_load_image_button_click(self, image_model: str, use_vae_tiling: bool, scheduler: str, rng_type: str, is_diffuser_busy: bool):
        if is_diffuser_busy:
            yield (
                gr.update(),
                gr.update(),
                gr.update(),
            )
            return

        if shared.diffuser["id"] != image_model:
            yield (
                gr.update(value="Loading...", interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
            try:
                shared.diffuser["id"] = image_model
                shared.diffuser["ref"] = stable_diffusion_cpp.StableDiffusion(
                    model_path=f"{constants.IMAGE_MODEL_DIR_PATH}{image_model}",
                    vae_tiling=use_vae_tiling,
                    rng_type=rng_type,
                    schedule=scheduler,
                )
            except:
                gr.Warning(constants.WARNING_GENERIC)
        yield (
            gr.update(value="Load", interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )
