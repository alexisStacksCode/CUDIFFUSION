import os
import datetime

import gradio as gr
from PIL import Image

from modules.core import constants
from modules.core import shared


def mark_diffuser_as_idle():
    return (
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(value="Generate", interactive=True),
        gr.update(value="Generate", interactive=True),
        False,
    )


def mark_diffuser_as_busy():
    return (
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(value="Generating...", interactive=False),
        gr.update(value="Generating...", interactive=False),
        True,
    )


def save_image(image: Image.Image) -> None:
    if not is_diffuser_loaded():
        return

    diffuser_string: str = os.path.splitext(shared.diffuser["id"])[0]
    time_string: str = datetime.datetime.now().strftime("%Y-%m-%d-%f")

    os.makedirs(constants.IMAGE_OUTPUT_DIR_PATH, exist_ok=True)
    image.save(f"{constants.IMAGE_OUTPUT_DIR_PATH}{diffuser_string}_{time_string}.png", "png")


def get_image_models() -> list[str]:
    image_models: list[str] = []
    if os.path.exists(constants.IMAGE_MODEL_DIR_PATH):
        for filename in os.listdir(constants.IMAGE_MODEL_DIR_PATH):
            if os.path.splitext(filename)[1] in (".safetensors", ".gguf"):
                image_models.append(filename)
    return image_models


def is_diffuser_loaded() -> bool:
    return shared.diffuser["id"] != "" and shared.diffuser["ref"] is not None
