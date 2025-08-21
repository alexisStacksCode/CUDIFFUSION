from typing import Any


VERSION: str = "1.0.0"
DEFAULT_SETTINGS: dict[str, Any] = {
    "image_model": {
        "use_vae_tiling": True,
        "scheduler": "default",
        "rng_type": "default",
    },
}
IMAGE_MODEL_DIR_PATH: str = "models/"
IMAGE_OUTPUT_DIR_PATH: str = "images/"
DATA_DIR_PATH: str = "data/"
SETTINGS_FILENAME: str = "settings.json"
WARNING_GENERIC: str = "An error occurred."
