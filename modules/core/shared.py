from typing import Any

import gradio as gr


setting_components: list[gr.Component] = []
setting_component_values: dict[int, Any] = {}
diffuser: dict[str, Any] = {
    "id": "",
    "ref": None,
}
