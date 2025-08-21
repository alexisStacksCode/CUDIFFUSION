from typing import Any

import gradio as gr
import gradio.themes

from modules.core import shared
from modules import settings
from modules import im_backend
from modules.ui import sidebar
from modules.ui import tab_t2i
from modules.ui import tab_i2i


if __name__ == "__main__":
    def create_base_interface() -> tuple[gr.Textbox, gr.Textbox, gr.Number, gr.Slider, gr.Dropdown, gr.Slider, gr.Slider, gr.Slider, gr.Button]:
        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                positive_prompt_textbox: gr.Textbox = gr.Textbox(
                    label="Positive Prompt",
                    interactive=True,
                )
                negative_prompt_textbox: gr.Textbox = gr.Textbox(
                    label="Negative Prompt",
                    interactive=True,
                )
                seed_number: gr.Number = gr.Number(
                    value=-1.0,
                    label="Seed",
                    placeholder="-1",
                    info="`-1` = random",
                    scale=3,
                    interactive=True,
                    precision=0,
                )
            with gr.Column(scale=1):
                steps_slider: gr.Slider = gr.Slider(
                    minimum=1.0,
                    maximum=100.0,
                    value=20.0,
                    step=1.0,
                    precision=0,
                    label="Steps",
                    interactive=True,
                    show_reset_button=False,
                )
                sampler_dropdown: gr.Dropdown = gr.Dropdown(
                    choices=(
                        "euler_a",
                        "euler",
                        "heun",
                        "dpm2",
                        "dpmpp2s_a",
                        "dpmpp2m",
                        "dpmpp2mv2",
                        "ipndm",
                        "ipndm_v",
                        "lcm",
                        "ddim_trailing",
                        "tcd",
                    ),
                    value="euler",
                    label="Sampler",
                    interactive=True,
                )
                cfg_scale_slider: gr.Slider = gr.Slider(
                    minimum=0.0,
                    maximum=30.0,
                    value=7.0,
                    step=0.5,
                    label="Classifier-Free Guidance Scale",
                    interactive=True,
                    show_reset_button=False,
                )
        with gr.Row(equal_height=True):
            width_slider: gr.Slider = gr.Slider(
                minimum=64.0,
                maximum=2048,
                value=512.0,
                step=64.0,
                precision=0,
                label="Width",
                scale=2,
                interactive=True,
                show_reset_button=False,
            )
            height_slider: gr.Slider = gr.Slider(
                minimum=64.0,
                maximum=2048.0,
                value=512.0,
                step=64.0,
                precision=0,
                label="Height",
                scale=2,
                interactive=True,
                show_reset_button=False,
            )
            generate_button: gr.Button = gr.Button(
                value="Generate",
                variant="primary",
                interactive=False,
                elem_classes="generate-image-button",
                scale=1,
            )
        return positive_prompt_textbox, negative_prompt_textbox, seed_number, steps_slider, sampler_dropdown, cfg_scale_slider, width_slider, height_slider, generate_button

    def on_demo_load():
        outputs: list[Any] = []
        for setting_component_value in shared.setting_component_values.values():
            outputs.append(setting_component_value)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    settings.load()
    settings.save()

    with gr.Blocks(theme=gradio.themes.Origin(), analytics_enabled=False, title="CUDIFFUSION", css_paths="main.css") as demo:
        sidebar_r: sidebar.Element = sidebar.Element()

        is_diffuser_busy: gr.State = gr.State(False)

        with gr.Row():
            clip_skip_slider: gr.Slider = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.0,
                step=1.0,
                precision=0,
                label="CLIP Skip",
                interactive=True,
                show_reset_button=False,
            )
            gr.Column(scale=5)
        with gr.Tabs():
            with gr.Tab("🎨 Text-to-Image") as tab_1:
                t2i_positive_prompt_textbox, t2i_negative_prompt_textbox, t2i_seed_number, t2i_steps_slider, t2i_sampler_dropdown, t2i_cfg_scale_slider, t2i_width_slider, t2i_height_slider, t2i_generate_button = create_base_interface()
                t2i_output: gr.Image = gr.Image(
                    width=560,
                    height=320,
                    type="pil",
                    label="Generated Image",
                    interactive=False,
                    show_fullscreen_button=False,
                )
            with gr.Tab("♻️ Image-to-Image") as tab_2:
                i2i_positive_prompt_textbox, i2i_negative_prompt_textbox, i2i_seed_number, i2i_steps_slider, i2i_sampler_dropdown, i2i_cfg_scale_slider, i2i_width_slider, i2i_height_slider, i2i_generate_button = create_base_interface()
                with gr.Row():
                    with gr.Column():
                        i2i_reference_image: gr.Image = gr.Image(
                            width=320,
                            height=320,
                            sources="upload",
                            type="pil",
                            label="Reference Image",
                            interactive=True,
                            show_fullscreen_button=False,
                        )
                    with gr.Column():
                        i2i_output: gr.Image = gr.Image(
                            width=560,
                            height=320,
                            type="pil",
                            label="Generated Image",
                            interactive=False,
                            show_fullscreen_button=False,
                        )
        gr.HTML("""
        <p style="text-align: center;">
            <strong>WARNING:</strong> Please do not refresh or close the page at any time unless the app isn't busy.
        </p>
        """)

        sidebar_r.load_image_model_button.click(
            fn=sidebar_r.on_load_image_button_click,
            inputs=(  # type: ignore
                sidebar_r.image_model_dropdown,
                sidebar_r.vae_tiling_checkbox.instance,
                sidebar_r.scheduler_dropdown.instance,
                sidebar_r.rng_type_dropdown.instance,
                is_diffuser_busy,
            ),
            outputs=(
                sidebar_r.load_image_model_button,
                t2i_generate_button,
                i2i_generate_button,
            ),
            show_progress="hidden",
        )

        t2i_generate_button.click(
            fn=tab_t2i.on_generate_button_click,
            inputs=(
                t2i_positive_prompt_textbox,
                is_diffuser_busy,
            ),
        ).success(
            fn=im_backend.mark_diffuser_as_busy,
            outputs=(
                tab_1,
                tab_2,
                t2i_generate_button,
                i2i_generate_button,
                is_diffuser_busy,
            ),
            show_progress="hidden",
        ).then(
            fn=tab_t2i.text_to_image,
            inputs=(
                clip_skip_slider,
                t2i_positive_prompt_textbox,
                t2i_negative_prompt_textbox,
                t2i_seed_number,
                t2i_steps_slider,
                t2i_sampler_dropdown,
                t2i_cfg_scale_slider,
                t2i_width_slider,
                t2i_height_slider,
            ),
            outputs=t2i_output,
            show_progress="hidden",
        ).then(
            fn=im_backend.mark_diffuser_as_idle,
            outputs=(
                tab_1,
                tab_2,
                t2i_generate_button,
                i2i_generate_button,
                is_diffuser_busy,
            ),
            show_progress="hidden",
        )

        i2i_generate_button.click(
            fn=tab_i2i.on_generate_button_click,
            inputs=(
                i2i_reference_image,
                is_diffuser_busy,
            ),
        ).success(
            fn=im_backend.mark_diffuser_as_busy,
            outputs=(
                tab_1,
                tab_2,
                t2i_generate_button,
                i2i_generate_button,
                is_diffuser_busy,
            ),
            show_progress="hidden",
        ).then(
            fn=tab_i2i.image_to_image,
            inputs=(
                clip_skip_slider,
                i2i_positive_prompt_textbox,
                i2i_negative_prompt_textbox,
                i2i_seed_number,
                i2i_steps_slider,
                i2i_sampler_dropdown,
                i2i_cfg_scale_slider,
                i2i_width_slider,
                i2i_height_slider,
                i2i_reference_image,
            ),
            outputs=i2i_output,
            show_progress="hidden",
        ).then(
            fn=im_backend.mark_diffuser_as_idle,
            outputs=(
                tab_1,
                tab_2,
                t2i_generate_button,
                i2i_generate_button,
                is_diffuser_busy,
            ),
            show_progress="hidden",
        )

        demo.load(
            fn=on_demo_load,
            outputs=shared.setting_components,
            show_progress="hidden",
        ).then(
            fn=lambda: (
                gr.update(choices=im_backend.get_image_models()),
                gr.update(interactive=im_backend.is_diffuser_loaded()),
                gr.update(interactive=im_backend.is_diffuser_loaded()),
            ),
            outputs=(
                sidebar_r.image_model_dropdown,
                t2i_generate_button,
                i2i_generate_button,
            ),
            show_progress="hidden",
        )

    demo.launch(inbrowser=True, server_port=4200)
