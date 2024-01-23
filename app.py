import torch

import numpy as np
import gradio as gr
import soundfile as sf
import tempfile

from transformers import pipeline
from huggingface_hub import InferenceClient

def _grab_best_device(use_gpu=True):
    if torch.cuda.device_count() > 0 and use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    return device

device = _grab_best_device()

title = """# MusicGen Prompt Upsampling 🎶
            MusicGen, a simple and controllable model for music generation.  
            **Model**: https://huggingface.co/facebook/musicgen-stereo-medium
            """

vibes = pipeline("text-to-audio",
                 "facebook/musicgen-stereo-medium",
                 torch_dtype=torch.float16,
                 device="cuda")

client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta",)


# Inference
def generate_audio(text,):
    prompt = f"Take the next sentence and enrich it with details. Keep it compact. {text}"
    output = client.text_generation(prompt, max_new_tokens=100)
    out = vibes(output)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, out["audio"][0].T, out["sampling_rate"])
    
    return f.name, output

css = """
#container{
    margin: 0 auto;
    max-width: 80rem;
}
#intro{
    max-width: 100%;
    text-align: center;
    margin: 0 auto;
}
"""

# Gradio blocks demo    
with gr.Blocks(css=css) as demo_blocks:
    gr.Markdown(title, elem_id="intro")

    with gr.Row(elem_id="container"):
        with gr.Column():
            inp_text = gr.Textbox(label="Input Prompt", info="What would you like MusicGen to synthesise?")
            btn = gr.Button("Generate Music! 🎶")
            
        with gr.Column():
            out = gr.Audio(autoplay=False, label=f"Generated Music", show_label=True,)
            prompt_text = gr.Textbox(label="Upsampled Prompt")

    with gr.Accordion("Use MusicGen with Transformers 🤗", open=False):
        gr.Markdown(
            """
            ```python
            import torch
            import soundfile as sf
            
            from transformers import pipeline

            synthesiser = pipeline("text-to-audio", 
                                    "facebook/musicgen-stereo-medium", 
                                    device="cuda:0", 
                                    torch_dtype=torch.float16)

            music = synthesiser("lo-fi music with a soothing melody", 
                                forward_params={"max_new_tokens": 256})

            sf.write("musicgen_out.wav", music["audio"][0].T, music["sampling_rate"])
            ```

        """
        )

    btn.click(generate_audio, inp_text, [out, prompt_text])
    

demo_blocks.queue().launch()