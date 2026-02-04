import torch
import gradio as gr

# Use a pipeline as a high-level helper
from transformers import pipeline

model_path=("../Models/models--sshleifer--distilbart-cnn-12-6/snapshots/"
                "a4f8f3ea906ed274767e9906dbaede7531d660ff")

text_summary = pipeline("summarization", model=model_path,
               dtype = torch.bfloat16)


def summary(input_text):
    output = text_summary(input_text)
    return output[0]["summary_text"]

custom_css = """
/* Page background */
.gradio-container {
    background-color: #E0E7FF;
}

/* Center content & limit width */
.gradio-container .wrap {
    max-width: 700px;
    margin: 0 auto;
}

/* This is the GREY culprit — restyle it */
.gradio-container .panel,
.gradio-container .panel-body,
.gradio-container .gr-box {
    background: white !important;
    border-radius: 18px !important;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.08);
    border: none !important;
}

/* Space things nicely */
.gradio-container .gr-box {
    padding: 24px !important;
}

/* Make button look intentional */
button.primary {
    background-color: #6366F1 !important;
}

"""

with gr.Blocks() as demo:
    gr.Markdown(
        "# 📝 Text Summarizer\nPaste text below to get a concise summary."
    )

    input_box = gr.Textbox(label="Input Text", lines=6)
    output_box = gr.Textbox(label="Summary", lines=6, interactive=False)
    btn = gr.Button("Summarize")

    btn.click(summary, input_box, output_box)

demo.launch()
