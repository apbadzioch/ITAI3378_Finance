import gradio as gr
from part1 import ask


# remove the gradio default footer
css = """
footer {display: none !important}
.custom-footer {
    text-align: center;
    padding: 10px;
    color: yellow;
    font-size: 0.85em;;
}
"""


def respond(message, history):
    return ask(message)

with gr.Blocks(css=css) as demo:
    gr.ChatInterface(fn=respond, title="10-K Chatbot")
    gr.HTML('<div class="custom-footer">RAG APP PROJECT</div>')

demo.launch(share=True)
