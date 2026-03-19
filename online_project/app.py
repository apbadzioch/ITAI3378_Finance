import gradio as gr
from part1 import ask, indexed_companies # add_company,

# -------------------------------------------------------------
# STYLING
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

# -------------------------------------------------------------------
# CHAT FUNCTION
def respond(message, history):
    return ask(message)

# -------------------------------------------------------------------
# ADD COMPANY FUNCTION
def handle_add_company(company_name, cik):
    """
    Called when user submits a new company to index.
    Validates inputs and calls add_company() from file.
    """
    if not company_name.strip() or not cik.strip():
        return "Please enter a company name or CIK number."
    result = add_company(company_name.strip(), cik.strip())
    updated_list = "Indexed companies: " + ", ".join(sorted(indexed_companies))
    return f"{result}\n{updated_list}"

# -------------------------------------------------------------------
# UI LAYOUT
with gr.Blocks(
        title="AI in Finance",
        css=css) as demo:

    with gr.Tabs():
        # Tab 1: Chat
        with gr.Tab("Chat"):
            gr.ChatInterface(
                fn=respond,
                examples=[
                    "What was Visa's total revenue?",
                    "What are DigitalOcean's main risk factors?",
                    "Compare Visa and DigitalOcean's net income.",
                    "What does visa say about competition in their 10-K?"
                ],
            )
        # Tab 2: Add a company
        with gr.Tab("Add Company"):
            gr.Markdown("### Add a new company to the index")
            gr.Markdown(
                "Find a company's CIK number at "
                "[SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar). "
                "Once added, you can ask questions about it in the Chat tab."
            )

            with gr.Row():
                company_input = gr.Textbox(
                    label="Company Name",
                    placeholder="e.g. Apple"
                )
                cik_input = gr.Textbox(
                    label="CIK Number",
                    placeholder="e.g. 0000320193"
                )
            add_btn = gr.Button("Add Company", variant="primary")
            add_output = gr.Textbox(label="Status", interactive=False)
            add_btn.click(
                fn=handle_add_company,
                inputs=[company_input, cik_input],
                outputs=add_output
            )

        # Tab 3: Currently Indexed
        with gr.Tab("Indexed Companies"):
            gr.Markdown("### Companies currently in the index")
            indexed_display = gr.Textbox(
                value="\n".join(sorted(indexed_companies)) or "No companies indexed yet.",
                label="Indexed Companies",
                interactive=False,
                lines=10
            )
            refresh_btn = gr.Button("Refresh List")
            refresh_btn.click(
                fn=lambda: "\n".join(sorted(indexed_companies)) or "No companies indexed yet.",
                outputs=indexed_display
            )

    gr.HTML('<div class="custom-footer">RAG APP PROJECT</div>')

demo.launch(share=True)
