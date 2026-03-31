import gradio as gr
from part1 import ask, indexed_companies, extract_sankey_structure
from charts import build_sankey

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
chat = gr.Chatbot(show_label=False)

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
# SANKEY GRAPH FUNCTION
def generate_sankey(company_name):
    if not company_name:
        return None

    data = extract_sankey_structure(company_name)

    if data is None:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_annotation(
            text=f"Could not extract financials for {company_name}.<br>Check the filing is indexed.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14)
        )
        return fig

    return build_sankey(company_name, data)

# -------------------------------------------------------------------
# UI LAYOUT
with gr.Blocks(
        title="AI in Finance",
        css=css) as demo:

    with gr.Tabs():
        # Tab 1: Chat
        with gr.Tab("Chat"):
            gr.ChatInterface(
                chatbot=chat,
                title="AI Financial Chatbot",
                fn=respond,
                examples=[
                    "e.g. What was Visa's total revenue?",
                    "e.g. What are DigitalOcean's main risk factors?",
                    "e.g. Compare Visa and DigitalOcean's net income.",
                    "e.g. What does visa say about competition in their 10-K?"
                ],
            )
        # Tab 2: Currently Indexed
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

        # Tab 3: Add a company
        with gr.Tab("Add Company"):
            gr.Markdown("### Add a new company to the index")
            gr.Markdown(
                "Find a company's CIK number at "
                "[SEC EDGAR](https://www.sec.gov/search-filings). "
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

        # Tab 4: Graphs
        with gr.Tab("Charts"):
            gr.Markdown("Revenue Flow Breakdown")
            company_selector = gr.Dropdown(
                choices=sorted(indexed_companies),
                label="Select Company",
                value=None,
            )
            sankey_plot = gr.Plot(
                show_label=False
            )
            generate_btn = gr.Button("Generate Sankey", variant="primary")
            generate_btn.click(
                fn=generate_sankey,
                inputs=company_selector,
                outputs=sankey_plot
            )

    gr.HTML('<div class="custom-footer">AI-Powered SEC Filing Analysis Engine.<br/>'
            'All information is publicly available on company website or sec.gov</div>')

demo.launch(share=True)
