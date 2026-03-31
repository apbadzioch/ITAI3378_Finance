import plotly.graph_objects as go

NODE_COLORS = {
    "income": "#4CAF50",
    "cost": "#F44336",
}
ROOT_COLOR = "#2196F3"

def build_sankey(company_name: str, data: dict) -> go.Figure:
    """
    Build a sankey chart from extracted 10-K financial data.
    """
    nodes = data["nodes"]
    links = data["links"]
    node_types = data.get("node_types", ["income"] * len(nodes))

    colors = [
        ROOT_COLOR if i == 0 else NODE_COLORS.get(t, "#9E9E9E")
        for i, t in enumerate(node_types)
    ]

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=colors,
        ),
        link=dict(
            source=[lnk["source"] for lnk in links],
            target=[lnk["target"] for lnk in links],
            value=[lnk["value"] for lnk in links],
        )
    ))

    fig.update_layout(
        title_text=f"{company_name} -- Revenue Flow (from 10-K)",
        font_size=13,
        height=500,
    )
    return fig
