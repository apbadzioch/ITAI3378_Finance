import plotly.graph_objects as go

def build_sankey(company_name: str, data: dict) -> go.Figure:
    """
    Build a sankey chart from extracted 10-K financial data.
    """
    labels= [
        "Revenue",
        "Cost of Revenue",
        "Gross Profit",
        "R&D",
        "Sales & Marketing",
        "G&A",
        "Operating Income",
        "Net Income",
    ]

    # source -> target -> value
    source = [0, 0, 2, 2, 2, 2, 6, 6]
    target = [1, 2, 3, 4, 5, 6, 7, 8]
    value = [
        data["cost_of_revenue"],
        data["gross_profit"],
        data["rd"],
        data["sales_marketing"],
        data["ga"],
        data["operating_income"],
        data["taxes"],
        data["net_income"],
    ]

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=[
                "#2196F3",  # Revenue - blue
                "#F44336",  # Cost of Revenue - red
                "#4CAF50",  # Gross Profit - green
                "#FF9800",  # R&D - orange
                "#FF9800",  # Sales & Marketing - orange
                "#FF9800",  # G&A - orange
                "#4CAF50",  # Operating Income - green
                "#F44336",  # Taxes - red
                "#2196F3",  # Net Income - blue
            ]
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    ))
    fig.update_layout(
        title_text=f"{company_name} -- Revenue Breakdown",
        font_size=13,
        height=500
    )
    return fig
