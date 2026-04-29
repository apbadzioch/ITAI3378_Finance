import os, json
import streamlit as st
import plotly.graph_objects as go
from part2 import agent_app, bootstrap
from langchain_core.messages import HumanMessage, AIMessage
from charts import build_sankey

# --- Page Config ---
st.set_page_config(page_title="Financial 10-K Analyst", layout="wide")
st.title("📊 Financial 10-K Analyst AI")
st.markdown("Analysis of SEC filings using agentic RAG systems.")


# --- Initialization ---
# We use @st.cache_resource so bootstrap only runs ONCE when the app starts
@st.cache_resource
def init_system():
    bootstrap()


init_system()

# Initialize Chat History in Streamlit Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI Layout ---
# Display existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], go.Figure):
            st.plotly_chart(msg["content"], use_container_width=True)
        else:
            st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Ask about a company's finances..."):
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Prepare messages for LangGraph
    # Convert Streamlit history to LangChain format
    langchain_msgs = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            langchain_msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            # Only add text to history, not Figure objects
            if not isinstance(m["content"], go.Figure):
                langchain_msgs.append(AIMessage(content=m["content"]))

    # 3. Invoke the Agent
    with st.chat_message("assistant"):
        with st.spinner("Analyzing financial data..."):
            try:
                # Invoke the graph
                result = agent_app.invoke({"messages": langchain_msgs})
                final_msg = result["messages"][-1].content

                # 2. THE PAYLOAD INTERCEPTOR (Replaces your old chart_fig logic)
                if isinstance(final_msg, str) and "DATA_PAYLOAD:" in final_msg:
                    raw_json = final_msg.split("DATA_PAYLOAD:")[1].strip()
                    payload = json.loads(raw_json)
                    fig = None

                    if payload["type"] == "sankey":
                        fig = build_sankey(payload["company"], payload["data"])

                    elif payload["type"] == "stock_chart":
                        fig = go.Figure(data=[go.Candlestick(
                            x=payload["dates"],
                            open=payload["open"],
                            high=payload["high"],
                            low=payload["low"],
                            close=payload["close"],
                            name=payload["ticker"]
                        )])
                        fig.update_layout(title=f"{payload['company']} Price History",
                                          xaxis_rangeslider_visible=False,
                                          template="plotly_dark")

                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.session_state.messages.append({"role": "assistant", "content": fig})

                # 3. REPORT GENERATION
                elif isinstance(final_msg, str) and final_msg.startswith("REPORT_PATH:"):
                    report_path = final_msg.replace("REPORT_PATH:", "").strip()
                    report_name = os.path.basename(report_path)
                    with open(report_path, "rb") as f:
                        st.success(f"Report ready: **{report_name}**")
                        st.download_button("Download Report", f, file_name=report_name)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Report generated: `{report_name}`"
                        })

                # 4. STANDARD TEXT RESPONSE
                else:
                    if "DATA_PAYLOAD:" not in final_msg:
                        st.markdown(final_msg)
                        st.session_state.messages.append({"role": "assistant", "content": final_msg})



            except Exception as e:
                error_text = f"An error occurred: {str(e)}"
                st.error(error_text)




