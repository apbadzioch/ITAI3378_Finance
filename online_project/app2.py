import streamlit as st
import plotly.graph_objects as go
from part2 import agent_app, bootstrap
from langchain_core.messages import HumanMessage, AIMessage

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
if prompt := st.chat_input("Ask about a company's 10-K..."):
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

                # The result is a list of messages.
                # We want the LAST one, but we also need to check if
                # any tool calls produced a Figure.
                final_msg = result["messages"][-1]

                # Check if the final output is a Plotly Figure (from the build_chart tool)
                if isinstance(final_msg.content, go.Figure):
                    st.plotly_chart(final_msg.content, use_container_width=True)
                    st.session_state.messages.append({"role": "assistant", "content": final_msg.content})
                else:
                    # It's a text response
                    st.markdown(final_msg.content)
                    st.session_state.messages.append({"role": "assistant", "content": final_msg.content})

            except Exception as e:
                error_text = f"An error occurred: {str(e)}"
                st.error(error_text)

