import tempfile
import streamlit as st

from agents.supervisor import SupervisorAgent
from graph.workflow import graph

st.set_page_config(
    page_title="Multi-Agent Academic Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-Agent Academic Research Assistant")
st.markdown("Powered by **LangGraph + Groq + Tavily + arXiv**")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Project Settings")

    citation_style = st.selectbox(
        "Citation Style",
        ["IEEE", "APA"]
    )

    uploaded_files = st.file_uploader(
        "Upload Research Papers",
        type=["pdf"],
        accept_multiple_files=True
    )

# -----------------------------
# User Input
# -----------------------------
query = st.text_input(
    "Research Topic",
    placeholder="Artificial Intelligence in Cybersecurity"
)

generate = st.button(
    "🚀 Generate Research"
)

# -----------------------------
# Workflow
# -----------------------------
if generate:

    if not query:
        st.warning("Please enter a research topic.")
        st.stop()

    saved_files = []

    if uploaded_files:

        for uploaded in uploaded_files:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp:

                tmp.write(uploaded.read())

                saved_files.append(tmp.name)

    state = SupervisorAgent.initialize_state(
        query=query,
        uploaded_files=saved_files
        citation_style=citation_style
    )

    with st.spinner("Running Multi-Agent Workflow..."):

        final_state = graph.invoke(state)

    st.success("Research Generated Successfully!")

    tabs = st.tabs([
        "📖 Literature Review",
        "💡 Research Plan",
        "📝 Research Paper",
        "📚 Citations"
    ])

    with tabs[0]:
        st.markdown(final_state["literature_review"])

    with tabs[1]:
        st.markdown(final_state["research_plan"])

    with tabs[2]:
        st.markdown(final_state["research_paper"])

    with tabs[3]:

        for citation in final_state["citations"]:
            st.code(citation)
