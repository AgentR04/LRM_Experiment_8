"""Task 4 - Frontend: Streamlit chatbot for the DJSCE RAG system.

Run:  streamlit run app.py
"""
import streamlit as st

from rag.pipeline import build_pipeline

st.set_page_config(page_title="DJSCE RAG Chatbot", page_icon="🎓", layout="centered")


@st.cache_resource(show_spinner="Loading RAG pipeline (embeddings + vector DB)...)")
def load_pipeline():
    return build_pipeline()


try:
    pipe = load_pipeline()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ System status")
    st.markdown(f"**LLM:** {pipe.llm_label}")
    st.markdown(f"**Embeddings:** `all-MiniLM-L6-v2` (local, CPU)")
    st.markdown(f"**Vector DB:** Chroma — **{pipe.count()}** chunks")
    st.divider()
    st.caption(
        "**Data sources**\n"
        "- CSE (DS) faculty page — djsce.ac.in\n"
        "- CSE (DS) syllabus PDFs (Sem III–VIII)\n"
        "- Faculty bio/CV PDFs (bonus)")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------- Chat UI ----------------
st.title("🎓 DJSCE RAG Chatbot")
st.caption("Ask anything about the CSE (Data Science) department — faculty & syllabus. "
           "Answers are generated **only** from data stored locally in the vector DB.")

SAMPLES = [
    "Who is the HOD of the CSE DS department?",
    "What subjects are taught in the 5th semester?",
    "What electives can a student choose from in the 6th semester?",
    "Which faculty joined DJSCE in 2007?",
]

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    with st.expander("📚 Retrieved sources"):
        for i, s in enumerate(sources, 1):
            meta = " · ".join(f"{v}" for v in s["meta"].values())
            st.markdown(f"**{i}. `{s['source']}`**" + (f" — {meta}" if meta else "") +
                        f" · relevance `{s['score']}`")
            st.markdown(f"> {s['snippet']}...")


def handle(question: str):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context & generating answer..."):
            result = pipe.answer(question, chat_history=st.session_state.messages)
        st.markdown(result["answer"])
        render_sources(result["sources"])
    st.session_state.messages.append({"role": "assistant",
                                      "content": result["answer"],
                                      "sources": result["sources"]})


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            render_sources(m["sources"])

cols = st.columns(len(SAMPLES))
for col, q in zip(cols, SAMPLES):
    if col.button(q, use_container_width=False):
        handle(q)

if query := st.chat_input("Ask a question about DJSCE CSE (DS)..."):
    handle(query)
