"""Task 4 - Frontend: Streamlit chatbot for the DJSCE RAG system.

Run:  streamlit run app.py
"""
import os
import subprocess
import sys
import time

import streamlit as st

from pathlib import Path

from rag.pipeline import build_pipeline

st.set_page_config(page_title="DJSCE RAG Chatbot", page_icon="🎓", layout="centered")


def _ensure_vector_db():
    """On first run (e.g. Streamlit Cloud deploy) the chroma_db/ directory is
    gitignored, so the vector store won't exist. Build it automatically so the
    deploy is self-initializing instead of erroring out.

    This only runs when the DB is missing; subsequent runs reuse the built DB
    (within the lifecycle of the Streamlit container).
    """
    root = Path(__file__).resolve().parent
    if (root / "chroma_db").exists():
        return

    st.info("Vector DB not found — building it now from the bundled data "
            "(faculty + syllabus). This downloads a ~90MB embedding model once "
            "and may take a minute or two on the first run.")

    build_script = root / "scripts" / "build_vectorstore.py"
    if not build_script.exists():
        st.error(f"Build script not found at {build_script}. "
                 "Please run `python scripts/build_vectorstore.py` locally and deploy again.")
        st.stop()

    with st.spinner("Building vector DB (chunking + embedding)..."):
        # Streamlit Cloud mounts the repo at /mount/src/<repo>; run from the
        # repo root so relative paths in build_vectorstore.py resolve correctly.
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, str(build_script)],
                cwd=str(root),
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            st.error("Vector DB build timed out after 10 minutes. "
                     "Check the data files under `data/` and try again.")
            st.stop()

        if proc.returncode != 0:
            st.error("Vector DB build failed.\n\n"
                     f"```\n{proc.stdout}\n{proc.stderr}\n```")
            st.stop()

        if not (root / "chroma_db").exists():
            st.error("Vector DB build finished but `chroma_db/` was not created. "
                     "Check the build logs above.")
            st.stop()

        elapsed = time.time() - start
        st.success(f"Vector DB ready in {elapsed:.0f}s.")


@st.cache_resource(show_spinner="Loading RAG pipeline (embeddings + vector DB)...")
def load_pipeline():
    _ensure_vector_db()
    return build_pipeline()


try:
    pipe = load_pipeline()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Failed to load RAG pipeline: {e}")
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
