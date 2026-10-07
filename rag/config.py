"""Central configuration for the DJSCE RAG chatbot."""
import os
from pathlib import Path

try:  # optional .env support
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]

# ---- Data locations -------------------------------------------------------
DATA_DIR = ROOT / "data"
FACULTY_DIR = DATA_DIR / "faculty"
FACULTY_BIOS_DIR = DATA_DIR / "faculty_bios"
SYLLABUS_PDF_DIR = DATA_DIR / "syllabus_pdfs"
SYLLABUS_TEXT_DIR = DATA_DIR / "syllabus_text"
CHROMA_DIR = ROOT / "chroma_db"

# ---- Embeddings / vector store -------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = "djsce_kb"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
RETRIEVER_K = 12  # wide enough to include full scheme/course tables

# ---- LLM ------------------------------------------------------------------
# LLM_PROVIDER: auto (default) | gemini | ollama | openai | none
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # any OpenAI-compatible API (Groq, OpenRouter, ...)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---- Source URLs ----------------------------------------------------------
BASE_URL = "https://www.djsce.ac.in"
FACULTY_PAGE_URL = f"{BASE_URL}/courses.php?course_id=5&sr_no=91"  # CSE(DS) faculty members
DEPARTMENT_NAME = "Computer Science and Engineering (Data Science)"

for _d in (FACULTY_DIR, FACULTY_BIOS_DIR, SYLLABUS_PDF_DIR, SYLLABUS_TEXT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
