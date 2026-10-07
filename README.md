# 🎓 DJSCE RAG Chatbot

A **local-first Retrieval-Augmented Generation (RAG)** chatbot for **Dwarkadas J. Sanghvi College of Engineering (DJSCE)**, focused on the **Computer Science and Engineering (Data Science) — CSE (DS)** department.

Ask things like:

- *Who is the HOD for the CSE DS department?* → **Dr. Kriti Srivastava**
- *What subjects are taught in the 5th semester?*
- *What are some electives a student can choose from in the 6th semester?*
- *Which faculty members joined DJSCE in 2007?*
- *Tell me about Dr. Kriti Srivastava's qualifications.*

Built with **LangChain + Chroma (vector DB) + sentence-transformers (local embeddings) + Streamlit**, and an LLM of your choice (local **LLaMA 3.1 via Ollama** recommended, or any OpenAI-compatible API).

---

## Architecture

```
djsce.ac.in (CSE-DS faculty page, syllabus PDFs)
        │  scripts/scrape_faculty.py / download_syllabus.py     ← Task 1: Data collection
        ▼
data/faculty/*.txt + CSV, data/faculty_bios/*.txt, data/syllabus_text/*.txt   (plain text)
        │  scripts/build_vectorstore.py                          ← Task 2: Preprocess & store
        ▼   chunk (LangChain RecursiveCharacterTextSplitter)
        ▼   embed (all-MiniLM-L6-v2, local)
        ▼   persist → chroma_db/ (Chroma vector DB, cosine)
        │
rag/pipeline.py                                              ← Task 3: RAG pipeline
   question ──► query router (semester / faculty intent → metadata filter)
             ──► Chroma similarity search (k=8)
             ──► hybrid rerank (dense score + lexical overlap)
             ──► prompt with context ──► LLM (Ollama / OpenAI-compatible)
             ──► answer + cited sources
        │
app.py  (Streamlit chat UI, with sources expander)           ← Task 4: Frontend
```

## Project structure

```
├── app.py                        # Streamlit chatbot UI (Task 4)
├── rag/
│   ├── config.py                 # paths, models, provider settings
│   ├── cleaner.py                # text cleaning (mojibake, whitespace)
│   ├── pdf_utils.py              # PDF → text (pypdf)
│   ├── llm_factory.py            # Ollama / OpenAI-compatible / fallback
│   └── pipeline.py               # retriever + query router + LLM chain
├── scripts/
│   ├── scrape_faculty.py         # Task 1: scrape faculty (+ bio/CV PDFs, bonus)
│   ├── download_syllabus.py      # Task 1: syllabus PDFs → text
│   ├── build_vectorstore.py      # Task 2: chunk → embed → Chroma
│   └── test_queries.py           # end-to-end sanity test
├── data/                         # generated: faculty, bios, syllabus text (+ PDFs)
├── chroma_db/                    # generated: vector store
├── requirements.txt
└── .env.example
```

## Setup

```bash
pip install -r requirements.txt
```

## Run (3 steps)

```bash
# 1) Collect data: faculty + syllabus (plain text files)
python scripts/scrape_faculty.py        # add --no-bios to skip CV PDFs
python scripts/download_syllabus.py --all   # Sem III–VIII + honors/minors (drop --all for core only)

# 2) Chunk → embed → vector DB (~1 min; downloads the 90MB MiniLM model once)
python scripts/build_vectorstore.py --reset

# 3) Chat!
streamlit run app.py
```

Sanity check without the UI:

```bash
python scripts/test_queries.py
```

## LLM configuration (Task 3)

**Currently configured: Google Gemini** (`gemini-3.5-flash-lite`) via your
AI Studio API key stored in `.env` (git-ignored). The model is set there
because the default `gemini-3.7-flash` free-tier quota fills up quickly;
`gemini-3.5-flash-lite` has its own quota headroom.

The pipeline also works with zero setup in **extractive mode** (returns the
most relevant excerpts with citations) whenever no LLM is reachable —
including transient API errors, which retry once and then fall back gracefully.

Other providers:

**Option A — fully local:** install [Ollama](https://ollama.com), then

```bash
ollama pull llama3.1
```

Remove/rename `.env` (or set `LLM_PROVIDER=ollama`) and the app auto-detects Ollama.

**Option B — any OpenAI-compatible API:** set in `.env`:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini            # or e.g. llama-3.1-8b-instant on Groq
OPENAI_BASE_URL=https://api.groq.com/openai/v1   # optional (Groq/OpenRouter/LM Studio)
```

Force a provider with `LLM_PROVIDER=gemini|ollama|openai|none`.

## How retrieval stays accurate

1. **Metadata filters** — the syllabus documents are stored with `semester`,
   `scheme` and `kind` (core/honors/minor) metadata. Questions mentioning a
   semester ("5th semester", "sem 5", "sixth semester") are routed to only that
   semester's chunks; people questions (HOD, who is X, joined…) are routed to
   faculty records/bios only.
2. **Hybrid reranking** — dense cosine similarity is combined with a lexical
   overlap score so acronyms like **HOD / CSE-DS** that embeddings miss still surface.
3. **Grounded prompting** — the system prompt forbids inventing facts and requires
   source citations; answers quote only retrieved context.

## Bonus features included

- ✅ **Richer faculty data** — every member's bio/CV PDF is downloaded and its text
  (qualifications, experience, publications, areas of interest) is chunked and indexed,
  plus derived *years at DJSCE* from the joining date.
- ✅ **Multiple schemes** — DJS23 (latest), DJS22 syllabi for Semesters III–VIII,
  plus Honors and Minor programs.
- 🔜 **Other departments** — the same pipeline works for any department: point
  `FACULTY_PAGE_URL` at another department's faculty page and add its syllabus
  URLs to `scripts/download_syllabus.py`, then re-run steps 1–2.
- 🔜 **Admissions / fees / achievements** — append more scraped text files to
  `data/` and re-run `build_vectorstore.py`; the router falls back to
  whole-corpus search for such queries.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| "Vector DB not found" | Run `python scripts/build_vectorstore.py` first |
| Extractive mode in sidebar | No LLM detected — install Ollama or set `OPENAI_API_KEY` |
| Slow first query | The embedding model downloads once (~90 MB) into the HF cache |
| Stale data | Re-run the two collection scripts, then `build_vectorstore.py --reset` |
