"""Task 3 - RAG Pipeline: retrieve relevant local context for a question and
generate an answer with an LLM (or extractive fallback when no LLM exists).
"""
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from rag.config import (CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL,
                        RETRIEVER_K)
from rag.llm_factory import get_llm

SYSTEM_PROMPT = """You are "DJSCE Helper", the official assistant for Dwarkadas J. Sanghvi College of Engineering (DJSCE), Mumbai.
You answer questions about the Computer Science and Engineering (Data Science) [CSE (DS)] department using ONLY the context provided below, which comes from the department's faculty records and official syllabus PDFs.

Rules:
- Answer clearly and concisely. Use bullet points for lists of subjects, electives or faculty.
- Every factual claim must come from the context. Cite the source file(s) in parentheses, e.g. (Source: Sem5_DJS23.txt, faculty_cse_ds.txt).
- If several semesters/schemes exist in the context, mention which one you are quoting (e.g. "Semester V, DJS23 scheme").
- If the context partially answers the question, give EVERY matching item you can find in it (e.g. list every course row, every matching faculty record) rather than a sample.
- Scheme tables may mark electives with symbols like @ or #, or sections such as "DEPARTMENT ELECTIVES" / "Programme Elective". When asked for subjects/courses in a semester, enumerate ALL course rows from the scheme table (core + electives + labs); when asked for electives, enumerate every row identified as a department, programme or open elective.
- Only if the context is genuinely unrelated to the question, reply: "I couldn't find that in the DJSCE data I have." and suggest what the user could check on the college website instead. NEVER invent names, dates or subjects.
"""

# ---- Lightweight query router ---------------------------------------------
# Improves retrieval by filtering the vector DB on detected intent:
#   * "5th semester" / "sem 5" / "fifth semester" -> only that semester's syllabus
#   * people questions (HOD, who is X, joined, ...) -> only faculty records/bios
_SEM_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4,
              "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8}
_FACULTY_RE = re.compile(
    r"\b(hod|head of (the )?dept(artment)?|who is|who are|faculty|professors?|"
    r"prof\.|dr\.|joined|joining|designation|teaching staff|teachers?|staff)\b", re.I)


def detect_semester(question: str) -> str | None:
    ql = question.lower()
    if "sem" not in ql:
        return None
    m = re.search(r"\bsem(?:ester)?\s*[:\- ]?\s*([1-8])\b", ql)
    if m:
        return f"Semester {m.group(1)}"
    m = re.search(r"\b([1-8])(?:st|nd|rd|th)\b", ql)
    if m:
        return f"Semester {m.group(1)}"
    for w, n in _SEM_WORDS.items():
        if re.search(rf"\b{w}\b", ql):
            return f"Semester {n}"
    return None


def build_filters(question: str) -> list[dict | None]:
    """Ordered candidate filters for the question (first non-empty wins).
    Specific intents come first, falling back to broader scopes."""
    sem = detect_semester(question)
    if sem:
        return [{"$and": [{"semester": {"$eq": sem}}, {"type": {"$eq": "syllabus"}}]},
                {"type": {"$eq": "syllabus"}}, None]
    if _FACULTY_RE.search(question):
        year = re.search(r"\b((?:19|20)\d{2})\b", question)
        if year:
            return [{"$and": [{"type": {"$eq": "faculty"}},
                              {"joining_year": {"$eq": year.group(1)}}]},
                    {"type": {"$in": ["faculty", "faculty_bio"]}}, None]
        return [{"type": {"$in": ["faculty", "faculty_bio"]}}, None]
    return [None]


def _expand_query(question: str) -> str:
    """Query expansion: for subject/elective questions, append the wording
    used inside the scheme tables so those chunks embed close to the query."""
    ql = question.lower()
    sem = detect_semester(question)
    if not sem or not re.search(
            r"\b(subjects?|courses?|electives?|taught|classes|curriculum|programme)\b", ql):
        return question
    n = int(re.search(r"\d+", sem).group(0))
    roman = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V",
             6: "VI", 7: "VII", 8: "VIII"}[n]
    extra = [f"Scheme of Semester {roman} course list course code scheme table"]
    if "elective" in ql:
        extra.append("department elective open elective programme elective courses offered")
    return question + " " + " ".join(extra)


_STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "of", "for", "in", "on",
              "to", "and", "or", "what", "which", "who", "whom", "whose", "how",
              "why", "when", "where", "do", "does", "did", "can", "tell", "me",
              "give", "list", "some", "at", "from", "it", "its", "be", "by"}


def _lexical_overlap(question: str, text: str) -> float:
    """Fraction of content-words from the question that appear in the text
    (naive singular matching: "electives" also matches "elective")."""
    q_terms = [w for w in re.findall(r"[a-z0-9]+", question.lower())
               if w not in _STOPWORDS and len(w) > 1]
    if not q_terms:
        return 0.0
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    hits = sum(1 for w in q_terms
               if w in tokens or (w.endswith("s") and w[:-1] in tokens))
    return hits / len(q_terms)


CONTEXT_TMPL = """Context from the DJSCE knowledge base:
{context}

Chat history (may be empty):
{chat_history}

User question: {question}"""


def _format_docs(docs: list[tuple[Document, float]]) -> str:
    blocks = []
    for i, (doc, score) in enumerate(docs, 1):
        meta = doc.metadata
        src = meta.get("source", "?")
        extra = ", ".join(str(v) for k, v in meta.items()
                          if k not in ("source",) and v and k != "type")
        head = f"[{i}] {src}" + (f" ({extra})" if extra else "")
        blocks.append(f"{head}\n{doc.page_content}")
    return "\n\n".join(blocks)


class DJSCEPipeline:
    def __init__(self, k: int = RETRIEVER_K):
        self.k = k
        self.embeddings_name = EMBEDDING_MODEL
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"Vector DB not found at {CHROMA_DIR}. "
                "Run scripts/build_vectorstore.py first.")
        self.db = Chroma(collection_name=COLLECTION_NAME,
                         persist_directory=str(CHROMA_DIR),
                         embedding_function=self._embeddings())
        self.llm, self.llm_label = get_llm()
        if self.llm is not None:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", CONTEXT_TMPL),
            ])
            self.chain = self.prompt | self.llm | StrOutputParser()

    @staticmethod
    def _embeddings():
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
                                     model_kwargs={"device": "cpu"},
                                     encode_kwargs={"normalize_embeddings": True})

    def count(self) -> int:
        return self.db._collection.count()

    def retrieve(self, question: str, k: int | None = None):
        # Fetch a wide candidate pool, then rerank down to k: the reranker
        # (dense + lexical) needs room to surface exact dates/names/courses.
        pool = max((k or self.k) * 3, 24)
        search_query = _expand_query(question)
        pairs: list = []
        for where in build_filters(question):
            pairs = self.db.similarity_search_with_score(search_query, k=pool, filter=where)
            if pairs:
                break
        # Chroma cosine *distance* -> similarity score = 1 - distance
        scored = [(doc, 1.0 - dist) for doc, dist in pairs]
        # Hybrid rerank: dense similarity + lexical overlap (helps acronyms
        # like HOD / CSE-DS that embeddings miss).
        reranked = [(doc, 0.75 * sim + 0.25 * _lexical_overlap(question, doc.page_content))
                    for doc, sim in scored]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[: k or self.k]

    def answer(self, question: str, chat_history: list | None = None) -> dict:
        docs = self.retrieve(question)
        history = []
        for m in (chat_history or [])[-6:]:
            if m["role"] == "user":
                history.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant" and not m["content"].startswith("\ud83d\udcda"):
                history.append(AIMessage(content=m["content"][:500]))

        if self.llm is not None:
            answer = None
            for attempt in (0, 1):  # one retry: Gemini/edge APIs 503 under load
                try:
                    answer = self.chain.invoke({
                        "context": _format_docs(docs),
                        "chat_history": history,
                        "question": question,
                    })
                    break
                except Exception as e:
                    err = f"{type(e).__name__}: {str(e)[:200]}"
                    if attempt == 0:
                        import time
                        time.sleep(3)
            if answer is None:
                # Invalid key / network down -> degrade instead of crashing
                self._llm_error = err
                answer = self._extractive(docs) + (
                    f"\n\n*(LLM call failed - `{err}` - showing raw "
                    "retrieved excerpts instead. Check the API key in `.env`.)*")
        else:
            answer = self._extractive(docs)

        sources = []
        for doc, score in docs[:4]:
            meta = doc.metadata
            snippet = " ".join(doc.page_content.split())[:180]
            sources.append({"source": meta.get("source", "?"),
                            "meta": {k: v for k, v in meta.items() if k != "source"},
                            "score": round(score, 3),
                            "snippet": snippet})
        return {"answer": answer.strip(), "sources": sources}

    @staticmethod
    def _extractive(docs) -> str:
        lines = ["*(Retrieval-only mode: no LLM configured - showing the most relevant "
                 "excerpts from the DJSCE knowledge base.)*", ""]
        for i, (doc, score) in enumerate(docs[:4], 1):
            meta = doc.metadata
            extra = ", ".join(str(v) for k, v in meta.items()
                              if k not in ("source", "type") and v)
            lines.append(f"**{i}. {meta.get('source', '?')}" + (f" ({extra})" if extra else "") + f"** - relevance {score:.2f}")
            lines.append(f"> {doc.page_content.strip()[:600]}")
            lines.append("")
        return "\n".join(lines)


def build_pipeline() -> DJSCEPipeline:
    return DJSCEPipeline()
