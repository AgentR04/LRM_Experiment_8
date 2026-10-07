"""Task 2 - Preprocessing & Storage: chunk -> embed -> vector database.

Reads the plain-text files produced by the collection scripts, splits them
into small readable sections (LangChain RecursiveCharacterTextSplitter +
custom faculty-record splitting), embeds each section locally with
sentence-transformers (all-MiniLM-L6-v2) and stores everything in a
persistent Chroma vector database at ./chroma_db.

Usage:
    python scripts/build_vectorstore.py
    python scripts/build_vectorstore.py --reset   # wipe DB first
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import (CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME,
                        EMBEDDING_MODEL, FACULTY_BIOS_DIR, FACULTY_DIR,
                        SYLLABUS_TEXT_DIR)
from rag.cleaner import clean_text


def load_documents() -> list[Document]:
    docs: list[Document] = []

    # ---- Faculty records: one chunk per member ----------------------------
    fac_file = FACULTY_DIR / "faculty_cse_ds.txt"
    if fac_file.exists():
        for block in fac_file.read_text(encoding="utf-8").split("\n\n"):
            block = block.strip()
            if not block:
                continue
            md = {"source": "faculty_cse_ds.txt", "type": "faculty"}
            m_name = re.search(r"Name:\s*(.+)", block)
            m_year = re.search(r"Joining Date:\s*\d{1,2}[.\-/]\d{1,2}[.\-/](\d{4})", block)
            if m_name:
                md["name"] = m_name.group(1).strip()
            if m_year:
                md["joining_year"] = m_year.group(1)  # enables "joined in 2007" filters
            docs.append(Document(page_content=block, metadata=md))
    print(f"Faculty records : {sum(1 for d in docs)}")

    # ---- Faculty bios (bonus): chunked CVs --------------------------------
    n_bio = 0
    for txt in sorted(FACULTY_BIOS_DIR.glob("*.txt")):
        text = clean_text(txt.read_text(encoding="utf-8"))
        if len(text) < 80:
            continue
        # Keep the person's identity in EVERY chunk (CV sections further down
        # never repeat the name, so name queries would otherwise miss them).
        first_line = text.splitlines()[0] if text.splitlines() else txt.stem
        m = re.search(r"Bio / CV of (.+?) ", first_line)
        person = m.group(1) if m else txt.stem.replace("_", " ")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=f"{person} - CV section: {chunk}",
                                 metadata={"source": txt.name, "type": "faculty_bio",
                                          "person": person}))
        n_bio += 1
    print(f"Faculty bios    : {n_bio} files")

    # ---- Syllabus documents ------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    n_syl = 0
    for txt in sorted(SYLLABUS_TEXT_DIR.glob("*.txt")):
        text = txt.read_text(encoding="utf-8")
        stem = txt.stem
        meta = {"source": txt.name, "type": "syllabus"}
        m = re.match(r"Sem(\d)_(Honors_|Minor_)?(DJS\d+)", stem)
        if m:
            meta["semester"] = f"Semester {m.group(1)}"
            meta["kind"] = (m.group(2) or "").strip("_").lower() or "core"
            meta["scheme"] = m.group(3)
        elif stem.startswith("Minor"):
            meta["semester"], meta["kind"], meta["scheme"] = "Elective", "minor", "DJS22"
        for i, chunk in enumerate(splitter.split_text(text)):
            docs.append(Document(page_content=chunk, metadata=dict(meta)))
        n_syl += 1
    print(f"Syllabus files  : {n_syl}")
    print(f"Total documents : {len(docs)}")
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="delete existing vector DB first")
    args = ap.parse_args()

    if args.reset and CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print(f"Removed old DB at {CHROMA_DIR}")

    docs = load_documents()
    if not docs:
        sys.exit("No documents found. Run scripts/scrape_faculty.py and "
                 "scripts/download_syllabus.py first.")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL} (first run downloads ~90MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Embedding + storing in Chroma -> {CHROMA_DIR}")
    db = Chroma.from_documents(
        docs,
        embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Vector DB ready: {db._collection.count()} chunks stored.")


if __name__ == "__main__":
    main()
