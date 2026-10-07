"""End-to-end sanity test: run the exercise's sample queries through the
RAG pipeline and print answers + retrieved sources.

Usage: python scripts/test_queries.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.pipeline import build_pipeline

QUERIES = [
    "Who is the HOD for the CSE DS department?",
    "What subjects are taught in the 5th semester?",
    "What are some electives a student can choose from in the 6th semester?",
    "Which faculty members joined DJSCE in 2007?",
    "Tell me about Dr. Kriti Srivastava's qualifications.",
]


def main() -> None:
    pipe = build_pipeline()
    print(f"Vector DB chunks : {pipe.count()}")
    print(f"LLM              : {pipe.llm_label}\n")
    for q in QUERIES:
        print("=" * 78)
        print("Q:", q)
        res = pipe.answer(q)
        print("-" * 78)
        print(res["answer"][:1500])
        print("-" * 78)
        for s in res["sources"]:
            meta = ", ".join(str(v) for v in s["meta"].values() if v)
            print(f"   [{s['score']:.2f}] {s['source']} ({meta})")
        print()


if __name__ == "__main__":
    main()
