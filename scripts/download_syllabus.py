"""Task 1 - Data Collection (part 2): download CSE(DS) syllabus PDFs and
convert them to clean plain-text files.

Default downloads the main scheme/syllabus for Semesters III-VIII.
Use --all to also fetch Honors / Minor syllabi (bonus).

Usage:
    python scripts/download_syllabus.py          # main semester syllabi
    python scripts/download_syllabus.py --all    # + honors/minors
"""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.config import BASE_URL, SYLLABUS_PDF_DIR, SYLLABUS_TEXT_DIR
from rag.cleaner import clean_text
from rag.pdf_utils import pdf_to_text

HEADERS = {"User-Agent": "Mozilla/5.0 (DJSCE-RAG-student-project)"}

# name -> url (as listed on https://www.djsce.ac.in/ug-computer-science-engineering-data-science)
MAIN_PDFS = {
    "Sem3_DJS23":  f"{BASE_URL}/docs/courses/Data Science/Syllabus/Semester III.pdf",
    "Sem4_DJS23":  f"{BASE_URL}/docs/courses/Data Science/Syllabus/Sem_4_DJS23_Final_Consolidated_Syllabus.pdf",
    "Sem5_DJS23":  f"{BASE_URL}/docs/courses/Data Science/Syllabus/Semester V.pdf",
    "Sem6_DJS23":  f"{BASE_URL}/docs/courses/Data Science/Syllabus/Sem_6_DJS23_Final_Consolidated_Syllabus.pdf",
    "Sem7_DJS23":  f"{BASE_URL}/docs/courses/Data Science/Syllabus/Final Semester VII Scheme and Syllabus 28072026.pdf",
    "Sem8_DJS22":  f"{BASE_URL}/docs/courses/Data Science/Syllabus/Sem_8_DJS22_Final_Consolidated_Syllabus.pdf",
}

BONUS_PDFS = {
    "Sem3_Honors_DJS23":  f"{BASE_URL}/docs/courses/Data Science/Syllabus/Semester III Honors.pdf",
    "Sem4_Honors_DJS23":  f"{BASE_URL}/docs/courses/Data Science/Syllabus/HON_Sem_4_DJS23_Final_Consolidated_Syllabus.pdf",
    "Sem5_Honors_DJS23":  f"{BASE_URL}/docs/courses/Data Science/Syllabus/Semester V Honors.pdf",
    "Sem6_Honors_DJS23":  f"{BASE_URL}/docs/courses/Data Science/Syllabus/HON_Sem_6_DJS23_Final_Consolidated_Syllabus.pdf",
    "Sem7_Honors_DJS23":  f"{BASE_URL}/docs/courses/Data Science/Syllabus/Semester VII Honors.pdf",
    "Sem8_Honors_DJS22":  f"{BASE_URL}/docs/courses/Data Science/Syllabus/HON_Sem_8_DJS22_Final_Consolidated_Syllabus.pdf",
    "Sem8_Minor_DJS22":   f"{BASE_URL}/docs/courses/Data Science/Syllabus/MINOR_Sem_8_DJS22_Final_Consolidated_Syllabus.pdf",
    "Minor_ML_DJS22":     f"{BASE_URL}/docs/courses/Final Scheme and Syllabus for Minor (ML) (1).pdf",
    "Sem3_DJS22":         f"{BASE_URL}/docs/courses/Data Science/Syllabus/Sem 3 - DJS22.pdf",
    "Sem4_DJS22":         f"{BASE_URL}/docs/courses/Data Science/CSE_DS DJS22 sem IV Syllabus.pdf",
    "Sem5_DJS22":         f"{BASE_URL}/docs/courses/Data Science/Final Sem 5 DJS22 Scheme and Syllabus 10-07.pdf",
    "Sem6_DJS22":         f"{BASE_URL}/docs/courses/Data Science/Final Consolidated Sem VI Scheme and Syllabus 16-01-2025.pdf",
    "Sem7_DJS22":         f"{BASE_URL}/docs/courses/Data Science/Syllabus/Sem 7 DJS22 Scheme and Syllabus with ILE 1-7-25.pdf",
}


def download(name: str, url: str) -> Path | None:
    pdf_path = SYLLABUS_PDF_DIR / f"{name}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 10_000:
        print(f"  cached: {pdf_path.name}")
        return pdf_path
    try:
        r = requests.get(url, headers=HEADERS, timeout=120)
        r.raise_for_status()
        if not r.content[:5].startswith(b"%PDF"):
            print(f"  NOT A PDF ({r.status_code}, {len(r.content)}B): {url}")
            return None
        pdf_path.write_bytes(r.content)
        print(f"  downloaded: {pdf_path.name} ({len(r.content) // 1024} KB)")
        return pdf_path
    except Exception as e:
        print(f"  FAILED {name}: {type(e).__name__}: {str(e)[:120]}")
        return None


def convert(pdf_path: Path) -> None:
    text = clean_text(pdf_to_text(pdf_path))
    out = SYLLABUS_TEXT_DIR / (pdf_path.stem + ".txt")
    header = (f"DJSCE - CSE (Data Science) syllabus document: {pdf_path.stem}\n"
              f"(Scheme & Syllabus, Dwarkadas J. Sanghvi College of Engineering)\n\n")
    out.write_text(header + text, encoding="utf-8")
    print(f"    -> {out.name} ({len(text)} chars)")


def main() -> None:
    wanted = dict(MAIN_PDFS)
    if "--all" in sys.argv:
        wanted.update(BONUS_PDFS)
    print(f"Downloading {len(wanted)} syllabus PDFs ...\n")
    ok = 0
    for name, url in wanted.items():
        print(f"[{name}]")
        pdf = download(name, url)
        if pdf:
            convert(pdf)
            ok += 1
    print(f"\nDone: {ok}/{len(wanted)} PDFs converted to text in {SYLLABUS_TEXT_DIR}")


if __name__ == "__main__":
    main()
