"""Task 1 - Data Collection: scrape CSE(DS) faculty from djsce.ac.in.

Collects Name, Designation, Joining date (+ bio PDF link) for every faculty
member on the CSE (Data Science) department page, then converts everything
into clean plain-text + CSV files (Task 1 - cleaning & conversion step).

Bonus: by default it also downloads each faculty member's "View Bio" CV PDF
and extracts its text (qualifications, experience, publications...).

Usage:
    python scripts/scrape_faculty.py            # scrape + bios
    python scripts/scrape_faculty.py --no-bios  # skip CV download/extraction
"""
import csv
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.config import (BASE_URL, DEPARTMENT_NAME, FACULTY_BIOS_DIR, FACULTY_DIR,
                        FACULTY_PAGE_URL)
from rag.cleaner import clean_text, sanitize_filename
from rag.pdf_utils import pdf_to_text

HEADERS = {"User-Agent": "Mozilla/5.0 (DJSCE-RAG-student-project)"}


def parse_faculty(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    members = []
    for card in soup.select("div.col-12.col-sm-12.col-md-3"):
        spans = [s.get_text(" ", strip=True) for s in card.find_all("span")]
        if not spans:
            continue
        name, designation, joined = "", "", ""
        for s in spans:
            low = s.lower()
            if not name and s:
                name = s
            elif low.startswith("joined on"):
                joined = s
            elif s and not designation:
                designation = s
        m = re.search(r"Joined on\s*([\d./\-]+)", joined)
        bio_link = card.find("a", href=re.compile("faculty-docs", re.I))
        if not name:
            continue
        members.append({
            "name": clean_text(name),
            "designation": clean_text(designation),
            "joining_date": m.group(1) if m else "Not listed",
            "bio_url": (bio_link["href"] if bio_link and bio_link.has_attr("href") else ""),
        })
    return members


def years_at_djsce(joining_date: str) -> str:
    """Approximate years of service at DJSCE, derived from the joining date."""
    m = re.match(r"(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})", joining_date.strip())
    if not m:
        return ""
    from datetime import date
    d, mo, y = map(int, m.groups())
    try:
        start = date(y, mo, d)
    except ValueError:
        return ""
    today = date.today()
    return f"{(today - start).days / 365.25:.0f}"


def record_text(m: dict) -> str:
    lines = [
        f"Faculty Member - CSE (DS), Department of Computer Science and Engineering (Data Science), Dwarkadas J. Sanghvi College of Engineering (DJSCE)",
        f"Name: {m['name']}",
        f"Designation: {m['designation']}",
        f"Joining Date: {m['joining_date']}",
    ]
    yrs = years_at_djsce(m["joining_date"])
    if yrs:
        lines.append(f"Years at DJSCE (approx.): {yrs}")
    if "head of the department" in m["designation"].lower():
        lines.append(f"Role: {m['name']} is the Head of the Department (HOD) of {DEPARTMENT_NAME}.")
    if m["bio_url"]:
        lines.append(f"Bio/CV: {m['bio_url']}")
    return "\n".join(lines)


def main(include_bios: bool = True) -> None:
    print(f"Fetching {FACULTY_PAGE_URL} ...")
    resp = requests.get(FACULTY_PAGE_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    members = parse_faculty(resp.text)
    print(f"Found {len(members)} faculty members.")

    FACULTY_DIR.mkdir(parents=True, exist_ok=True)

    # --- Plain text (one record per faculty member, blank-line separated) ---
    txt_path = FACULTY_DIR / "faculty_cse_ds.txt"
    blocks = []
    for i, m in enumerate(members, 1):
        blocks.append(f"[Faculty Record {i}]\n" + record_text(m))
    txt_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Wrote {txt_path}")

    # --- CSV ---------------------------------------------------------------
    csv_path = FACULTY_DIR / "faculty_cse_ds.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["name", "designation", "joining_date", "years_at_djsce", "bio_url"])
        w.writeheader()
        for m in members:
            w.writerow({**m, "years_at_djsce": years_at_djsce(m["joining_date"])})
    print(f"Wrote {csv_path}")

    # --- Bonus: bio/CV PDFs -> text ----------------------------------------
    if not include_bios:
        print("--no-bios: skipped bio/CV download.")
        print("Done.")
        return
    ok = 0
    for m in members:
        if not m["bio_url"]:
            continue
        url = m["bio_url"]
        if url.startswith("/"):
            url = BASE_URL + url
        fname = sanitize_filename(m["name"]) + ".pdf"
        pdf_path = FACULTY_BIOS_DIR / fname
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            pdf_path.write_bytes(r.content)
            text = pdf_to_text(pdf_path)
            out = FACULTY_BIOS_DIR / (fname[:-4] + ".txt")
            if text.strip():
                header = (f"Bio / CV of {m['name']} ({m['designation']}), "
                          f"{DEPARTMENT_NAME}, DJSCE.\n")
                out.write_text(header + text, encoding="utf-8")
                ok += 1
                print(f"  bio ok: {m['name']} ({len(text)} chars)")
            else:
                out.write_text(header + "[CV PDF appears to be a scanned image; text not extractable.]\n",
                               encoding="utf-8")
                print(f"  bio empty (scanned?): {m['name']}")
            time.sleep(0.3)
        except Exception as e:
            print(f"  bio FAILED: {m['name']}: {type(e).__name__}: {e}")
    print(f"Bio PDFs extracted: {ok}")
    print("Done.")


if __name__ == "__main__":
    main(include_bios="--no-bios" not in sys.argv)
