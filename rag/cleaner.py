"""Text cleaning helpers: mojibake fixes, whitespace normalisation."""
import html
import re

# The DJSCE site stores some UTF-8 punctuation double-encoded; fix the common ones.
MOJIBAKE = {
    "â&euro;“": "–",
    "â&euro;”": "–",
    "â&euro;˜": "'",
    "â&euro;™": "'",
    "â&euro;œ": '"',
    "â&euro;�": '"',
    "â&euro;¦": "...",
    "â&euro;™": "'",
}


def clean_text(text: str) -> str:
    if not text:
        return ""
    for src, dst in MOJIBAKE.items():
        text = text.replace(src, dst)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    # Drop control chars except newline/tab
    text = re.sub(r"[^\S\n]+", " ", text)          # collapse runs of spaces/tabs
    text = re.sub(r" ?\n ?", "\n", text)            # trim spaces around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)          # collapse blank lines
    return text.strip()


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", name).strip("_")
