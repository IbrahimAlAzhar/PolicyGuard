"""Extract text from downloaded review HTML/PDF files and identify candidate weakness spans.

This is a lightweight rule-based starter. Later you can replace it with your LLM/NLI classifier.

Example:
python -m collectors.parse_review_text \
  --input-dir data/reviews \
  --out data/processed/weakness_candidates.jsonl
"""
import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List

from bs4 import BeautifulSoup
from pypdf import PdfReader
from tqdm import tqdm

from utils.io import append_jsonl, load_config

DEFAULT_WEAKNESS_TERMS = [
    "limitation", "limitations", "concern", "weakness", "weaknesses", "unclear",
    "underpowered", "not representative", "generalizability", "generalisability",
    "external validity", "selection bias", "sampling bias", "confounding", "missing subgroup",
    "not stratified", "insufficient demographic", "lack of diversity", "race", "ethnicity",
    "gender", "sex", "age", "socioeconomic", "health equity", "disparity", "bias",
]

REVIEWER_PATTERNS = [
    r"Reviewer\s*#?\s*\d+",
    r"Referee\s*#?\s*\d+",
    r"Reviewer's report",
    r"Comments to the Author",
    r"Major comments",
    r"Minor comments",
    r"Weaknesses",
    r"Concerns",
]


def read_pdf_text(path: str) -> str:
    parts = []
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
    except Exception:
        return ""
    return "\n".join(parts)


def read_html_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text("\n")
    except Exception:
        return ""


def read_text(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".pdf"):
        return read_pdf_text(path)
    if lower.endswith((".html", ".htm")):
        return read_html_text(path)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def clean_text(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sentence_split(text: str) -> List[str]:
    # Simple splitter for review text; good enough for first filtering.
    text = re.sub(r"\s+", " ", text)
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sents if len(s.strip()) > 25]


def reviewer_section_hint(text: str, char_index: int) -> str:
    prefix = text[:char_index]
    hints = []
    for pat in REVIEWER_PATTERNS:
        for m in re.finditer(pat, prefix, flags=re.IGNORECASE):
            hints.append((m.start(), m.group(0)))
    if not hints:
        return "unknown"
    return sorted(hints, key=lambda x: x[0])[-1][1]


def keyword_hits(sentence: str, terms: List[str]) -> List[str]:
    s = sentence.lower()
    return [t for t in terms if t.lower() in s]


def classify_societal(sentence: str) -> str:
    s = sentence.lower()
    direct_terms = ["race", "racial", "ethnicity", "ethnic", "gender", "sex", "female", "male", "women", "men", "age", "socioeconomic", "income", "education", "rural", "urban", "minority", "underserved", "disparity", "inequity", "equity", "subgroup", "stratified"]
    indirect_terms = ["generalizability", "generalisability", "external validity", "representative", "population", "deployment", "implementation", "access", "clinical setting", "health system"]
    if any(t in s for t in direct_terms):
        return "direct_demographic_or_equity"
    if any(t in s for t in indirect_terms):
        return "indirect_societal_or_external_validity"
    return "scientific_or_methodological"


def extract_candidates(path: str, terms: List[str]) -> Iterable[Dict]:
    raw = read_text(path)
    text = clean_text(raw)
    if not text:
        return []
    candidates = []
    for sent in sentence_split(text):
        hits = keyword_hits(sent, terms)
        if not hits:
            continue
        idx = text.find(sent[:80])
        candidates.append({
            "source_file": path,
            "reviewer_section_hint": reviewer_section_hint(text, idx if idx >= 0 else 0),
            "candidate_span": sent,
            "keyword_hits": hits,
            "societal_impact_relevance_rule": classify_societal(sent),
        })
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    terms = cfg.get("weakness_terms") or DEFAULT_WEAKNESS_TERMS

    files = []
    for ext in ["*.pdf", "*.html", "*.htm", "*.txt"]:
        files.extend(Path(args.input_dir).rglob(ext))

    open(args.out, "w", encoding="utf-8").close()
    n = 0
    for path in tqdm(files, desc="Parse review files"):
        for rec in extract_candidates(str(path), terms):
            append_jsonl(args.out, rec)
            n += 1
    print(f"Saved {n} candidate weakness spans to {args.out}")


if __name__ == "__main__":
    main()
