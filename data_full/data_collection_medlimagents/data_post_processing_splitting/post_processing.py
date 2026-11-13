"""
process_pubmed.py

Filter PubMed PMC .csv.gz files for medical AI / digital health / public health
papers with usable limitations sections. Writes filtered rows to one CSV in
crash-safe batches of 10.

Filters:
  HARD:
    - limitations_section not NaN/empty
    - MIN_LIMITATIONS_WORDS <= word_count(limitations_section) <= MAX_LIMITATIONS_WORDS
    - relevance_score >= MIN_RELEVANCE_SCORE
    - PMID not seen before (dedup across files)
  RELEVANCE SCORING (lenient — papers add up across signals):
    +3  journal keyword match (e.g. 'lancet digital', 'jama network open')
    +1..3  per tech keyword hit in title/abstract/keywords (capped at 3)
    +1..3  per health/society keyword hit  (capped at 3)
    +2  bonus if BOTH tech and health hits found (intersection signal)

Output:
    data/pubmed_post_processed/filtered_pubmed.csv

Tune the CONFIG block to widen / narrow the funnel.
"""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG — tune these without reading the rest of the file
# ---------------------------------------------------------------------------
PUBMED_DIR = Path("/lstr/sahara/datalab-ml/ibrahim/limagents_update/med_limagnets/data/pubmed")
OUT_DIR = Path("/lstr/sahara/datalab-ml/ibrahim/limagents_update/med_limagnets/data/pubmed_post_processed")
OUTPUT_CSV = OUT_DIR / "filtered_pubmed.csv"

MIN_LIMITATIONS_WORDS = 15
MAX_LIMITATIONS_WORDS = 500   # relaxed from a strict 100 — most real sections are 100-400
MIN_RELEVANCE_SCORE = 3       # lenient. raise to 5+ if corpus is too large
BATCH_SIZE = 10

JOURNAL_KEYWORDS = [
    # high-signal medical AI / informatics venues
    "lancet digital", "lancet", "nejm", "nature medicine",
    "jama network open", "jama internal", "jama pediatr", "jama cardio",
    "npj digital", "plos medicine", "plos digital", "plos one",
    "digital health", "digital medicine",
    "informatics", "biomedical informatics", "medical informatics",
    "artificial intelligence in medicine", "jmir",
    "health affairs", "bmj health", "bmj quality",
    "american journal of public health", "ajph",
    "annals of internal medicine", "annals of family medicine",
    "circulation", "radiology",
]

TECH_KEYWORDS = [
    "machine learning", "deep learning", "artificial intelligence",
    "neural network", "convolutional", "transformer", "predictive model",
    "diagnostic algorithm", "clinical decision support", "decision support",
    "natural language processing", "computer vision",
    "ai-based", "ml model", "ai model", "prediction model",
    "automated detection", "automated diagnosis",
    "telemedicine", "wearable", "ehr", "electronic health record",
    "risk score", "risk stratification",
]

HEALTH_KEYWORDS = [
    "public health", "epidemiolog", "health equity", "disparit",
    "demograph", "clinical trial", "retrospective cohort", "prospective cohort",
    "patient outcome", "hospital", "intervention", "diagnos",
    "treatment", "screening", "disease prevalence",
    "minority", "rural", "underserved", "vulnerable",
    "primary care", "emergency department", "icu",
    "mortality", "morbidity", "readmission",
]


# ---------------------------------------------------------------------------
# Filters & scoring
# ---------------------------------------------------------------------------
def word_count(text) -> int:
    if not isinstance(text, str):
        return 0
    return len(text.split())


def keyword_hits(text: str, keywords: list[str]) -> int:
    if not isinstance(text, str):
        return 0
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


def relevance_score(row: dict) -> tuple[int, int, int]:
    """Return (score, tech_hits, health_hits)."""
    score = 0

    if any(kw in str(row.get("journal", "")).lower() for kw in JOURNAL_KEYWORDS):
        score += 3

    title = str(row.get("title", "") or "")
    abstract = str(row.get("abstract", "") or "")
    kw_col = str(row.get("keywords", "") or "") + " " + str(row.get("subjects", "") or "")
    combined = " ".join([title, abstract, kw_col]).lower()

    tech = keyword_hits(combined, TECH_KEYWORDS)
    health = keyword_hits(combined, HEALTH_KEYWORDS)

    score += min(tech, 3)
    score += min(health, 3)
    if tech > 0 and health > 0:
        score += 2

    return score, tech, health


def filter_row(row: dict) -> tuple[bool, int, int]:
    """Return (keep, score, word_count)."""
    lim = row.get("limitations_section")
    if not isinstance(lim, str) or not lim.strip():
        return False, 0, 0
    wc = word_count(lim)
    if wc < MIN_LIMITATIONS_WORDS or wc > MAX_LIMITATIONS_WORDS:
        return False, 0, wc
    score, _, _ = relevance_score(row)
    if score < MIN_RELEVANCE_SCORE:
        return False, score, wc
    return True, score, wc


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def write_batch(batch: list[dict], header_written: bool) -> bool:
    if not batch:
        return header_written
    pd.DataFrame(batch).to_csv(
        OUTPUT_CSV, mode="a", header=not header_written, index=False
    )
    return True


def process_file(fp: Path, seen_pmids: set) -> list[dict]:
    try:
        df = pd.read_csv(fp, compression="gzip", low_memory=False)
    except Exception as e:
        print(f"  [error] {fp.name}: {e}")
        return []

    initial = len(df)
    kept = []
    for row in df.to_dict(orient="records"):
        pmid = str(row.get("pmid", ""))
        if not pmid or pmid in seen_pmids:
            continue
        keep, score, wc = filter_row(row)
        if not keep:
            continue
        row["relevance_score"] = score
        row["limitations_word_count"] = wc
        kept.append(row)
        seen_pmids.add(pmid)
    print(f"  {fp.name}: {initial:,} → {len(kept):,} kept")
    return kept


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(PUBMED_DIR.glob("*.csv.gz"))
    print(f"Found {len(files)} .csv.gz files")

    if OUTPUT_CSV.exists():
        print(f"[note] {OUTPUT_CSV} exists. Delete it to start fresh, "
              "or rename to keep both. Aborting.")
        return

    seen_pmids: set = set()
    pending: list[dict] = []
    header_written = False
    total_kept = 0

    for i, fp in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {fp.name}")
        rows = process_file(fp, seen_pmids)
        if not rows:
            continue
        pending.extend(rows)
        while len(pending) >= BATCH_SIZE:
            header_written = write_batch(pending[:BATCH_SIZE], header_written)
            total_kept += BATCH_SIZE
            pending = pending[BATCH_SIZE:]

    # Final flush
    if pending:
        header_written = write_batch(pending, header_written)
        total_kept += len(pending)

    print(f"\nTotal kept: {total_kept:,}")
    print(f"Output:     {OUTPUT_CSV}")

    if total_kept > 0:
        out = pd.read_csv(OUTPUT_CSV)
        print(f"\nTop 10 journals:")
        print(out["journal"].value_counts().head(10).to_string())
        print(f"\nRelevance score distribution:")
        print(out["relevance_score"].value_counts().sort_index().to_string())
        print(f"\nLimitations word count: "
              f"median={out['limitations_word_count'].median():.0f}, "
              f"p10={out['limitations_word_count'].quantile(0.1):.0f}, "
              f"p90={out['limitations_word_count'].quantile(0.9):.0f}")


if __name__ == "__main__":
    main()