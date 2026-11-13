#!/usr/bin/env python3
"""
Flatten the open-health peer-review harvest into ONE unified CSV.

Granularity: one row per reviewer report / review artifact (comment-level).
Text: full extracted review text is included in the `review_text` column.

It reads the JSONL produced by download_open_health_peer_reviews.py:
  <out-dir>/records/health_peer_review_records.jsonl   (primary)
  <out-dir>/records/f1000_structured_reports.jsonl      (optional, merged in)

For each review record it emits one row per:
  - F1000/Open Research structured report  (structured.f1000.reports[])
  - eLife review section                    (structured.elife.review_sections[])
  - discovered peer-review artifact         (discovered_peer_artifacts[])
When a record has none of the above, it emits a single metadata-only fallback
row so nothing is silently dropped.

With --extract-files, review_text for discovered artifacts is pulled from the
downloaded raw HTML/XML/PDF on disk (tags stripped). HTML/XML use stdlib; PDF
uses pypdf or pdfminer if installed, otherwise that cell is left empty.

Usage:
  python peer_reviews_to_csv.py \
    --out-dir data/health/open_peer_reviews/all_sources \
    --csv unified_peer_reviews.csv \
    --extract-files

  # or point directly at one or more jsonl files:
  python peer_reviews_to_csv.py \
    --records-file a.jsonl --records-file b.jsonl --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

COLUMNS = [
    "row_type",            # f1000_report | elife_section | discovered_artifact | metadata_only
    "review_doi",
    "review_url",
    "review_title",
    "review_publisher",
    "article_doi",
    "article_title",
    "article_journal",
    "article_year",
    "article_pmid",
    "article_pmcid",
    "report_type",         # reviewer-report / author-comment / editor-report / heading / anchor text
    "report_doi",
    "reviewer_names",
    "review_text",
    "review_text_chars",
    "artifact_url",
    "source_file_path",    # raw file on disk this text came from, if any
    "article_abstract",
]


def clean_text(value: str) -> str:
    if not value:
        return ""
    value = TAG_RE.sub(" ", value)
    value = html.unescape(value)
    return WS_RE.sub(" ", value).strip()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def extract_file_text(path_str: str, max_chars: int) -> str:
    if not path_str:
        return ""
    path = Path(path_str)
    if not path.exists():
        return ""
    suffix = path.suffix.lower()
    try:
        if suffix in {".html", ".htm", ".xml", ".txt", ".json"}:
            raw = path.read_text(encoding="utf-8", errors="replace")
            return clean_text(raw)[:max_chars]
        if suffix == ".pdf":
            return extract_pdf_text(path)[:max_chars]
    except Exception:  # noqa: BLE001
        return ""
    return ""


def extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        return clean_text(" ".join((page.extract_text() or "") for page in reader.pages))
    except Exception:  # noqa: BLE001
        pass
    try:
        from pdfminer.high_level import extract_text  # type: ignore

        return clean_text(extract_text(str(path)))
    except Exception:  # noqa: BLE001
        return ""


def base_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "review_doi": record.get("review_doi", ""),
        "review_url": record.get("review_url", ""),
        "review_title": record.get("review_title", ""),
        "review_publisher": record.get("review_publisher", ""),
        "article_doi": record.get("article_doi", ""),
        "article_title": record.get("article_title", ""),
        "article_journal": record.get("article_journal", ""),
        "article_year": record.get("article_year", ""),
        "article_pmid": record.get("article_pmid", ""),
        "article_pmcid": record.get("article_pmcid", ""),
        "article_abstract": clean_text(record.get("article_abstract", "")),
    }


def finalize(row: dict[str, Any]) -> dict[str, Any]:
    text = row.get("review_text", "") or ""
    row["review_text"] = text
    row["review_text_chars"] = len(text)
    return {col: row.get(col, "") for col in COLUMNS}


def rows_from_record(
    record: dict[str, Any],
    *,
    extract_files: bool,
    max_chars: int,
) -> list[dict[str, Any]]:
    base = base_row(record)
    rows: list[dict[str, Any]] = []
    structured = record.get("structured") or {}

    # 1) F1000 / Open Research structured reports.
    f1000 = structured.get("f1000") or {}
    for report in f1000.get("reports") or []:
        row = dict(base)
        row.update(
            {
                "row_type": "f1000_report",
                "report_type": report.get("report_type", ""),
                "report_doi": report.get("report_doi", ""),
                "reviewer_names": "; ".join(report.get("reviewer_names") or []),
                "review_text": clean_text(report.get("report_text", "")),
                "artifact_url": "",
                "source_file_path": f1000.get("xml_path", ""),
            }
        )
        rows.append(row)

    # 2) eLife review sections.
    elife = structured.get("elife") or {}
    for section in elife.get("review_sections") or []:
        row = dict(base)
        row.update(
            {
                "row_type": "elife_section",
                "report_type": section.get("heading", ""),
                "report_doi": "",
                "reviewer_names": "",
                "review_text": clean_text(section.get("text", "")),
                "artifact_url": "",
                "source_file_path": "",
            }
        )
        rows.append(row)

    # 3) Discovered peer-review artifacts (BMJ/BMC/Nature/etc.).
    for artifact in record.get("discovered_peer_artifacts") or []:
        download = artifact.get("download") or {}
        path = download.get("path", "") if download.get("ok") else ""
        text = extract_file_text(path, max_chars) if (extract_files and path) else ""
        row = dict(base)
        row.update(
            {
                "row_type": "discovered_artifact",
                "report_type": artifact.get("anchor_text", ""),
                "report_doi": "",
                "reviewer_names": "",
                "review_text": text,
                "artifact_url": artifact.get("url", ""),
                "source_file_path": path,
            }
        )
        rows.append(row)

    # 4) Fallback so no record is dropped.
    if not rows:
        row = dict(base)
        row.update(
            {
                "row_type": "metadata_only",
                "report_type": "",
                "report_doi": "",
                "reviewer_names": "",
                "review_text": "",
                "artifact_url": "",
                "source_file_path": "",
            }
        )
        rows.append(row)

    return [finalize(r) for r in rows]


def default_record_files(out_dir: Path) -> list[Path]:
    records = out_dir / "records"
    candidates = [
        records / "health_peer_review_records.jsonl",
        records / "f1000_structured_reports.jsonl",
    ]
    return [p for p in candidates if p.exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, help="Harvest output dir (expects records/ subfolder).")
    parser.add_argument(
        "--records-file",
        type=Path,
        action="append",
        default=[],
        help="Explicit JSONL file(s) to read. Repeatable. Overrides --out-dir discovery.",
    )
    parser.add_argument("--csv", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--extract-files",
        action="store_true",
        help="Pull review text for discovered artifacts from raw HTML/XML/PDF on disk.",
    )
    parser.add_argument("--max-chars", type=int, default=200000, help="Max review_text chars per cell.")
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop duplicate rows keyed on (review_doi, report_doi, report_type, artifact_url, first 200 text chars).",
    )
    args = parser.parse_args()

    files = list(args.records_file)
    if not files and args.out_dir:
        files = default_record_files(args.out_dir)
    if not files:
        parser.error("No input files. Pass --out-dir <dir> or one or more --records-file <path>.")

    args.csv.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple] = set()
    n_records = 0
    n_rows = 0
    by_type: dict[str, int] = {}

    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for path in files:
            print(f"reading {path}", flush=True)
            for record in read_jsonl(path):
                # f1000_structured_reports.jsonl rows are not full records; normalize them.
                if "structured" not in record and "reports" in record:
                    record = {
                        "review_doi": record.get("article_doi", ""),
                        "article_doi": record.get("article_doi", ""),
                        "article_title": record.get("article_title", ""),
                        "structured": {"f1000": record},
                    }
                n_records += 1
                for row in rows_from_record(record, extract_files=args.extract_files, max_chars=args.max_chars):
                    if args.dedupe:
                        key = (
                            row["review_doi"],
                            row["report_doi"],
                            row["report_type"],
                            row["artifact_url"],
                            row["review_text"][:200],
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                    writer.writerow(row)
                    n_rows += 1
                    by_type[row["row_type"]] = by_type.get(row["row_type"], 0) + 1

    print(f"\ndone. records read={n_records}  rows written={n_rows}", flush=True)
    for row_type, count in sorted(by_type.items()):
        print(f"  {row_type}: {count}", flush=True)
    print(f"csv: {args.csv}", flush=True)


if __name__ == "__main__":
    main()