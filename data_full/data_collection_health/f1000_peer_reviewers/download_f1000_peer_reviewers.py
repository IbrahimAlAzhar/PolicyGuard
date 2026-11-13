#!/usr/bin/env python3
"""
Download biomedical/public-health open peer-review reports from F1000-style
platforms using Crossref + Europe PMC + publisher JATS XML.

Why this source:
  Crossref records peer-review DOIs as type=peer-review and links them to the
  reviewed paper DOI through relation.is-review-of. F1000-style platforms expose
  article JATS XML at:

      https://.../articles/{volume}-{article}/v{version}/xml

  That XML contains reviewer reports as:

      <sub-article article-type="reviewer-report">

Example:
  python download_f1000_peer_reviews.py \
    --max-review-records 5000 \
    --out-dir f1000_health_peer_reviews \
    --mailto you@example.edu

Outputs:
  crossref_peer_review_index.jsonl  - Crossref review DOI -> article DOI records
  f1000_articles_with_reviews.jsonl - parsed article + reviewer-report text
  skipped.jsonl                     - records skipped due to filter/fetch errors
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any


CROSSREF_PREFIX_URL = "https://api.crossref.org/prefixes/{prefix}/works"
CROSSREF_TYPES_URL = "https://api.crossref.org/types/peer-review/works"
EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

F1000_REVIEW_PREFIX = "10.5256"
F1000_ARTICLE_PREFIX = "10.12688/"
SSL_CONTEXT: ssl.SSLContext | None = None


def request_text(url: str, timeout: int = 45) -> tuple[str, str]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "open-peer-review-harvester/0.1 "
            "(mailto supplied via Crossref API when configured)"
        },
    )
    with urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT) as response:
        body = response.read().decode("utf-8", errors="replace")
        return body, response.geturl()


def request_json(url: str, params: dict[str, Any], timeout: int = 45) -> dict[str, Any]:
    full_url = url + "?" + urllib.parse.urlencode(params, doseq=True)
    text, _ = request_text(full_url, timeout=timeout)
    return json.loads(text)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def first(values: Any, default: str = "") -> str:
    if isinstance(values, list) and values:
        return str(values[0])
    if isinstance(values, str):
        return values
    return default


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def element_text(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    return re.sub(r"\s+", " ", "".join(elem.itertext())).strip()


def find_first_text(elem: ET.Element, tag_name: str) -> str:
    for child in elem.iter():
        if local_name(child.tag) == tag_name:
            return element_text(child)
    return ""


def reviewed_dois(crossref_item: dict[str, Any]) -> list[str]:
    relation = crossref_item.get("relation") or {}
    vals = relation.get("is-review-of") or relation.get("isReviewOf") or []
    dois = []
    for value in vals:
        if isinstance(value, dict) and value.get("id-type") == "doi" and value.get("id"):
            doi = str(value["id"]).strip().lower()
            if doi not in dois:
                dois.append(doi)
    return dois


def iter_crossref_peer_reviews(args: argparse.Namespace):
    cursor = "*"
    seen = 0
    endpoint = (
        CROSSREF_PREFIX_URL.format(prefix=urllib.parse.quote(args.prefix, safe=""))
        if args.prefix
        else CROSSREF_TYPES_URL
    )
    while seen < args.max_review_records:
        rows = min(args.rows, args.max_review_records - seen)
        params: dict[str, Any] = {
            "rows": rows,
            "cursor": cursor,
            "select": "DOI,title,relation,URL,type,publisher,created,issued,published-online",
        }
        if args.prefix:
            params["filter"] = "type:peer-review"
        if args.query:
            params["query"] = args.query
        if args.from_date:
            existing = params.get("filter")
            date_filter = f"from-pub-date:{args.from_date}"
            params["filter"] = f"{existing},{date_filter}" if existing else date_filter
        if args.mailto:
            params["mailto"] = args.mailto

        data = request_json(endpoint, params)
        message = data.get("message") or {}
        items = message.get("items") or []
        if not items:
            break
        for item in items:
            seen += 1
            yield item
            if seen >= args.max_review_records:
                break
        next_cursor = message.get("next-cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(args.sleep)


def europe_pmc_lookup(doi: str) -> dict[str, Any] | None:
    params = {
        "query": f'DOI:"{doi}"',
        "format": "json",
        "resultType": "core",
        "pageSize": 1,
    }
    data = request_json(EUROPE_PMC_SEARCH_URL, params)
    if int(data.get("hitCount") or 0) < 1:
        return None
    results = ((data.get("resultList") or {}).get("result") or [])
    return results[0] if results else None


def resolve_doi_url(doi: str) -> str:
    quoted = urllib.parse.quote(doi, safe="/.")
    _, final_url = request_text(f"https://doi.org/{quoted}")
    return final_url.split("?", 1)[0].rstrip("/")


def f1000_xml_url(article_doi: str) -> str:
    final_url = resolve_doi_url(article_doi)
    if not re.search(r"/v\d+$", final_url):
        raise ValueError(f"Resolved URL does not look like a versioned F1000 article: {final_url}")
    return final_url + "/xml"


def parse_f1000_xml(xml_text: str) -> dict[str, Any]:
    root = ET.fromstring(xml_text)

    article_title = ""
    body_text = ""
    for child in root:
        if local_name(child.tag) == "front":
            article_title = find_first_text(child, "article-title")
        elif local_name(child.tag) == "body":
            body_text = element_text(child)

    reports = []
    for sub in root:
        if local_name(sub.tag) != "sub-article":
            continue
        article_type = sub.attrib.get("article-type", "")
        if article_type not in {"reviewer-report", "author-comment", "editor-report"}:
            continue

        report_doi = ""
        title = ""
        reviewer_names: list[str] = []
        for elem in sub.iter():
            lname = local_name(elem.tag)
            if lname == "article-id" and elem.attrib.get("pub-id-type") == "doi":
                report_doi = element_text(elem).lower()
            elif lname == "article-title" and not title:
                title = element_text(elem)
            elif lname == "name":
                surname = find_first_text(elem, "surname")
                given = find_first_text(elem, "given-names")
                name = " ".join(part for part in [given, surname] if part).strip()
                if name and name not in reviewer_names:
                    reviewer_names.append(name)

        reports.append(
            {
                "report_type": article_type,
                "report_id": sub.attrib.get("id", ""),
                "report_doi": report_doi,
                "report_title": title,
                "reviewer_names": reviewer_names,
                "report_text": element_text(sub),
            }
        )

    return {
        "article_title_xml": article_title,
        "article_body_text": body_text,
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-review-records", type=int, default=5000)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--mailto", default="")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification. Use only if Python cannot find CA certificates on your cluster/container.",
    )
    parser.add_argument("--from-date", default="")
    parser.add_argument("--query", default="")
    parser.add_argument(
        "--prefix",
        default=F1000_REVIEW_PREFIX,
        help="Crossref DOI prefix to harvest. Default 10.5256 = F1000 peer-review reports. Use empty string for all Crossref peer-review records.",
    )
    parser.add_argument(
        "--no-europe-pmc-filter",
        action="store_true",
        help="Do not require the reviewed article DOI to exist in Europe PMC.",
    )
    args = parser.parse_args()

    global SSL_CONTEXT
    if args.insecure:
        SSL_CONTEXT = ssl._create_unverified_context()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out_dir / "crossref_peer_review_index.jsonl"
    output_path = args.out_dir / "f1000_articles_with_reviews.jsonl"
    skipped_path = args.out_dir / "skipped.jsonl"

    article_to_reviews: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in iter_crossref_peer_reviews(args):
        review_doi = str(item.get("DOI", "")).lower()
        for article_doi in reviewed_dois(item):
            index_record = {
                "review_doi": review_doi,
                "review_title": first(item.get("title")),
                "review_url": item.get("URL", ""),
                "review_publisher": item.get("publisher", ""),
                "article_doi": article_doi,
                "crossref_item": item,
            }
            append_jsonl(index_path, index_record)
            article_to_reviews[article_doi].append(index_record)

    for i, (article_doi, review_records) in enumerate(article_to_reviews.items(), 1):
        try:
            if not article_doi.startswith(F1000_ARTICLE_PREFIX):
                append_jsonl(
                    skipped_path,
                    {"article_doi": article_doi, "reason": "not_f1000_style_article_doi"},
                )
                continue

            epmc = None
            if not args.no_europe_pmc_filter:
                epmc = europe_pmc_lookup(article_doi)
                if epmc is None:
                    append_jsonl(
                        skipped_path,
                        {"article_doi": article_doi, "reason": "not_found_in_europe_pmc"},
                    )
                    continue
                time.sleep(args.sleep)

            xml_url = f1000_xml_url(article_doi)
            xml_text, _ = request_text(xml_url)
            parsed = parse_f1000_xml(xml_text)
            record = {
                "article_doi": article_doi,
                "article_url": xml_url.rsplit("/xml", 1)[0],
                "xml_url": xml_url,
                "crossref_review_records": review_records,
                "europe_pmc": epmc,
                **parsed,
            }
            append_jsonl(output_path, record)
            time.sleep(args.sleep)
        except (urllib.error.URLError, TimeoutError, ValueError, ET.ParseError, json.JSONDecodeError) as exc:
            append_jsonl(
                skipped_path,
                {
                    "article_doi": article_doi,
                    "reason": type(exc).__name__,
                    "detail": str(exc)[:500],
                },
            )

        if i % 100 == 0:
            print(f"processed {i} unique article DOIs", flush=True)

    print(f"Wrote: {index_path}")
    print(f"Wrote: {output_path}")
    print(f"Wrote: {skipped_path}")


if __name__ == "__main__":
    main()
