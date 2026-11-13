#!/usr/bin/env python3
"""
Download health/biomedical/public-health papers plus open peer-review comments.

Main idea:
  1. Crossref indexes many public peer-review records as type=peer-review.
  2. Each review record usually links to the reviewed paper DOI via relation.is-review-of.
  3. Europe PMC is used as the biomedical/health filter for the reviewed paper DOI.
  4. The script downloads:
       - Crossref peer-review metadata
       - Europe PMC paper metadata
       - review DOI landing pages/files
       - article DOI landing pages
       - discovered peer-review artifacts such as "Peer Review File",
         "reviewer reports", "author response", "decision letter",
         "pre-publication history", etc.
  5. For F1000/Open Research articles, it additionally downloads JATS XML and
     extracts structured <sub-article article-type="reviewer-report"> text.

This is intentionally source-agnostic. It catches F1000/Open Research,
eLife-like review pages, BMJ/BMC/Nature peer-review files when discoverable,
and any other publisher that registers peer-review records in Crossref.

Example:
  python download_open_health_peer_reviews.py \
    --out-dir /lstr/sahara/datalab-ml/ibrahim/limagents_update/socio_limit_shield/data/health/open_peer_reviews/all_sources \
    --max-records 200000 \
    --mailto z2006888@metis.niu.edu

Targeted high-yield run:
  python download_open_health_peer_reviews.py \
    --out-dir data/health/open_peer_reviews/health_query \
    --query "public health" --query medicine --query biomedical --query epidemiology \
    --max-records 50000 \
    --mailto z2006888@metis.niu.edu

If your cluster Python has broken CA certificates:
  add --insecure
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import mimetypes
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


CROSSREF_PEER_REVIEW_URL = "https://api.crossref.org/types/peer-review/works"
CROSSREF_PREFIX_URL = "https://api.crossref.org/prefixes/{prefix}/works"
EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

F1000_REVIEW_PREFIX = "10.5256/"
F1000_ARTICLE_PREFIX = "10.12688/"
ELIFE_PREFIX = "10.7554/elife."

SSL_CONTEXT: ssl.SSLContext | None = None

SOCIAL_DEMOGRAPHIC_TERMS = re.compile(
    r"(public health|health equity|equity|inequit|disparit|socio.?economic|income|poverty|"
    r"deprivation|education|employment|occupation|race|racial|ethnic|ethnicity|minority|"
    r"underserved|under.?represented|gender|sex|women|men|transgender|lgbt|msm|age|older|"
    r"adolescent|children|rural|urban|culture|cultural|migrant|immigrant|refugee|"
    r"indigenous|native|tribal|low.?resource|low.?income|middle.?income|lmics?)",
    re.I,
)

PEER_ARTIFACT_TERMS = re.compile(
    r"(peer.?review|reviewer.?report|review.?report|referee.?report|review.?file|"
    r"author.?response|response.?to.?reviewers|decision.?letter|editor.?decision|"
    r"pre.?publication.?history|review.?history|transparent.?peer.?review|"
    r"public.?review|elife.?assessment)",
    re.I,
)

SKIP_URL_RE = re.compile(
    r"(facebook|twitter|linkedin|reddit|mendeley|mailto:|javascript:|#|/login|/signin)",
    re.I,
)


def log(message: str) -> None:
    print(message, flush=True)


def safe_name(value: str, max_len: int = 160) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    if not cleaned:
        cleaned = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return cleaned[:max_len]


def sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def request_bytes(
    url: str,
    timeout: int = 60,
    accept: str = "*/*",
    user_agent: str = "open-health-peer-review-harvester/0.1",
) -> tuple[bytes, str, str]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": accept,
        },
    )
    with urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT) as response:
        data = response.read()
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
        return data, final_url, content_type


def request_text(url: str, timeout: int = 60, accept: str = "*/*") -> tuple[str, str, str]:
    data, final_url, content_type = request_bytes(url, timeout=timeout, accept=accept)
    encoding = "utf-8"
    match = re.search(r"charset=([^;\s]+)", content_type, re.I)
    if match:
        encoding = match.group(1).strip()
    return data.decode(encoding, errors="replace"), final_url, content_type


def request_json(url: str, params: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    full_url = url + "?" + urllib.parse.urlencode(params, doseq=True)
    text, _, _ = request_text(full_url, timeout=timeout, accept="application/json")
    return json.loads(text)


def write_download(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)


def extension_from_url_or_type(url: str, content_type: str) -> str:
    path = urllib.parse.urlparse(url).path
    suffix = Path(path).suffix.lower()
    if suffix in {".html", ".htm", ".xml", ".pdf", ".doc", ".docx", ".txt", ".json"}:
        return suffix
    guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip())
    if guessed:
        return guessed
    if "html" in content_type:
        return ".html"
    if "xml" in content_type:
        return ".xml"
    if "pdf" in content_type:
        return ".pdf"
    return ".bin"


def download_url(url: str, out_dir: Path, stem: str, sleep: float) -> dict[str, Any]:
    try:
        data, final_url, content_type = request_bytes(url)
        ext = extension_from_url_or_type(final_url, content_type)
        path = out_dir / f"{safe_name(stem)}_{sha1(final_url)[:10]}{ext}"
        write_download(path, data)
        time.sleep(sleep)
        return {
            "ok": True,
            "url": url,
            "final_url": final_url,
            "content_type": content_type,
            "path": str(path),
            "bytes": len(data),
        }
    except Exception as exc:  # noqa: BLE001 - log and continue for harvesting.
        return {
            "ok": False,
            "url": url,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }


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
    dois: list[str] = []
    for value in vals:
        if isinstance(value, dict) and value.get("id-type") == "doi" and value.get("id"):
            doi = str(value["id"]).strip().lower()
            if doi not in dois:
                dois.append(doi)
    return dois


def iter_crossref_peer_reviews(
    *,
    max_records: int,
    rows: int,
    sleep: float,
    mailto: str,
    query: str,
    prefix: str,
    from_date: str,
) -> Iterable[dict[str, Any]]:
    cursor = "*"
    seen = 0
    if prefix:
        endpoint = CROSSREF_PREFIX_URL.format(prefix=urllib.parse.quote(prefix, safe=""))
    else:
        endpoint = CROSSREF_PEER_REVIEW_URL

    while seen < max_records:
        params: dict[str, Any] = {
            "rows": min(rows, max_records - seen),
            "cursor": cursor,
            "select": "DOI,title,relation,URL,type,publisher,container-title,created,issued,published-online,link",
        }
        filters = []
        if prefix:
            filters.append("type:peer-review")
        if from_date:
            filters.append(f"from-pub-date:{from_date}")
        if filters:
            params["filter"] = ",".join(filters)
        if query:
            params["query"] = query
        if mailto:
            params["mailto"] = mailto

        data = request_json(endpoint, params)
        message = data.get("message") or {}
        items = message.get("items") or []
        if not items:
            break

        for item in items:
            seen += 1
            yield item
            if seen >= max_records:
                break

        next_cursor = message.get("next-cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(sleep)


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


def epmc_text(epmc: dict[str, Any]) -> str:
    pieces = [
        epmc.get("title", ""),
        epmc.get("abstractText", ""),
        epmc.get("journalTitle", ""),
        epmc.get("authorString", ""),
    ]
    keywords = epmc.get("keywordList", {}).get("keyword", []) if isinstance(epmc.get("keywordList"), dict) else []
    if isinstance(keywords, list):
        pieces.extend(str(x) for x in keywords)
    return " ".join(str(x) for x in pieces if x)


def resolve_doi_url(doi: str) -> str:
    quoted = urllib.parse.quote(doi, safe="/.")
    _, final_url, _ = request_text(f"https://doi.org/{quoted}", accept="text/html,*/*")
    return final_url.split("#", 1)[0].rstrip("/")


def discover_peer_artifact_links(base_url: str, html_text: str) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    seen: set[str] = set()
    # Simple but practical: collect anchor href + text.
    for match in re.finditer(
        r"<a\b[^>]*?href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
        html_text,
        flags=re.I | re.S,
    ):
        href = html.unescape(match.group(1)).strip()
        anchor_text = re.sub(r"<[^>]+>", " ", match.group(2))
        anchor_text = html.unescape(re.sub(r"\s+", " ", anchor_text)).strip()
        joined = " ".join([href, anchor_text])
        if SKIP_URL_RE.search(href):
            continue
        if not PEER_ARTIFACT_TERMS.search(joined):
            continue
        url = urllib.parse.urljoin(base_url, href)
        if url in seen:
            continue
        seen.add(url)
        links.append({"url": url, "anchor_text": anchor_text})

    # Some pages expose artifact URLs in JSON, not anchors.
    for match in re.finditer(r"https?://[^\"'<>\\\s]+", html_text):
        url = html.unescape(match.group(0))
        if url in seen or SKIP_URL_RE.search(url):
            continue
        if PEER_ARTIFACT_TERMS.search(url):
            seen.add(url)
            links.append({"url": url, "anchor_text": ""})
    return links


def f1000_xml_url(article_doi: str) -> str:
    article_url = resolve_doi_url(article_doi)
    if not re.search(r"/v\d+$", article_url):
        raise ValueError(f"Resolved URL does not look like a versioned F1000 article: {article_url}")
    return article_url + "/xml"


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


def try_f1000_structured(article_doi: str, out_dir: Path, sleep: float) -> dict[str, Any] | None:
    if not article_doi.startswith(F1000_ARTICLE_PREFIX):
        return None
    xml_url = f1000_xml_url(article_doi)
    xml_bytes, final_url, content_type = request_bytes(xml_url, accept="application/xml,text/xml,*/*")
    xml_path = out_dir / "raw" / "f1000_xml" / f"{safe_name(article_doi)}.xml"
    write_download(xml_path, xml_bytes)
    parsed = parse_f1000_xml(xml_bytes.decode("utf-8", errors="replace"))
    time.sleep(sleep)
    return {
        "xml_url": xml_url,
        "xml_final_url": final_url,
        "xml_content_type": content_type,
        "xml_path": str(xml_path),
        **parsed,
    }


def try_elife_review_text(article_doi: str, out_dir: Path, sleep: float) -> dict[str, Any] | None:
    # eLife DOIs are usually 10.7554/eLife.NNNNN.
    if not article_doi.lower().startswith(ELIFE_PREFIX):
        return None
    article_id = article_doi.rsplit(".", 1)[-1]
    candidates = [
        f"https://elifesciences.org/articles/{article_id}.xml",
        f"https://elifesciences.org/articles/{article_id}",
    ]
    downloaded = []
    review_sections = []
    for url in candidates:
        result = download_url(url, out_dir / "raw" / "elife", f"elife_{article_id}", sleep)
        downloaded.append(result)
        if not result.get("ok"):
            continue
        path = Path(str(result["path"]))
        text = path.read_text(encoding="utf-8", errors="replace")
        # Extract broad review-related sections. Exact eLife markup changed over time.
        for heading in [
            "eLife assessment",
            "Public Reviews",
            "Decision letter",
            "Author response",
            "Reviewer #",
            "Reviewing Editor",
        ]:
            idx = text.lower().find(heading.lower())
            if idx >= 0:
                snippet = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text[idx : idx + 20000])).strip()
                review_sections.append({"heading": heading, "text": html.unescape(snippet)})
    if not review_sections:
        return {"downloads": downloaded, "review_sections": []}
    return {"downloads": downloaded, "review_sections": review_sections}


def collect_for_review_record(
    *,
    crossref_item: dict[str, Any],
    article_doi: str,
    epmc: dict[str, Any],
    out_dir: Path,
    sleep: float,
    download_artifacts: bool,
) -> dict[str, Any]:
    review_doi = str(crossref_item.get("DOI", "")).lower()
    review_url = str(crossref_item.get("URL") or f"https://doi.org/{review_doi}")
    title = first(crossref_item.get("title"))
    stem = f"{review_doi}_{article_doi}"

    record: dict[str, Any] = {
        "source": "crossref_type_peer_review",
        "review_doi": review_doi,
        "review_url": review_url,
        "review_title": title,
        "review_publisher": crossref_item.get("publisher", ""),
        "article_doi": article_doi,
        "article_title": epmc.get("title", ""),
        "article_abstract": epmc.get("abstractText", ""),
        "article_journal": epmc.get("journalTitle", ""),
        "article_year": epmc.get("pubYear", ""),
        "article_pmid": epmc.get("pmid", ""),
        "article_pmcid": epmc.get("pmcid", ""),
        "europe_pmc": epmc,
        "crossref": crossref_item,
        "downloads": {},
        "discovered_peer_artifacts": [],
        "structured": {},
    }

    if not download_artifacts:
        return record

    # Download review landing page or file.
    record["downloads"]["review"] = download_url(
        review_url,
        out_dir / "raw" / "review_pages",
        f"review_{stem}",
        sleep,
    )

    # Download article DOI page and discover peer-review files.
    try:
        article_url = resolve_doi_url(article_doi)
        article_download = download_url(
            article_url,
            out_dir / "raw" / "article_pages",
            f"article_{article_doi}",
            sleep,
        )
        record["downloads"]["article"] = article_download

        if article_download.get("ok") and str(article_download.get("content_type", "")).lower().find("html") >= 0:
            article_html = Path(str(article_download["path"])).read_text(encoding="utf-8", errors="replace")
            links = discover_peer_artifact_links(str(article_download.get("final_url") or article_url), article_html)
            for link in links[:50]:
                artifact_download = download_url(
                    link["url"],
                    out_dir / "raw" / "peer_artifacts",
                    f"artifact_{article_doi}",
                    sleep,
                )
                record["discovered_peer_artifacts"].append(
                    {
                        **link,
                        "download": artifact_download,
                    }
                )
    except Exception as exc:  # noqa: BLE001
        record["downloads"]["article"] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }

    # Source-specific structured extraction.
    try:
        f1000 = try_f1000_structured(article_doi, out_dir, sleep)
        if f1000:
            record["structured"]["f1000"] = f1000
    except Exception as exc:  # noqa: BLE001
        record["structured"]["f1000_error"] = {
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }

    try:
        elife = try_elife_review_text(article_doi, out_dir, sleep)
        if elife:
            record["structured"]["elife"] = elife
    except Exception as exc:  # noqa: BLE001
        record["structured"]["elife_error"] = {
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }

    return record


def load_seen(path: Path, key: str) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = record.get(key)
            if value:
                seen.add(str(value).lower())
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=100000)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--mailto", default="")
    parser.add_argument("--from-date", default="")
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Crossref search query. Can be repeated. Empty means no query.",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        help=(
            "Crossref DOI prefix. Can be repeated. Use 10.5256 for F1000 peer-review reports. "
            "If omitted, searches all Crossref type=peer-review records."
        ),
    )
    parser.add_argument(
        "--require-social-keyword",
        action="store_true",
        help="Keep only Europe PMC hits whose title/abstract/journal contains demographic/social-equity terms.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only write Crossref + Europe PMC metadata; do not download pages/artifacts.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification. Use only if Python cannot find CA certificates on the cluster.",
    )
    args = parser.parse_args()

    global SSL_CONTEXT
    if args.insecure:
        SSL_CONTEXT = ssl._create_unverified_context()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / "records" / "health_peer_review_records.jsonl"
    skipped_path = args.out_dir / "records" / "skipped.jsonl"
    f1000_path = args.out_dir / "records" / "f1000_structured_reports.jsonl"
    index_path = args.out_dir / "records" / "crossref_peer_review_index.jsonl"

    seen_review_dois = load_seen(records_path, "review_doi")
    seen_index = load_seen(index_path, "review_doi")

    queries = args.query or [""]
    prefixes = args.prefix or [""]
    processed = 0
    kept = 0
    epmc_cache: dict[str, dict[str, Any] | None] = {}

    for prefix in prefixes:
        for query in queries:
            log(f"=== Crossref harvest prefix={prefix or 'ALL'} query={query or 'NONE'} ===")
            try:
                iterator = iter_crossref_peer_reviews(
                    max_records=args.max_records,
                    rows=args.rows,
                    sleep=args.sleep,
                    mailto=args.mailto,
                    query=query,
                    prefix=prefix,
                    from_date=args.from_date,
                )
                for item in iterator:
                    review_doi = str(item.get("DOI", "")).lower()
                    if review_doi in seen_index:
                        continue

                    article_dois = reviewed_dois(item)
                    if not article_dois:
                        append_jsonl(
                            skipped_path,
                            {"review_doi": review_doi, "reason": "no_relation_is_review_of", "crossref": item},
                        )
                        seen_index.add(review_doi)
                        continue

                    append_jsonl(
                        index_path,
                        {
                            "review_doi": review_doi,
                            "review_title": first(item.get("title")),
                            "review_url": item.get("URL", ""),
                            "article_dois": article_dois,
                            "crossref": item,
                        },
                    )
                    seen_index.add(review_doi)

                    for article_doi in article_dois:
                        processed += 1
                        if review_doi in seen_review_dois:
                            continue

                        if article_doi not in epmc_cache:
                            try:
                                epmc_cache[article_doi] = europe_pmc_lookup(article_doi)
                                time.sleep(args.sleep)
                            except Exception as exc:  # noqa: BLE001
                                append_jsonl(
                                    skipped_path,
                                    {
                                        "review_doi": review_doi,
                                        "article_doi": article_doi,
                                        "reason": "europe_pmc_lookup_error",
                                        "error_type": type(exc).__name__,
                                        "error": str(exc)[:500],
                                    },
                                )
                                epmc_cache[article_doi] = None

                        epmc = epmc_cache[article_doi]
                        if epmc is None:
                            append_jsonl(
                                skipped_path,
                                {
                                    "review_doi": review_doi,
                                    "article_doi": article_doi,
                                    "reason": "not_found_in_europe_pmc",
                                },
                            )
                            continue

                        if args.require_social_keyword and not SOCIAL_DEMOGRAPHIC_TERMS.search(epmc_text(epmc)):
                            append_jsonl(
                                skipped_path,
                                {
                                    "review_doi": review_doi,
                                    "article_doi": article_doi,
                                    "reason": "no_social_demographic_keyword",
                                    "article_title": epmc.get("title", ""),
                                },
                            )
                            continue

                        try:
                            record = collect_for_review_record(
                                crossref_item=item,
                                article_doi=article_doi,
                                epmc=epmc,
                                out_dir=args.out_dir,
                                sleep=args.sleep,
                                download_artifacts=not args.metadata_only,
                            )
                            append_jsonl(records_path, record)
                            kept += 1
                            seen_review_dois.add(review_doi)

                            f1000 = (record.get("structured") or {}).get("f1000")
                            if f1000 and f1000.get("reports"):
                                append_jsonl(
                                    f1000_path,
                                    {
                                        "article_doi": article_doi,
                                        "article_title": epmc.get("title", ""),
                                        "article_url": record.get("downloads", {}).get("article", {}).get("final_url", ""),
                                        "europe_pmc": epmc,
                                        **f1000,
                                    },
                                )
                        except Exception as exc:  # noqa: BLE001
                            append_jsonl(
                                skipped_path,
                                {
                                    "review_doi": review_doi,
                                    "article_doi": article_doi,
                                    "reason": "record_collection_error",
                                    "error_type": type(exc).__name__,
                                    "error": str(exc)[:500],
                                },
                            )

                        if processed % 100 == 0:
                            log(f"processed reviewed DOI links={processed}; kept health records={kept}")
            except Exception as exc:  # noqa: BLE001
                append_jsonl(
                    skipped_path,
                    {
                        "prefix": prefix,
                        "query": query,
                        "reason": "crossref_iterator_error",
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                    },
                )

    log("=== done ===")
    log(f"records: {records_path}")
    log(f"f1000 structured: {f1000_path}")
    log(f"skipped: {skipped_path}")
    log(f"processed reviewed DOI links={processed}; kept health records={kept}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
