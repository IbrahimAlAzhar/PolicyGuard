"""Detect and optionally download open peer-review / review-history material.

Input: JSONL records from Europe PMC/Crossref/etc.
Output: JSONL records with candidate review-package URLs.

Example:
python -m collectors.review_package_detector \
  --infile data/metadata/europepmc_health_equity.jsonl \
  --out data/metadata/review_package_candidates.jsonl \
  --download
"""
import argparse
import os
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from tqdm import tqdm

from utils.io import append_jsonl, download_file, get_session, load_config, read_jsonl, safe_get, slugify

REVIEW_KEYWORDS = [
    "peer review",
    "peer-review",
    "peerreview",
    "review history",
    "reviewer comments",
    "reviewer reports",
    "review reports",
    "pre-publication history",
    "prepublication history",
    "author response",
    "decision letter",
    "review process file",
    "transparent peer review",
    "referee report",
    "refereed preprint",
]


def unique(seq):
    seen = set()
    out = []
    for x in seq:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def candidate_article_urls(rec: Dict) -> List[str]:
    urls = []
    for key in ["url", "URL", "article_url", "final_paper_url"]:
        if rec.get(key):
            urls.append(rec[key])
    if rec.get("doi"):
        urls.append("https://doi.org/" + rec["doi"])
    if rec.get("DOI"):
        urls.append("https://doi.org/" + rec["DOI"])

    # Europe PMC full-text URL list
    for u in rec.get("fullTextUrlList", []) or []:
        if isinstance(u, dict) and u.get("url"):
            urls.append(u["url"])

    # Crossref links
    for link in rec.get("links", []) or []:
        if isinstance(link, dict) and link.get("URL"):
            urls.append(link["URL"])

    # bioRxiv/medRxiv PDF or DOI page
    if rec.get("pdf_url"):
        urls.append(rec["pdf_url"])

    return unique(urls)


def bmj_variants(url: str) -> List[str]:
    out = []
    if "bmj" not in url.lower():
        return out
    clean = url.split("?")[0].rstrip("/")
    # BMJ article pages often support these suffixes for accepted articles.
    if not clean.endswith(".reviewer-comments"):
        out.append(clean + ".reviewer-comments")
    if not clean.endswith(".draft-revisions.pdf"):
        out.append(clean + ".draft-revisions.pdf")
    return out


def html_review_links(base_url: str, html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "lxml")
    found = []
    for a in soup.find_all("a"):
        text = " ".join(a.get_text(" ", strip=True).split())
        href = a.get("href")
        if not href:
            continue
        blob = f"{text} {href}".lower()
        if any(k in blob for k in REVIEW_KEYWORDS):
            found.append({
                "url": urljoin(base_url, href),
                "anchor_text": text,
                "detector": "html_anchor_keyword",
            })
    return found


def probe_url(session, url: str, sleep: float) -> Optional[Dict]:
    r = safe_get(session, url, sleep=sleep, timeout=40)
    if not r:
        return None
    ctype = r.headers.get("content-type", "").lower()
    return {
        "url": r.url,
        "status_code": r.status_code,
        "content_type": ctype,
        "text": r.text if "text/html" in ctype else None,
    }


def detect_for_record(session, rec: Dict, sleep: float) -> Dict:
    source_urls = candidate_article_urls(rec)
    review_links = []
    visited = set()

    for url in source_urls:
        if not url or url in visited:
            continue
        visited.add(url)

        # Probe direct BMJ variants without needing HTML parse.
        for v in bmj_variants(url):
            if v in visited:
                continue
            visited.add(v)
            pr = probe_url(session, v, sleep)
            if pr:
                review_links.append({
                    "url": pr["url"],
                    "anchor_text": "BMJ inferred review-history variant",
                    "detector": "bmj_url_variant",
                    "content_type": pr["content_type"],
                })

        pr = probe_url(session, url, sleep)
        if not pr:
            continue
        if pr["text"]:
            review_links.extend(html_review_links(pr["url"], pr["text"]))

    # de-duplicate by URL
    deduped = []
    seen = set()
    for link in review_links:
        u = link.get("url")
        if u and u not in seen:
            deduped.append(link)
            seen.add(u)

    return {
        "source_record": {
            "source": rec.get("source"),
            "doi": rec.get("doi") or rec.get("DOI"),
            "title": rec.get("title"),
            "journal": rec.get("journalTitle") or rec.get("container_title"),
            "article_urls_checked": source_urls,
        },
        "review_links": deduped,
        "n_review_links": len(deduped),
    }


def download_review_links(session, detected: Dict, review_dir: str, sleep: float) -> Dict:
    doi = detected["source_record"].get("doi") or detected["source_record"].get("title") or "record"
    downloaded = []
    for i, link in enumerate(detected.get("review_links", []), start=1):
        url = link.get("url")
        if not url:
            continue
        ext = ".pdf" if ".pdf" in url.lower() else ".html"
        fname = slugify(f"{doi}_review_{i}") + ext
        out_path = os.path.join(review_dir, fname)
        ok = download_file(session, url, out_path, sleep=sleep)
        item = dict(link)
        item["local_path"] = out_path if ok else None
        item["download_ok"] = ok
        downloaded.append(item)
    detected["downloaded_review_links"] = downloaded
    return detected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--review-dir", default="data/reviews")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    user_agent = cfg.get("user_agent")
    sleep = float(cfg.get("request_sleep_seconds", 1.0))
    session = get_session(user_agent)
    os.makedirs(args.review_dir, exist_ok=True)

    records = list(read_jsonl(args.infile))
    open(args.out, "w", encoding="utf-8").close()

    kept = 0
    for rec in tqdm(records, desc="Detect review packages"):
        detected = detect_for_record(session, rec, sleep)
        if args.download and detected["n_review_links"] > 0:
            detected = download_review_links(session, detected, args.review_dir, sleep)
        append_jsonl(args.out, detected)
        if detected["n_review_links"] > 0:
            kept += 1

    print(f"Saved {len(records)} detection records to {args.out}")
    print(f"Records with at least one review-package candidate: {kept}")


if __name__ == "__main__":
    main()
