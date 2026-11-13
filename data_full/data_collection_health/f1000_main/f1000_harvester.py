#!/usr/bin/env python3
"""
F1000Research metadata + peer-review harvester.

Key fact about F1000Research:
  Each article's full-text JATS XML ALREADY CONTAINS the peer review reports,
  embedded as <sub-article> elements (article-type is usually "ref-report").
  So downloading the article XML gives you: article metadata + ALL peer review
  reports in a single file. You parse the text later.

How the whole corpus is enumerated (confirmed from the live site):
  https://f1000research.com/published-xml-urls  returns an HTML page whose table
  lists a direct XML URL for every published article *version*, e.g.
      https://f1000research.com/articles/3-28/v1/xml
      https://f1000research.com/articles/7-1801/v2/xml
  We scrape those URLs and download each one. (Note: multiple versions of the
  same article appear as separate rows, e.g. .../13-1318/v2/xml and
  .../13-1318/v4/xml -- keep them all; each version has its own reviews.)

API docs: https://f1000research.com/developers
  - Rate limit : max 100 requests / 60 seconds (else HTTP 401)
  - Solr search (optional filtered route): extapi/search?q=...

This script:
  1. Builds the list of article XML URLs (whole corpus, or a Solr query).
  2. Downloads each article's raw XML (metadata + peer reviews) to disk.
  3. Is resumable, rate-limited, and retries transient failures.

Usage:
  python f1000_harvester.py                          # whole corpus
  python f1000_harvester.py --query 'R_ABS:"malaria"'  # filtered subset
  python f1000_harvester.py --limit 20               # quick test
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError

BASE = "https://f1000research.com"
CORPUS_URLS = f"{BASE}/published-xml-urls"
SEARCH_EP = f"{BASE}/extapi/search"
XML_EP = f"{BASE}/extapi/article/xml"      # used for the Solr/DOI route

USER_AGENT = "f1000-harvester/1.1 (research; contact: you@example.com)"

# Any F1000Research DOI looks like 10.12688/f1000research.<id>.<version>
DOI_RE = re.compile(r"10\.12688/f1000research\.\d+(?:\.\d+)?")
# Article XML URLs on the corpus page look like /articles/3-28/v1/xml
ARTICLE_XML_RE = re.compile(
    r"https?://f1000research\.com/articles/[0-9]+-[0-9]+/v[0-9]+/xml")
PAGES_RE = re.compile(r'totalNumberOfPages"?\s*[:>]\s*"?(\d+)')

# --- Rate limiting -----------------------------------------------------------
# Docs allow 100 requests / 60s. We stay under it: ~1 req / 0.8s.
MIN_INTERVAL = 0.8
_last_call = [0.0]


def _throttle():
    dt = time.time() - _last_call[0]
    if dt < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - dt)
    _last_call[0] = time.time()


def fetch(url, retries=5, backoff=5.0):
    """GET a URL as bytes, honoring rate limits and retrying on 401/429/5xx."""
    for attempt in range(retries):
        _throttle()
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read()
        except HTTPError as e:
            if e.code in (401, 429, 500, 502, 503, 504):
                wait = backoff * (attempt + 1)
                print(f"  HTTP {e.code} -> waiting {wait:.0f}s "
                      f"(attempt {attempt+1}/{retries})", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
        except URLError as e:
            wait = backoff * (attempt + 1)
            print(f"  network error {e} -> waiting {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} retries: {url}")


# --- Discovery ---------------------------------------------------------------
def url_to_id(xml_url):
    """
    https://f1000research.com/articles/3-28/v1/xml  ->  '3-28_v1'
    Used as the output filename stem (unique per article version).
    """
    m = re.search(r"/articles/([0-9]+-[0-9]+)/(v[0-9]+)/xml", xml_url)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return re.sub(r"[^A-Za-z0-9_-]", "_", xml_url)[-60:]


def get_all_xml_urls_from_corpus():
    """
    Whole-corpus route. Scrape every article XML URL out of the corpus HTML page.
    Returns a de-duplicated, order-preserving list of full XML URLs.
    """
    raw = fetch(CORPUS_URLS).decode("utf-8", "replace")
    print(f"  [corpus] downloaded {len(raw):,} chars from {CORPUS_URLS}")
    urls, seen = [], set()
    for u in ARTICLE_XML_RE.findall(raw):
        if u not in seen:
            seen.add(u)
            urls.append(u)
    print(f"  [corpus] found {len(urls):,} article-version XML URLs")
    return urls


def get_xml_urls_from_search(query):
    """
    Filtered route via Solr search. Format-agnostic: scrape F1000 DOIs straight
    from the raw response (JSON or XML) and turn each into an extapi XML URL.
    Pages 100 results at a time; never crashes on a non-JSON body.
    """
    dois, seen = [], set()
    page, total_pages = 1, 1
    while page <= total_pages:
        url = (f"{SEARCH_EP}?q={urllib.parse.quote(query)}"
               f"&wt=json&rows=100&page={page}")
        raw = fetch(url).decode("utf-8", "replace")
        if page == 1:
            print(f"  [search] q={query!r} -> {len(raw):,} chars. First 300:")
            print("  " + raw[:300].replace("\n", "\n  "))
        m = PAGES_RE.search(raw)
        if m:
            total_pages = int(m.group(1)) or 1
        before = len(dois)
        for d in DOI_RE.findall(raw):
            if d not in seen:
                seen.add(d)
                dois.append(d)
        print(f"  search page {page}/{total_pages}: {len(dois)} DOIs so far")
        if len(dois) == before:
            if before == 0:
                print("  [search] no DOIs on first page -- see snippet above.")
            break
        page += 1
    # Convert DOIs to extapi XML URLs.
    return [f"{XML_EP}?doi={urllib.parse.quote(d)}" for d in dois]


# --- Downloading -------------------------------------------------------------
def download(xml_urls, outdir):
    os.makedirs(outdir, exist_ok=True)
    manifest_path = os.path.join(outdir, "_manifest.jsonl")

    done = set()
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            for ln in fh:
                try:
                    done.add(json.loads(ln)["url"])
                except Exception:
                    pass
    if done:
        print(f"  resuming: {len(done):,} already downloaded, will skip those")

    with open(manifest_path, "a") as manifest:
        for i, url in enumerate(xml_urls, 1):
            if url in done:
                continue
            if "doi=" in url:
                stem = re.sub(r"[^A-Za-z0-9._-]", "_",
                              urllib.parse.parse_qs(
                                  urllib.parse.urlparse(url).query)
                              .get("doi", ["item"])[0])
            else:
                stem = url_to_id(url)
            xml_path = os.path.join(outdir, f"{stem}.xml")
            try:
                data = fetch(url)
                with open(xml_path, "wb") as fh:
                    fh.write(data)
                manifest.write(json.dumps(
                    {"url": url, "file": os.path.basename(xml_path),
                     "bytes": len(data)}) + "\n")
                manifest.flush()
                if i % 25 == 0 or i == len(xml_urls):
                    print(f"[{i}/{len(xml_urls)}] saved {stem} "
                          f"({len(data):,} bytes)")
            except Exception as e:
                print(f"[{i}/{len(xml_urls)}] FAILED {url}: {e}", file=sys.stderr)


# --- Optional: quick review extractor (adapt as needed) ----------------------
def extract_reviews(xml_bytes):
    """Pull peer-review report sub-articles out of one F1000Research JATS XML."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_bytes)

    def txt(el):
        return " ".join(el.itertext()).strip() if el is not None else ""

    reviews = []
    for sub in root.iter("sub-article"):
        atype = sub.get("article-type", "")
        if "report" not in atype and "review" not in atype:
            continue
        names = []
        for contrib in sub.iter("contrib"):
            sn = contrib.find(".//surname")
            gn = contrib.find(".//given-names")
            nm = " ".join(x.text for x in (gn, sn) if x is not None and x.text)
            if nm:
                names.append(nm)
        reviews.append({
            "article_type": atype,
            "reviewers": names,
            "report_text": txt(sub.find(".//body")),
        })
    return reviews


# --- CLI ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Harvest F1000Research XML + reviews")
    ap.add_argument("--outdir", default="f1000_xml")
    ap.add_argument("--query", default=None,
                    help='Solr query, e.g. R_ABS:"malaria". Omit for whole corpus.')
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N articles (0 = no limit; for testing)")
    args = ap.parse_args()

    print("Discovering article XML URLs...")
    if args.query:
        urls = get_xml_urls_from_search(args.query)
    else:
        urls = get_all_xml_urls_from_corpus()
    print(f"Found {len(urls):,} article versions.")

    if args.limit:
        urls = urls[:args.limit]
        print(f"Limiting to first {len(urls)} for this run.")

    download(urls, args.outdir)
    print(f"\nDone. Raw XML in {args.outdir}/  "
          f"(each file = article metadata + all peer review reports).")


if __name__ == "__main__":
    main()