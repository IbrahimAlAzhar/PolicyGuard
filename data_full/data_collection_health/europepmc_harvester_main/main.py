#!/usr/bin/env python3
"""
Europe PMC harvester -- full-text XML (metadata + embedded peer reviews).

WHY THIS IS THE WORKHORSE
  Europe PMC hosts the Open-Access full text of a huge slice of biomedical /
  public-health literature as JATS XML. For journals that practise OPEN peer
  review (notably the BMC series, and others), the reviewer reports + author
  responses are embedded in that XML as <sub-article> elements, e.g.
      <sub-article article-type="reviewer-report"> ... </sub-article>
      <sub-article article-type="author-comment"> ... </sub-article>
  So one full-text XML download per article = article metadata + all reviews.

  Configure the --query to target a publisher/journal. Examples:
      JOURNAL:"BMC Medicine"                (one BMC journal)
      (PUBLISHER:"BMC" OR ISSN:"...")       (broaden)
      JOURNAL:"eLife"                       (eLife classic articles)
      JOURNAL:"PLoS ONE"                    (PLOS)
  Always AND with IN_EPMC:y AND OPEN_ACCESS:y so full text is retrievable.

ENDPOINTS (verified working)
  search    : https://www.ebi.ac.uk/europepmc/webservices/rest/search
              ?query=...&format=json&pageSize=1000&cursorMark=*&resultType=core
              -> hitCount, nextCursorMark, resultList.result[] (pmcid, doi, ...)
  full text : https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/fullTextXML
              e.g. .../PMC13292518/fullTextXML

NOTE: Not every OA article carries embedded reviews -- only open-peer-review
      journals do. The harvester saves the full XML regardless; you filter for
      files that actually contain reviewer-report sub-articles when parsing.
"""

import argparse, json, os, re, sys, time, urllib.parse, urllib.request
from urllib.error import HTTPError, URLError

SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FTXML = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
UA = "epmc-harvester/1.0 (research; contact: you@example.com)"

MIN_INTERVAL = 0.4          # Europe PMC is generous; ~2.5 req/s is safe
_last = [0.0]


def _throttle():
    dt = time.time() - _last[0]
    if dt < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - dt)
    _last[0] = time.time()


def fetch(url, retries=5, backoff=5.0):
    for a in range(retries):
        _throttle()
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(backoff * (a + 1)); continue
            raise
        except URLError as e:
            print(f"  net error {e} -> wait", file=sys.stderr)
            time.sleep(backoff * (a + 1))
    raise RuntimeError(f"failed: {url}")


def search_all(query, want_reviews_only=False):
    """Page through every hit via cursorMark. Returns list of result dicts."""
    q = query
    if want_reviews_only:
        # Only articles Europe PMC flags as having linked evaluations/reviews.
        q = f"({query}) AND (HAS_REVIEWS:y)"
    results, cursor, page = [], "*", 0
    while True:
        url = (f"{SEARCH}?query={urllib.parse.quote(q)}"
               f"&format=json&pageSize=1000&resultType=core"
               f"&cursorMark={urllib.parse.quote(cursor)}")
        data = json.loads(fetch(url).decode("utf-8", "replace"))
        hits = data.get("resultList", {}).get("result", [])
        results.extend(hits)
        page += 1
        print(f"  page {page}: +{len(hits)} (total {len(results)} / "
              f"{data.get('hitCount')})")
        nxt = data.get("nextCursorMark")
        if not hits or not nxt or nxt == cursor:
            break
        cursor = nxt
    return results


def harvest(query, outdir, reviews_only):
    os.makedirs(outdir, exist_ok=True)
    print(f"Searching Europe PMC: {query!r}")
    results = search_all(query, reviews_only)
    print(f"Found {len(results):,} records. Saving metadata + full-text XML...")

    # 1) dump the metadata for every hit (one JSON line each)
    meta_path = os.path.join(outdir, "metadata.jsonl")
    with open(meta_path, "w") as mh:
        for r in results:
            mh.write(json.dumps(r) + "\n")

    # 2) download full-text XML for those that are in Europe PMC OA
    xml_dir = os.path.join(outdir, "fulltext_xml")
    os.makedirs(xml_dir, exist_ok=True)
    manifest = os.path.join(outdir, "_manifest.jsonl")
    done = set()
    if os.path.exists(manifest):
        for ln in open(manifest):
            try: done.add(json.loads(ln)["pmcid"])
            except Exception: pass

    with open(manifest, "a") as mf:
        n = 0
        for r in results:
            pmcid = r.get("pmcid")
            if not pmcid or r.get("inEPMC") != "Y" or pmcid in done:
                continue
            n += 1
            path = os.path.join(xml_dir, f"{pmcid}.xml")
            try:
                xml = fetch(FTXML.format(pmcid=pmcid))
                with open(path, "wb") as fh:
                    fh.write(xml)
                has_rev = b"reviewer-report" in xml or b'"ref-report"' in xml \
                    or b"author-comment" in xml
                mf.write(json.dumps({"pmcid": pmcid, "doi": r.get("doi"),
                                     "bytes": len(xml),
                                     "has_review_subarticle": bool(has_rev)}) + "\n")
                mf.flush()
                if n % 25 == 0:
                    print(f"  [{n}] {pmcid} ({len(xml):,} B, "
                          f"reviews={'Y' if has_rev else 'n'})")
            except Exception as e:
                print(f"  FAILED {pmcid}: {e}", file=sys.stderr)
    print(f"\nDone. Metadata -> {meta_path}; full text -> {xml_dir}/")
    print("Parse the XML for <sub-article article-type='reviewer-report'> etc.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True,
                    help='Europe PMC query, e.g. JOURNAL:"BMC Medicine"')
    ap.add_argument("--outdir", default="epmc_data")
    ap.add_argument("--open-access", action="store_true", default=True)
    ap.add_argument("--reviews-only", action="store_true",
                    help="restrict to records Europe PMC flags as having reviews")
    args = ap.parse_args()
    q = args.query
    if args.open_access and "OPEN_ACCESS" not in q and "IN_EPMC" not in q:
        q = f"({q}) AND IN_EPMC:y AND OPEN_ACCESS:y"
    harvest(q, args.outdir, args.reviews_only)