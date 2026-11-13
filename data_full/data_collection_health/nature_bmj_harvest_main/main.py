#!/usr/bin/env python3
"""
Nature / BMJ transparent-peer-review harvester  (BEST-EFFORT).

READ THIS FIRST -- feasibility is lower than the other sources:
  * Neither Nature nor BMJ offers an API for peer-review TEXT.
  * Nature Communications (and a growing set of Nature journals) publish a
    "Peer Review File" PDF alongside ~70% of papers (author opt-in). BMJ /
    BMJ Open publish a "pre-publication history" with reviewer reports as PDF.
  * So the only route to the full review text is scraping each article page for
    the peer-review PDF link. Pages are JS-heavy and publishers rate-limit /
    restrict bulk access, so expect partial success. Respect each publisher's
    text-and-data-mining policy and robots.txt.

This script therefore does two things:

  PART 1 (reliable) -- Peer-review + article METADATA via Crossref:
      Enumerate every DOI for a journal by ISSN, and separately pull any
      Crossref 'peer-review' records the publisher deposited (these carry the
      reviewed-article relation but usually NOT the full text).
        https://api.crossref.org/journals/{ISSN}/works
        https://api.crossref.org/works?filter=type:peer-review,issn:{ISSN}

  PART 2 (best-effort) -- Peer Review File PDFs:
      For each article, fetch the landing page and look for a peer-review PDF
      link (Nature: 'MOESM' media objects / links labelled peer review; BMJ:
      pre-publication history PDF). Save any PDF found. Many will be missing --
      that's expected. For JS-rendered pages that hide the link, use a browser
      automation tool instead of this simple fetch.

ISSNs:
  Nature Communications : 2041-1723
  Nature (main)         : 1476-4687
  The BMJ               : 1756-1833   (research articles carry review history)
  BMJ Open              : 2044-6055
"""

import argparse, json, os, re, sys, time, urllib.parse, urllib.request
from urllib.error import HTTPError, URLError

CR_WORKS = "https://api.crossref.org/journals/{issn}/works"
CR_REVIEWS = "https://api.crossref.org/works"
UA = "nature-bmj-harvester/1.0 (research; mailto:you@example.com)"

MIN_INTERVAL = 1.0     # be polite to Crossref + publishers
_last = [0.0]


def _throttle():
    dt = time.time() - _last[0]
    if dt < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - dt)
    _last[0] = time.time()


def fetch(url, retries=4, backoff=6.0, allow_fail=False):
    for a in range(retries):
        _throttle()
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read()
        except HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(backoff * (a + 1)); continue
            if allow_fail:
                return None
            raise
        except URLError as e:
            if allow_fail:
                return None
            time.sleep(backoff * (a + 1))
    return None


# ---------- PART 1: Crossref metadata ----------
def crossref_works(issn, outdir):
    """Every work (article) DOI + metadata for the journal, via cursor paging."""
    out = os.path.join(outdir, f"crossref_works_{issn}.jsonl")
    cursor, n, total = "*", 0, None
    with open(out, "w") as fh:
        while True:
            url = (f"{CR_WORKS.format(issn=issn)}?rows=1000"
                   f"&cursor={urllib.parse.quote(cursor)}")
            raw = fetch(url, allow_fail=True)
            if not raw:
                break
            msg = json.loads(raw.decode("utf-8", "replace")).get("message", {})
            total = msg.get("total-results")
            items = msg.get("items", [])
            for it in items:
                fh.write(json.dumps(it) + "\n")
            n += len(items)
            print(f"  works {issn}: +{len(items)} (total {n}/{total})")
            cursor = msg.get("next-cursor")
            if not items or not cursor:
                break
    print(f"  -> {out} ({n} works)")
    return out


def crossref_peer_reviews(issn, outdir):
    """Crossref 'peer-review' records deposited for this journal, if any."""
    out = os.path.join(outdir, f"crossref_peerreviews_{issn}.jsonl")
    cursor, n = "*", 0
    with open(out, "w") as fh:
        while True:
            url = (f"{CR_REVIEWS}?filter=type:peer-review,issn:{issn}"
                   f"&rows=1000&cursor={urllib.parse.quote(cursor)}")
            raw = fetch(url, allow_fail=True)
            if not raw:
                break
            msg = json.loads(raw.decode("utf-8", "replace")).get("message", {})
            items = msg.get("items", [])
            for it in items:
                fh.write(json.dumps(it) + "\n")
            n += len(items)
            print(f"  peer-review recs {issn}: +{len(items)} (total {n})")
            cursor = msg.get("next-cursor")
            if not items or not cursor:
                break
    print(f"  -> {out} ({n} peer-review records)")


# ---------- PART 2: best-effort Peer Review File PDFs ----------
PDF_LINK_RE = re.compile(
    r'href="([^"]+?(?:MOESM\d+_ESM|peer[_-]?review[^"]*|prepub[^"]*)\.pdf[^"]*)"',
    re.I)


def scrape_review_pdfs(works_jsonl, outdir, limit=0):
    pdf_dir = os.path.join(outdir, "peer_review_pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    manifest = os.path.join(outdir, "_pdf_manifest.jsonl")
    done = set()
    if os.path.exists(manifest):
        for ln in open(manifest):
            try: done.add(json.loads(ln)["doi"])
            except Exception: pass

    n = 0
    with open(manifest, "a") as mf:
        for ln in open(works_jsonl):
            it = json.loads(ln)
            doi = it.get("DOI")
            urls = it.get("resource", {}).get("primary", {}).get("URL") or ""
            if not doi or doi in done or not urls:
                continue
            n += 1
            if limit and n > limit:
                break
            html = fetch(urls, allow_fail=True)
            found = []
            if html:
                text = html.decode("utf-8", "replace")
                for m in PDF_LINK_RE.findall(text):
                    pdf_url = m if m.startswith("http") else \
                        urllib.parse.urljoin(urls, m)
                    pdf = fetch(pdf_url, allow_fail=True)
                    if pdf and pdf[:4] == b"%PDF":
                        name = re.sub(r"[^A-Za-z0-9]", "_", doi) + \
                            f"_{len(found)}.pdf"
                        with open(os.path.join(pdf_dir, name), "wb") as fh:
                            fh.write(pdf)
                        found.append(name)
            mf.write(json.dumps({"doi": doi, "landing": urls,
                                 "pdfs": found}) + "\n"); mf.flush()
            if n % 20 == 0:
                print(f"  [{n}] {doi}: {len(found)} pdf(s)")
    print(f"  peer-review PDFs -> {pdf_dir}/  (best-effort; many will be absent)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--issn", required=True, help="journal ISSN, e.g. 2041-1723")
    ap.add_argument("--outdir", default="nature_bmj_data")
    ap.add_argument("--scrape-pdfs", action="store_true",
                    help="attempt to download peer-review PDFs (fragile)")
    ap.add_argument("--limit", type=int, default=0, help="cap PDF scrape (test)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"=== Crossref metadata for ISSN {args.issn} ===")
    works = crossref_works(args.issn, args.outdir)
    crossref_peer_reviews(args.issn, args.outdir)

    if args.scrape_pdfs:
        print("=== best-effort Peer Review File PDFs ===")
        scrape_review_pdfs(works, args.outdir, args.limit)
    print("Done.")