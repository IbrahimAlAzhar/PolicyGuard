#!/usr/bin/env python3
"""
bioRxiv / medRxiv harvester -- preprint metadata + public preprint reviews.

IMPORTANT: bioRxiv and medRxiv are PREPRINT servers. They do NOT run formal
peer review themselves, so there are no reviewer reports "on" the preprint.
Public reviews of these preprints are produced by third-party services
(Review Commons, eLife, PREreview, Biophysics Colab, etc.) and AGGREGATED by
Sciety. Sciety exposes them as machine-readable DocMaps that link each review's
text to the preprint DOI. So this script has two parts:

  PART A -- Preprint metadata via the bioRxiv API (medRxiv = the medical server):
      https://api.biorxiv.org/details/{server}/{start}/{end}/{cursor}/json
      server = 'biorxiv' or 'medrxiv'; dates = YYYY-MM-DD; cursor steps by 100.

  PART B -- Reviews via Sciety DocMaps:
      Index    : https://sciety.org/docmaps/v1/index
      Each item points to a docmap URL whose JSON contains the preprint DOI and
      the evaluation (review) content URLs. We save every docmap as raw JSON;
      you later join docmap.preprint_doi -> your preprint metadata and pull the
      review text from the content URLs inside each docmap.

Everything is saved raw so you can parse text later.
"""

import argparse, datetime, json, os, sys, time, urllib.request
from urllib.error import HTTPError, URLError

BIORXIV = "https://api.biorxiv.org/details/{server}/{start}/{end}/{cursor}/json"
SCIETY_INDEX = "https://sciety.org/docmaps/v1/index"
UA = "biorxiv-harvester/1.0 (research; contact: you@example.com)"

MIN_INTERVAL = 0.6
_last = [0.0]


def _throttle():
    dt = time.time() - _last[0]
    if dt < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - dt)
    _last[0] = time.time()


def fetch(url, retries=5, backoff=5.0, allow_404=False):
    for a in range(retries):
        _throttle()
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except HTTPError as e:
            if e.code == 404 and allow_404:
                return None
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(backoff * (a + 1)); continue
            raise
        except URLError as e:
            print(f"  net error {e} -> wait", file=sys.stderr)
            time.sleep(backoff * (a + 1))
    return None


# ---------- PART A: preprint metadata ----------
def harvest_metadata(server, start, end, outdir):
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{server}_metadata.jsonl")
    cursor, total, saved = 0, None, 0
    with open(out, "w") as fh:
        while True:
            url = BIORXIV.format(server=server, start=start, end=end, cursor=cursor)
            raw = fetch(url)
            if not raw:
                break
            data = json.loads(raw.decode("utf-8", "replace"))
            msg = (data.get("messages") or [{}])[0]
            total = int(msg.get("total", 0) or 0)
            coll = data.get("collection", [])
            for rec in coll:
                fh.write(json.dumps(rec) + "\n")
            saved += len(coll)
            print(f"  {server}: cursor {cursor}, +{len(coll)} (saved {saved}/{total})")
            if not coll or saved >= total:
                break
            cursor += len(coll)
    print(f"  {server} metadata -> {out} ({saved} preprint versions)")


# ---------- PART B: Sciety reviews (DocMaps) ----------
def harvest_reviews(outdir, updated_after=None):
    os.makedirs(outdir, exist_ok=True)
    idx_url = SCIETY_INDEX + (f"?updatedAfter={updated_after}" if updated_after else "")
    print(f"Fetching Sciety docmap index: {idx_url}")
    raw = fetch(idx_url)
    if not raw:
        print("  could not fetch Sciety index", file=sys.stderr); return
    index = json.loads(raw.decode("utf-8", "replace"))
    with open(os.path.join(outdir, "sciety_index.json"), "wb") as fh:
        fh.write(raw)

    # The index is a list (or has 'articles'): each item exposes a docmap URL.
    items = index.get("articles", index) if isinstance(index, dict) else index
    docmap_urls = []
    for it in items:
        u = it.get("docmap") if isinstance(it, dict) else None
        if u:
            docmap_urls.append(u)
    print(f"  index lists {len(docmap_urls):,} docmaps")

    dm_dir = os.path.join(outdir, "docmaps")
    os.makedirs(dm_dir, exist_ok=True)
    manifest = os.path.join(outdir, "_docmaps_manifest.jsonl")
    done = set()
    if os.path.exists(manifest):
        for ln in open(manifest):
            try: done.add(json.loads(ln)["url"])
            except Exception: pass

    with open(manifest, "a") as mf:
        for i, u in enumerate(docmap_urls, 1):
            if u in done:
                continue
            raw = fetch(u, allow_404=True)
            if not raw:
                continue
            name = f"docmap_{i:07d}.json"
            with open(os.path.join(dm_dir, name), "wb") as fh:
                fh.write(raw)
            mf.write(json.dumps({"url": u, "file": name}) + "\n"); mf.flush()
            if i % 50 == 0:
                print(f"  docmap [{i}/{len(docmap_urls)}]")
    print(f"  reviews (docmaps) -> {dm_dir}/  (each has preprint DOI + review URLs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", choices=["biorxiv", "medrxiv", "both"],
                    default="medrxiv")
    ap.add_argument("--start", default="2019-01-01", help="YYYY-MM-DD")
    ap.add_argument("--end", default=datetime.date.today().isoformat())
    ap.add_argument("--outdir", default="biorxiv_data")
    ap.add_argument("--reviews", action="store_true",
                    help="also harvest Sciety docmap reviews")
    ap.add_argument("--updated-after", default=None,
                    help="only Sciety docmaps updated after YYYY-MM-DD")
    args = ap.parse_args()

    servers = ["biorxiv", "medrxiv"] if args.server == "both" else [args.server]
    for s in servers:
        print(f"=== {s} metadata {args.start}..{args.end} ===")
        harvest_metadata(s, args.start, args.end, args.outdir)
    if args.reviews:
        print("=== Sciety reviews (DocMaps) ===")
        harvest_reviews(os.path.join(args.outdir, "reviews"), args.updated_after)
    print("Done.")