#!/usr/bin/env python3
"""
eLife harvester -- native eLife API (public reviews + eLife assessments).

eLife publishes ALL peer reviews openly under its "Reviewed Preprint" model:
each paper carries public reviewer reports + an "eLife assessment" + author
response. The eLife API serves these as JSON. This script:

  1. Pages the reviewed-preprints list to enumerate every paper (id, doi, title).
  2. For each id, saves the raw JSON of:
       - the detail record                /reviewed-preprints/{id}
       - the reviews/assessment content   /reviewed-preprints/{id}/reviews
     (whichever the API returns; both are saved when available).
  Also pages the classic /articles list so you can additionally pull decision
  letters + author responses for pre-2023 Version-of-Record eLife papers.

API base: https://api.elifesciences.org   (docs: https://api.elifesciences.org/)
The API is versioned via vendor Accept headers, e.g.
  Accept: application/vnd.elife.reviewed-preprint-list+json; version=1

If any sub-endpoint 404s (schema changes over time), you still keep the list +
detail JSON; and the Europe PMC harvester (query JOURNAL:"eLife") is a fallback
for the full-text reviews of classic articles.
"""

import argparse, json, os, sys, time, urllib.request
from urllib.error import HTTPError, URLError

BASE = "https://api.elifesciences.org"
UA = "elife-harvester/1.0 (research; contact: you@example.com)"

MIN_INTERVAL = 0.4
_last = [0.0]


def _throttle():
    dt = time.time() - _last[0]
    if dt < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - dt)
    _last[0] = time.time()


def fetch(url, accept=None, retries=5, backoff=5.0, allow_404=False):
    headers = {"User-Agent": UA}
    if accept:
        headers["Accept"] = accept
    for a in range(retries):
        _throttle()
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read()
        except HTTPError as e:
            if e.code == 404 and allow_404:
                return None
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(backoff * (a + 1)); continue
            if allow_404:
                return None
            raise
        except URLError as e:
            print(f"  net error {e} -> wait", file=sys.stderr)
            time.sleep(backoff * (a + 1))
    return None


def list_all(kind):
    """
    kind = 'reviewed-preprints' or 'articles'. Pages the list endpoint.
    Returns list of item dicts (each has 'id', 'doi', 'title', ...).
    """
    accept = f"application/vnd.elife.{('reviewed-preprint-list' if kind=='reviewed-preprints' else 'article-list')}+json; version=1"
    items, page, per = [], 1, 100
    while True:
        url = f"{BASE}/{kind}?page={page}&per-page={per}&order=asc"
        raw = fetch(url, accept=accept, allow_404=True)
        if not raw:
            break
        data = json.loads(raw.decode("utf-8", "replace"))
        batch = data.get("items", [])
        items.extend(batch)
        total = data.get("total")
        print(f"  {kind} page {page}: +{len(batch)} (total {len(items)}/{total})")
        if not batch or (total and len(items) >= total):
            break
        page += 1
    return items


def harvest(outdir, do_articles):
    os.makedirs(outdir, exist_ok=True)

    # ---- Reviewed Preprints (the open-review model) ----
    rp = list_all("reviewed-preprints")
    with open(os.path.join(outdir, "reviewed_preprints_list.jsonl"), "w") as fh:
        for it in rp:
            fh.write(json.dumps(it) + "\n")

    rev_dir = os.path.join(outdir, "reviewed_preprints")
    os.makedirs(rev_dir, exist_ok=True)
    manifest = os.path.join(outdir, "_manifest.jsonl")
    done = set()
    if os.path.exists(manifest):
        for ln in open(manifest):
            try: done.add(json.loads(ln)["id"])
            except Exception: pass

    with open(manifest, "a") as mf:
        for i, it in enumerate(rp, 1):
            _id = str(it.get("id"))
            if not _id or _id in done:
                continue
            rec = {"id": _id, "doi": it.get("doi"), "saved": []}
            for suffix, tag in [("", "detail"), ("/reviews", "reviews")]:
                raw = fetch(f"{BASE}/reviewed-preprints/{_id}{suffix}",
                            allow_404=True)
                if raw:
                    p = os.path.join(rev_dir, f"{_id}_{tag}.json")
                    with open(p, "wb") as fh:
                        fh.write(raw)
                    rec["saved"].append(tag)
            mf.write(json.dumps(rec) + "\n"); mf.flush()
            if i % 25 == 0:
                print(f"  [{i}/{len(rp)}] {_id} saved {rec['saved']}")

    # ---- Classic Version-of-Record articles (decision letters etc.) ----
    if do_articles:
        arts = list_all("articles")
        with open(os.path.join(outdir, "articles_list.jsonl"), "w") as fh:
            for it in arts:
                fh.write(json.dumps(it) + "\n")
        art_dir = os.path.join(outdir, "articles")
        os.makedirs(art_dir, exist_ok=True)
        for i, it in enumerate(arts, 1):
            _id = str(it.get("id"))
            raw = fetch(f"{BASE}/articles/{_id}", allow_404=True)
            if raw:
                with open(os.path.join(art_dir, f"{_id}.json"), "wb") as fh:
                    fh.write(raw)
            if i % 50 == 0:
                print(f"  article [{i}/{len(arts)}] {_id}")

    print(f"\nDone. Raw eLife JSON in {outdir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="elife_data")
    ap.add_argument("--articles", action="store_true",
                    help="also harvest classic VoR article JSON (large)")
    args = ap.parse_args()
    harvest(args.outdir, args.articles)