"""Collect bioRxiv/medRxiv metadata using the official API.

The details endpoint returns preprint metadata.
The pubs endpoint returns preprint-to-published-article links when known.

Example:
python -m collectors.biorxiv_medrxiv_collect \
  --server medrxiv --start-date 2024-01-01 --end-date 2024-12-31 \
  --out data/metadata/medrxiv_2024.jsonl
"""
import argparse
from typing import Dict, Iterable

from tqdm import tqdm

from utils.io import append_jsonl, get_session, load_config, safe_get

BASE = "https://api.biorxiv.org"


def normalize_preprint(item: Dict, server: str) -> Dict:
    doi = item.get("doi")
    pdf_url = None
    if doi:
        # This URL pattern works for biorxiv/medrxiv article PDFs in many cases.
        host = "www.medrxiv.org" if server == "medrxiv" else "www.biorxiv.org"
        pdf_url = f"https://{host}/content/{doi}v{item.get('version', '1')}.full.pdf"
    return {
        "source": server,
        "record_type": "preprint_details",
        "doi": doi,
        "title": item.get("title"),
        "authors": item.get("authors"),
        "author_corresponding": item.get("author_corresponding"),
        "author_corresponding_institution": item.get("author_corresponding_institution"),
        "date": item.get("date"),
        "version": item.get("version"),
        "type": item.get("type"),
        "license": item.get("license"),
        "category": item.get("category"),
        "abstract": item.get("abstract"),
        "jatsxml": item.get("jatsxml"),
        "pdf_url": pdf_url,
        "raw": item,
    }


def normalize_pub(item: Dict, server: str) -> Dict:
    return {
        "source": server,
        "record_type": "preprint_publication_link",
        "preprint_doi": item.get("biorxiv_doi"),
        "published_doi": item.get("published_doi"),
        "published_journal": item.get("published_journal"),
        "preprint_title": item.get("preprint_title"),
        "preprint_authors": item.get("preprint_authors"),
        "preprint_date": item.get("preprint_date"),
        "published_date": item.get("published_date"),
        "raw": item,
    }


def iter_endpoint(session, url_template: str, key: str, max_records: int, sleep: float):
    cursor = 0
    fetched = 0
    with tqdm(total=max_records, desc=key) as pbar:
        while fetched < max_records:
            url = url_template.format(cursor=cursor)
            r = safe_get(session, url, sleep=sleep)
            if not r:
                break
            data = r.json()
            items = data.get("collection", []) or []
            if not items:
                break
            for item in items:
                yield item
                fetched += 1
                pbar.update(1)
                if fetched >= max_records:
                    break
            # API uses 100-record pages; cursor increases by result count.
            cursor += len(items)
            if len(items) < 100:
                break


def collect(server: str, start_date: str, end_date: str, max_records: int, include_pubs: bool, user_agent: str, sleep: float) -> Iterable[Dict]:
    session = get_session(user_agent)
    interval = f"{start_date}/{end_date}"

    details_template = f"{BASE}/details/{server}/{interval}/{{cursor}}/json"
    for item in iter_endpoint(session, details_template, f"{server} preprints", max_records, sleep):
        yield normalize_preprint(item, server)

    if include_pubs:
        pubs_template = f"{BASE}/pubs/{server}/{interval}/{{cursor}}/json"
        for item in iter_endpoint(session, pubs_template, f"{server} publication links", max_records, sleep):
            yield normalize_pub(item, server)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", choices=["biorxiv", "medrxiv"], required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--no-pubs", action="store_true", help="Do not call publication-link endpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    user_agent = cfg.get("user_agent")
    sleep = float(cfg.get("request_sleep_seconds", 1.0))

    open(args.out, "w", encoding="utf-8").close()
    n = 0
    for rec in collect(args.server, args.start_date, args.end_date, args.max_records, not args.no_pubs, user_agent, sleep):
        append_jsonl(args.out, rec)
        n += 1
    print(f"Saved {n} records to {args.out}")


if __name__ == "__main__":
    main()
