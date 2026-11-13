"""Discover DOI metadata through Crossref REST API.

Example:
python -m collectors.crossref_discover \
  --query 'health equity subgroup external validity clinical prediction model' \
  --from-date 2022-01-01 --until-date 2026-12-31 \
  --max-records 500 --out data/metadata/crossref_health_equity.jsonl
"""
import argparse
from typing import Dict, Iterable
from urllib.parse import urlencode

from tqdm import tqdm

from utils.io import append_jsonl, get_session, load_config, safe_get

CROSSREF_WORKS = "https://api.crossref.org/works"


def normalize_work(item: Dict) -> Dict:
    links = item.get("link", []) or []
    return {
        "source": "crossref",
        "doi": item.get("DOI"),
        "type": item.get("type"),
        "title": " ".join(item.get("title", []) or []),
        "container_title": " ".join(item.get("container-title", []) or []),
        "publisher": item.get("publisher"),
        "published_print": item.get("published-print"),
        "published_online": item.get("published-online"),
        "created": item.get("created"),
        "URL": item.get("URL"),
        "ISSN": item.get("ISSN"),
        "subject": item.get("subject"),
        "abstract": item.get("abstract"),
        "license": item.get("license"),
        "links": links,
        "raw": item,
    }


def crossref_search(query: str, from_date: str, until_date: str, max_records: int, user_agent: str, sleep: float) -> Iterable[Dict]:
    session = get_session(user_agent)
    cursor = "*"
    fetched = 0

    with tqdm(total=max_records, desc="Crossref records") as pbar:
        while fetched < max_records:
            rows = min(100, max_records - fetched)
            params = {
                "query.bibliographic": query,
                "filter": f"type:journal-article,from-pub-date:{from_date},until-pub-date:{until_date}",
                "select": "DOI,type,title,container-title,publisher,published-print,published-online,created,URL,ISSN,subject,abstract,license,link",
                "rows": rows,
                "cursor": cursor,
            }
            url = CROSSREF_WORKS + "?" + urlencode(params)
            r = safe_get(session, url, sleep=sleep)
            if not r:
                break
            data = r.json().get("message", {})
            items = data.get("items", [])
            if not items:
                break
            for item in items:
                yield normalize_work(item)
                fetched += 1
                pbar.update(1)
                if fetched >= max_records:
                    break
            next_cursor = data.get("next-cursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--from-date", default="2022-01-01")
    parser.add_argument("--until-date", default="2026-12-31")
    parser.add_argument("--max-records", type=int, default=500)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    user_agent = cfg.get("user_agent")
    sleep = float(cfg.get("request_sleep_seconds", 1.0))

    open(args.out, "w", encoding="utf-8").close()
    n = 0
    for rec in crossref_search(args.query, args.from_date, args.until_date, args.max_records, user_agent, sleep):
        append_jsonl(args.out, rec)
        n += 1
    print(f"Saved {n} records to {args.out}")


if __name__ == "__main__":
    main()
