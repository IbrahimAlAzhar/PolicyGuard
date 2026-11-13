"""Discover biomedical/public-health articles through Europe PMC.

Example:
python -m collectors.europepmc_discover \
  --query '("health equity" OR disparity OR race OR ethnicity OR subgroup) AND (public health OR clinical)' \
  --from-year 2022 --max-records 500 \
  --out data/metadata/europepmc_health_equity.jsonl
"""
import argparse
import json
from typing import Dict, Iterable, List
from urllib.parse import urlencode

from tqdm import tqdm

from utils.io import append_jsonl, get_session, load_config, safe_get

EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def normalize_result(item: Dict) -> Dict:
    full_text_urls = []
    ft = item.get("fullTextUrlList", {}).get("fullTextUrl", [])
    if isinstance(ft, dict):
        ft = [ft]
    for u in ft or []:
        full_text_urls.append({
            "url": u.get("url"),
            "documentStyle": u.get("documentStyle"),
            "availability": u.get("availability"),
            "site": u.get("site"),
        })

    return {
        "source": "europe_pmc",
        "id": item.get("id"),
        "pmid": item.get("pmid"),
        "pmcid": item.get("pmcid"),
        "doi": item.get("doi"),
        "title": item.get("title"),
        "journalTitle": item.get("journalTitle"),
        "pubYear": item.get("pubYear"),
        "authorString": item.get("authorString"),
        "abstractText": item.get("abstractText"),
        "isOpenAccess": item.get("isOpenAccess"),
        "inEPMC": item.get("inEPMC"),
        "hasReferences": item.get("hasReferences"),
        "hasTextMinedTerms": item.get("hasTextMinedTerms"),
        "fullTextUrlList": full_text_urls,
        "raw": item,
    }


def europepmc_search(query: str, from_year: int, max_records: int, user_agent: str, sleep: float) -> Iterable[Dict]:
    session = get_session(user_agent)
    cursor_mark = "*"
    fetched = 0

    # Europe PMC query syntax supports FIRST_PDATE; PUB_YEAR is also commonly used.
    combined_query = f"({query}) AND FIRST_PDATE:[{from_year}-01-01 TO 3000-12-31]"

    with tqdm(total=max_records, desc="Europe PMC records") as pbar:
        while fetched < max_records:
            page_size = min(100, max_records - fetched)
            params = {
                "query": combined_query,
                "format": "json",
                "pageSize": page_size,
                "cursorMark": cursor_mark,
                "resultType": "core",
            }
            url = EUROPE_PMC_SEARCH + "?" + urlencode(params)
            r = safe_get(session, url, sleep=sleep)
            if not r:
                break
            data = r.json()
            results = data.get("resultList", {}).get("result", [])
            if not results:
                break
            for item in results:
                yield normalize_result(item)
                fetched += 1
                pbar.update(1)
                if fetched >= max_records:
                    break
            next_cursor = data.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor_mark:
                break
            cursor_mark = next_cursor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Europe PMC query string")
    parser.add_argument("--from-year", type=int, default=2022)
    parser.add_argument("--max-records", type=int, default=500)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    user_agent = cfg.get("user_agent")
    sleep = float(cfg.get("request_sleep_seconds", 1.0))

    n = 0
    # overwrite output
    open(args.out, "w", encoding="utf-8").close()
    for rec in europepmc_search(args.query, args.from_year, args.max_records, user_agent, sleep):
        append_jsonl(args.out, rec)
        n += 1
    print(f"Saved {n} records to {args.out}")


if __name__ == "__main__":
    main()
