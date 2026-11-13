#!/usr/bin/env python3
"""Download CDC PLACES data (census tract, county, ZCTA)."""
import sys
import time
import requests
from pathlib import Path

OUT_DIR = Path("/lstr/sahara/datalab-ml/ibrahim/limagents_update/med_limagnets/data/cdc_places")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Socrata bulk CSV exports — most current release
DATASETS = {
    "places_tract.csv":  "https://data.cdc.gov/api/views/cwsq-ngmh/rows.csv?accessType=DOWNLOAD",
    "places_county.csv": "https://data.cdc.gov/api/views/swc5-untb/rows.csv?accessType=DOWNLOAD",
    "places_zcta.csv":   "https://data.cdc.gov/api/views/qnzd-25i4/rows.csv?accessType=DOWNLOAD",
    "places_place.csv":  "https://data.cdc.gov/api/views/eav7-hnsx/rows.csv?accessType=DOWNLOAD",
}

def download(url, dest):
    print(f"Downloading {dest.name}", flush=True)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        got = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    got += len(chunk)
        size_mb = dest.stat().st_size / 1e6
        print(f"  Saved {dest.name} ({size_mb:.1f} MB)", flush=True)

def main():
    for name, url in DATASETS.items():
        dest = OUT_DIR / name
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"Skip {name} (already exists)")
            continue
        for attempt in range(3):
            try:
                download(url, dest)
                break
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {e}", flush=True)
                time.sleep(15)
        else:
            print(f"FAILED: {name}", flush=True)
            sys.exit(1)
        time.sleep(3)
    print("All CDC PLACES downloads complete.")

if __name__ == "__main__":
    main()