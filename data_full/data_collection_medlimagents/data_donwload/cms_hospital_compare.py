#!/usr/bin/env python3
"""Download CMS Hospital Compare datasets via the provider-data catalog API."""
import re
import sys
import time
import requests
from pathlib import Path

OUT_DIR = Path("/lstr/sahara/datalab-ml/ibrahim/limagents_update/med_limagnets/data/cms_hospital_compare")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATALOG_URL = "https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items"
KEYWORDS = ["hospital", "hcahps", "hai", "readmission", "mortality",
            "complications", "payment and value of care", "outpatient",
            "psychiatric", "emergency department"]

def safe_name(s, maxlen=120):
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return s[:maxlen] or "untitled"

def is_hospital(ds):
    title = (ds.get("title") or "").lower()
    if any(k in title for k in KEYWORDS):
        return True
    themes = ds.get("theme") or []
    for t in themes:
        if isinstance(t, dict):
            t = t.get("data", "")
        if isinstance(t, str) and "hospital" in t.lower():
            return True
    return False

def download(url, dest):
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    print(f"  Saved {dest.name} ({dest.stat().st_size/1e6:.1f} MB)", flush=True)

def main():
    print("Fetching CMS catalog...", flush=True)
    r = requests.get(CATALOG_URL, timeout=120)
    r.raise_for_status()
    catalog = r.json()
    hospital = [d for d in catalog if is_hospital(d)]
    print(f"Found {len(hospital)} hospital-related datasets", flush=True)

    for ds in hospital:
        title = safe_name(ds.get("title", "untitled"))
        dists = ds.get("distribution", []) or []
        for i, dist in enumerate(dists):
            url = dist.get("downloadURL") or (dist.get("data") or {}).get("downloadURL")
            if not url:
                continue
            if not url.lower().endswith((".csv", ".zip")):
                continue
            ext = ".zip" if url.lower().endswith(".zip") else ".csv"
            suffix = f"_{i}" if i > 0 else ""
            dest = OUT_DIR / f"{title}{suffix}{ext}"
            if dest.exists() and dest.stat().st_size > 100:
                print(f"Skip {dest.name}")
                continue
            for attempt in range(3):
                try:
                    print(f"Downloading {dest.name}", flush=True)
                    download(url, dest)
                    break
                except Exception as e:
                    print(f"  Attempt {attempt+1} failed: {e}", flush=True)
                    time.sleep(10)
            time.sleep(1)
    print("All CMS Hospital Compare downloads complete.")

if __name__ == "__main__":
    main()