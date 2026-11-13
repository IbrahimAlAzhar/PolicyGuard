#!/usr/bin/env python3
"""Download HRSA HPSA (shortage) and MUA (underserved) facility/area data."""
import sys
import time
import requests
from pathlib import Path

OUT_DIR = Path("/lstr/sahara/datalab-ml/ibrahim/limagents_update/med_limagnets/data/hrsa_shortage_areas")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Stable HRSA download endpoints
FILES = {
    "hpsa_primary_care.csv":   "https://data.hrsa.gov/DataDownload/DD_Files/BCD_HPSA_FCT_DET_PC.csv",
    "hpsa_mental_health.csv":  "https://data.hrsa.gov/DataDownload/DD_Files/BCD_HPSA_FCT_DET_MH.csv",
    "hpsa_dental.csv":         "https://data.hrsa.gov/DataDownload/DD_Files/BCD_HPSA_FCT_DET_DH.csv",
    "mua.csv":                 "https://data.hrsa.gov/DataDownload/DD_Files/MUA_DET.csv",
    "hpsa_components.csv":     "https://data.hrsa.gov/DataDownload/DD_Files/BCD_HPSA_FCT_CMP.csv",
}

def download(url, dest):
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    print(f"  Saved {dest.name} ({dest.stat().st_size/1e6:.1f} MB)", flush=True)

def main():
    for name, url in FILES.items():
        dest = OUT_DIR / name
        if dest.exists() and dest.stat().st_size > 100:
            print(f"Skip {name}")
            continue
        for attempt in range(3):
            try:
                print(f"Downloading {name}", flush=True)
                download(url, dest)
                break
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {e}", flush=True)
                time.sleep(10)
        else:
            print(f"FAILED: {name}", flush=True)
            sys.exit(1)
        time.sleep(2)
    print("All HRSA downloads complete.")

if __name__ == "__main__":
    main()