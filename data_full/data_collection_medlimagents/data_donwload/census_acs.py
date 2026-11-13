#!/usr/bin/env python3
"""Download Census ACS 5-year tract-level demographics for all US states."""
import os
import sys
import time
import json
import requests
from pathlib import Path

OUT_DIR = Path("/lstr/sahara/datalab-ml/ibrahim/limagents_update/med_limagnets/data/census_acs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional: export CENSUS_API_KEY=... in PBS file (free key from https://api.census.gov/data/key_signup.html)
API_KEY = os.environ.get("CENSUS_API_KEY", "").strip()
YEAR = 2022
BASE = f"https://api.census.gov/data/{YEAR}/acs/acs5"

# Demographic / socioeconomic / health-relevant variables
VARS = [
    "NAME",
    "B01003_001E",  # Total population
    "B01002_001E",  # Median age
    "B19013_001E",  # Median household income
    "B17001_002E",  # Pop below poverty
    "B17001_001E",  # Poverty universe
    "B25077_001E",  # Median home value
    "B15003_022E",  # Bachelor's degree (25+)
    "B15003_001E",  # Education universe (25+)
    "B27001_001E",  # Health insurance universe
    "B27010_017E",  # No insurance under 19
    "B02001_002E",  # White alone
    "B02001_003E",  # Black alone
    "B02001_005E",  # Asian alone
    "B03003_003E",  # Hispanic/Latino
    "B23025_005E",  # Unemployed
    "B23025_002E",  # Labor force
    "B08303_001E",  # Travel time to work
    "B25064_001E",  # Median gross rent
]

STATE_FIPS = [f"{i:02d}" for i in [
    1,2,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
    31,32,33,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50,51,53,54,55,56
]]

def fetch_state(state):
    params = {"get": ",".join(VARS), "for": "tract:*", "in": f"state:{state}"}
    if API_KEY:
        params["key"] = API_KEY
    r = requests.get(BASE, params=params, timeout=180)
    r.raise_for_status()
    return r.json()

def main():
    if not API_KEY:
        print("WARNING: CENSUS_API_KEY not set. You may hit anonymous rate limits.")
    for st in STATE_FIPS:
        dest = OUT_DIR / f"acs5_{YEAR}_tract_state{st}.json"
        if dest.exists() and dest.stat().st_size > 100:
            print(f"Skip state {st}")
            continue
        for attempt in range(3):
            try:
                data = fetch_state(st)
                with open(dest, "w") as f:
                    json.dump(data, f)
                print(f"State {st}: {len(data)-1} tracts -> {dest.name}", flush=True)
                break
            except Exception as e:
                print(f"  State {st} attempt {attempt+1} failed: {e}", flush=True)
                time.sleep(20)
        else:
            print(f"FAILED state {st}", flush=True)
        time.sleep(1)
    print("All ACS downloads complete.")

if __name__ == "__main__":
    main()