#!/usr/bin/env python3
"""Download CDC/ATSDR Social Vulnerability Index (SVI) via the lpiep/cdc-svi mirror.

Background: CDC removed the SVI from svi.cdc.gov in January 2025 (administration
policy). The original direct CSV URLs all return 404. The lpiep/cdc-svi GitHub
repo archived the complete CSV + GDB collection (years 2000, 2010, 2014, 2016,
2018, 2020, 2022) as downloaded from svi.cdc.gov on 2025-01-31, which is the
most reliable current source.

This script clones the mirror and copies the CSV files into the data directory.
"""
import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path("/lstr/sahara/datalab-ml/ibrahim/limagents_update/med_limagnets/data/cdc_svi")
OUT.mkdir(parents=True, exist_ok=True)

REPO = OUT / "_lpiep_cdc_svi_mirror"
REPO_URL = "https://github.com/lpiep/cdc-svi.git"

def run(cmd, cwd=None):
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, cwd=cwd)

def main():
    if shutil.which("git") is None:
        print("ERROR: git not on PATH.")
        print("Fix: `conda install -n image_lim -c conda-forge git` from a login node.")
        sys.exit(1)

    if REPO.exists() and (REPO / ".git").exists():
        print(f"Mirror already at {REPO} — pulling latest")
        run(["git", "pull", "--ff-only"], cwd=str(REPO))
    else:
        if REPO.exists():
            shutil.rmtree(REPO)
        run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)])

    csv_src = REPO / "csv"
    if not csv_src.exists():
        print(f"ERROR: csv/ not found in mirror at {csv_src}")
        sys.exit(1)

    # Copy every archived CSV (preserves any subdirectory layout in the mirror)
    count, total_bytes = 0, 0
    for p in sorted(csv_src.rglob("*.csv")):
        rel = p.relative_to(csv_src)
        dest = OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        size = dest.stat().st_size
        total_bytes += size
        count += 1
        print(f"  {rel}  ({size/1e6:.1f} MB)")

    print(f"\nCopied {count} SVI CSV files, {total_bytes/1e9:.2f} GB total -> {OUT}")
    print("All available SVI years (2000, 2010, 2014, 2016, 2018, 2020, 2022) are included.")

if __name__ == "__main__":
    main()