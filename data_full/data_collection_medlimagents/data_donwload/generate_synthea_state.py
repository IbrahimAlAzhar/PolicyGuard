#!/usr/bin/env python3
"""Generate Synthea patients for ONE state, picked by PBS_ARRAY_INDEX.

Designed to run as a PBS array job (1..51). Each task writes to its own
output subdirectory so parallel tasks don't clobber each other.
"""
import os
import subprocess
import sys
from pathlib import Path

STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "District of Columbia", "Florida", "Georgia",
    "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
    "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
]

BASE = Path("/lstr/sahara/datalab-ml/ibrahim/limagents_update/med_limagnets/data/synthea")
REPO = BASE / "synthea"
OUT_BASE = BASE / "output"
OUT_BASE.mkdir(parents=True, exist_ok=True)

POP  = os.environ.get("SYNTHEA_POP",  "50000")
SEED = os.environ.get("SYNTHEA_SEED", "42")

def main():
    idx_str = os.environ.get("PBS_ARRAY_INDEX") or os.environ.get("STATE_INDEX")
    if not idx_str:
        print("ERROR: PBS_ARRAY_INDEX not set (run as part of array job).")
        sys.exit(2)
    idx = int(idx_str)
    if not (1 <= idx <= len(STATES)):
        print(f"ERROR: index {idx} out of range 1..{len(STATES)}")
        sys.exit(2)

    state = STATES[idx - 1]
    state_dir = state.replace(" ", "_")
    out_dir = OUT_BASE / state_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not REPO.exists():
        print(f"ERROR: Synthea not built. Run setup_synthea_build.pbs first.")
        sys.exit(2)

    print(f"=== State {idx}/{len(STATES)}: {state} ===", flush=True)
    print(f"Population: {POP}, seed: {SEED}", flush=True)
    print(f"Output dir: {out_dir}", flush=True)

    # Per-state seed to keep cohorts independent across the run
    seed = str(int(SEED) + idx)
    cmd = [
        "./run_synthea",
        "-p", POP,
        "-s", seed,
        f"--exporter.baseDirectory={out_dir}/",
        state,
    ]
    subprocess.check_call(cmd, cwd=str(REPO))

    csv_dir = out_dir / "csv"
    if csv_dir.exists():
        total = sum(p.stat().st_size for p in csv_dir.iterdir())
        print(f"Done {state}: {total/1e9:.2f} GB of CSV", flush=True)
    else:
        print(f"Done {state}: no csv/ subdir found", flush=True)

if __name__ == "__main__":
    main() 
