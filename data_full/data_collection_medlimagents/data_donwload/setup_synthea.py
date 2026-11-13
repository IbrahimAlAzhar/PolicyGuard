#!/usr/bin/env python3
"""One-time Synthea setup: clone + build + flip exporter to CSV.

Run this ONCE before submitting the all-states array job.
Generation happens in generate_synthea_state.py.
"""
import subprocess
import sys
from pathlib import Path

BASE = Path("/lstr/sahara/datalab-ml/ibrahim/limagents_update/med_limagnets/data/synthea")
BASE.mkdir(parents=True, exist_ok=True)
REPO = BASE / "synthea"

def run(cmd, cwd=None):
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, cwd=cwd)

def main():
    try:
        out = subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT)
        print("Java:", out.decode().splitlines()[0])
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: java not on PATH. Load a Java 11+ module in the PBS file.")
        sys.exit(1)

    if not REPO.exists():
        run(["git", "clone", "--depth", "1",
             "https://github.com/synthetichealth/synthea.git", str(REPO)])
    else:
        print(f"Synthea repo already at {REPO} — skipping clone")

    print("Building Synthea (Gradle, ~3-5 min)...", flush=True)
    run(["./gradlew", "build", "-x", "test"], cwd=str(REPO))

    # Configure exporter to CSV only (smaller, tabular, faster)
    props = REPO / "src/main/resources/synthea.properties"
    text = props.read_text()
    for old, new in [
        ("exporter.csv.export = false",     "exporter.csv.export = true"),
        ("exporter.fhir.export = true",     "exporter.fhir.export = false"),
        ("exporter.hospital.fhir.export = true",     "exporter.hospital.fhir.export = false"),
        ("exporter.practitioner.fhir.export = true", "exporter.practitioner.fhir.export = false"),
    ]:
        text = text.replace(old, new)
    props.write_text(text)
    print("Exporter configured for CSV. Build complete — ready for array job.")

if __name__ == "__main__":
    main()