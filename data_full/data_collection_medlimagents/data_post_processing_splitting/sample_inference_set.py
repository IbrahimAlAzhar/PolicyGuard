"""
sample_inference_set.py

Build the 500-paper inference set + train/test split for the RL phase.
Remaining papers form the RAG corpus for Experiment 2.

Strategy:
    1. Drop papers older than YEAR_MIN.
    2. Sort by (relevance_score desc, publication_year desc).
    3. Greedy take with per-journal cap so no journal dominates.
    4. 70/30 split → 350 train, 150 test (RL phase).
    5. Everything NOT in the 500 → rag_corpus.csv.

Outputs (in data/pubmed_post_processed/):
    inference_set_500.csv
    train_350.csv
    test_150.csv
    rag_corpus.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA_ROOT = Path("/lstr/sahara/datalab-ml/ibrahim/limagents_update/med_limagnets/data")
SRC = DATA_ROOT / "pubmed_post_processed" / "filtered_pubmed_rel_above_3.csv"
OUT_DIR = DATA_ROOT / "pubmed_post_processed"

# Config — tweak if your numbers come out wrong
YEAR_MIN = 2018            # widened from 2020 — PMC has ~2yr OA lag, 2020+ alone gives only ~700 papers
TARGET_N = 500
PER_JOURNAL_CAP = 40       # raised from 25 — PLoS ONE / JAMA NO have plenty of papers; allow more
PER_YEAR_JOURNAL_CAP = 15  # NEW: prevents one (journal, year) from filling the journal quota
TRAIN_FRAC = 0.70
SEED = 42


def main():
    print(f"Loading {SRC.name}...")
    df = pd.read_csv(SRC, low_memory=False)
    print(f"  rows loaded:                    {len(df):,}")

    # Diagnostics step 1: drop missing year
    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
    n_with_year = df["publication_year"].notna().sum()
    print(f"  rows with valid year:           {n_with_year:,}")

    # Year filter
    df = df[df["publication_year"] >= YEAR_MIN].copy()
    print(f"  after year >= {YEAR_MIN}:           {len(df):,}")

    # Diagnostics step 2: unique journals at this stage
    n_unique_journals = df["journal"].astype(str).str.lower().nunique()
    journal_capacity = n_unique_journals * PER_JOURNAL_CAP
    print(f"  unique journals (post-year):    {n_unique_journals:,}")
    print(f"  theoretical cap = J × cap:      {journal_capacity:,}  "
          f"(this is the ceiling on selection)")
    if journal_capacity < TARGET_N:
        print(f"  [warn] journal cap ceiling ({journal_capacity}) < TARGET_N ({TARGET_N}). "
              f"Raise PER_JOURNAL_CAP or lower YEAR_MIN to fix.")

    # Sort: high relevance first, then most recent
    df = df.sort_values(
        ["relevance_score", "publication_year"], ascending=[False, False]
    ).reset_index(drop=True)

    # Greedy take with BOTH per-journal and per-(journal,year) caps.
    # The per-(journal,year) cap is what fixes the "all from 2020" artifact —
    # no single (journal, year) combo can dominate the journal's quota.
    picked_rows = []
    journal_counts: dict = {}
    journal_year_counts: dict = {}
    for _, row in df.iterrows():
        if len(picked_rows) >= TARGET_N:
            break
        j = str(row.get("journal", "")).strip().lower()
        y = int(row.get("publication_year", 0) or 0)
        if journal_counts.get(j, 0) >= PER_JOURNAL_CAP:
            continue
        if journal_year_counts.get((j, y), 0) >= PER_YEAR_JOURNAL_CAP:
            continue
        picked_rows.append(row)
        journal_counts[j] = journal_counts.get(j, 0) + 1
        journal_year_counts[(j, y)] = journal_year_counts.get((j, y), 0) + 1

    picked = pd.DataFrame(picked_rows).reset_index(drop=True)
    print(f"  picked after greedy take:       {len(picked):,}")

    if len(picked) < TARGET_N:
        print(f"  [note] only {len(picked)} survived caps (target was {TARGET_N}). "
              "Proceeding with what we have; train/test will be split as fractions.")

    # Train/test split — FRACTION-BASED, not fixed counts (fixes the empty-test bug)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(picked))
    n_train = int(round(len(picked) * TRAIN_FRAC))
    n_test = len(picked) - n_train
    train = picked.iloc[perm[:n_train]].reset_index(drop=True)
    test = picked.iloc[perm[n_train:]].reset_index(drop=True)
    print(f"  split: train={len(train):,}  test={len(test):,}  "
          f"(TRAIN_FRAC={TRAIN_FRAC})")

    # RAG corpus = everything NOT in the 500
    picked_pmids = set(picked["pmid"].astype(str))
    all_df = pd.read_csv(SRC, low_memory=False)
    rag = all_df[~all_df["pmid"].astype(str).isin(picked_pmids)].reset_index(drop=True)

    # Write — filenames reflect actual counts so you don't mislead yourself later
    inference_path = OUT_DIR / f"inference_set_{len(picked)}.csv"
    train_path = OUT_DIR / f"train_{len(train)}.csv"
    test_path = OUT_DIR / f"test_{len(test)}.csv"
    rag_path = OUT_DIR / "rag_corpus.csv"
    picked.to_csv(inference_path, index=False)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    rag.to_csv(rag_path, index=False)

    print(f"\nWrote:")
    print(f"  {inference_path.name}   {len(picked):,}")
    print(f"  {train_path.name}       {len(train):,}")
    print(f"  {test_path.name}        {len(test):,}")
    print(f"  {rag_path.name}          {len(rag):,}")

    print(f"\nYear distribution in 500:")
    print(picked["publication_year"].value_counts().sort_index().to_string())
    print(f"\nTop 10 journals in 500:")
    print(picked["journal"].value_counts().head(10).to_string())
    print(f"\nRelevance score in 500:")
    print(picked["relevance_score"].describe().to_string())


if __name__ == "__main__":
    main()