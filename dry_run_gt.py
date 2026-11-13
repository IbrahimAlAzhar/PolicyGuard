#!/usr/bin/env python
"""
DRY RUN (no GPU, no vLLM, no API key): show how the author-mentioned limitations (ground truth) are
split into units, exactly as generate.py / build_reference.py / evaluate.py will split them.

Rule: limitations_section is split on '\\n'; bullets ('-', '1.', 'a)') and '**' are removed; units with
fewer than --min_words words, header lines ending with ':' and exact duplicates are dropped.

Writes <output_dir>/dry_run_gt.txt  (every row: raw text, kept units, dropped lines, counts + summary)
and    <output_dir>/dry_run_gt.csv  (one row per paper: id, title, n_units, units_json)

  python dry_run_gt.py --n_rows 100
  python dry_run_gt.py --n_rows 100 --start_row 50 --min_words 3 --with_ph
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medlim import config  # noqa: E402
from medlim.units import split_units  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_csv", default=config.DEFAULT_INPUT_CSV)
    ap.add_argument("--output_dir", default=config.OUTPUT_DIR)
    ap.add_argument("--gt_col", default=config.COL_AUTHOR_LIM)
    ap.add_argument("--n_rows", type=int, default=100, help="-1 = all rows")
    ap.add_argument("--start_row", type=int, default=0)
    ap.add_argument("--min_words", type=int, default=3)
    ap.add_argument("--with_ph", action="store_true", help="also show the public-health context of each paper")
    ap.add_argument("--out_name", default="dry_run_gt")
    a = ap.parse_args()

    df = pd.read_csv(a.input_csv, low_memory=False)
    if a.gt_col not in df.columns:
        raise SystemExit(f"column '{a.gt_col}' not in {a.input_csv}; columns: {list(df.columns)}")
    end = None if a.n_rows < 0 else a.start_row + a.n_rows
    sub = df.iloc[a.start_row:end]
    if a.with_ph:
        from medlim import publichealth as PH

    L, rows, counts = [], [], []
    W = L.append
    W("=" * 110)
    W(f" DRY RUN - author-mentioned limitations (ground truth)   {time.strftime('%Y-%m-%d %H:%M')}")
    W(f" input : {a.input_csv}  ({len(df)} rows in file)")
    W(f" rows  : {a.start_row}..{a.start_row + len(sub) - 1} ({len(sub)} papers);  column: {a.gt_col}")
    W(f" rule  : split on '\\n'; bullets/'**' removed; < {a.min_words} words, header lines and duplicates dropped")
    W("=" * 110)
    for i, r in sub.iterrows():
        raw = r[a.gt_col]
        units, dropped = split_units(raw, ("\n",), a.min_words)
        counts.append(len(units))
        pid = r.get(config.COL_ID, i)
        W(f"\n{'#' * 110}\n# ROW {i}   {config.COL_ID}={pid}   -> {len(units)} author limitation(s)")
        W(f"# title: {str(r.get(config.COL_TITLE, ''))[:150]}")
        W("#" * 110)
        txt = raw if isinstance(raw, str) else ""
        W(f"RAW ({len(txt):,} chars, {txt.count(chr(10)) + 1 if txt else 0} lines):")
        W("   " + (txt[:1500].replace("\n", "\n   ") + (" ..." if len(txt) > 1500 else "") if txt else "(empty)"))
        W("UNITS:")
        for k, u in enumerate(units):
            W(f"   GT{k + 1}: {u}")
        if not units:
            W("   (none -> this paper will rely on pseudo labels only)")
        for d in dropped:
            W(f"   (dropped) {d[:200]}")
        if a.with_ph:
            ctx = PH.build_context(str(r.get(config.COL_TITLE, "") or ""), str(r.get(config.COL_ABSTRACT, "") or ""),
                                   str(r.get(config.COL_TEXT, "") or ""))
            W("PUBLIC-HEALTH CONTEXT:")
            W("   " + (ctx["text"].replace("\n", "\n   ") or "(none)"))
            if ctx["notes"]:
                W(f"   notes: {ctx['notes']}")
        rows.append({"row": i, config.COL_ID: pid, "title": r.get(config.COL_TITLE, ""), "n_units": len(units),
                     "units_json": json.dumps(units, ensure_ascii=False), "n_dropped": len(dropped)})

    n = np.array(counts)
    long_ones = [x for x in rows if x["n_units"] == 1]
    W(f"\n{'=' * 110}\nSUMMARY over {len(n)} papers")
    W(f"   total author limitation units : {n.sum()}")
    W(f"   per paper: mean {n.mean():.2f}, median {np.median(n):.0f}, min {n.min()}, max {n.max()}")
    W(f"   papers with 0 units (no ground truth): {(n == 0).sum()}")
    W(f"   papers with exactly 1 unit (often a whole paragraph on one line - check splitting): {len(long_ones)}")
    W("   distribution (#units: #papers): " + ", ".join(f"{k}:{v}" for k, v in sorted(Counter(n.tolist()).items())))
    W("=" * 110)
    txt = "\n".join(L) + "\n"
    os.makedirs(a.output_dir, exist_ok=True)
    p_txt = os.path.join(a.output_dir, f"{a.out_name}.txt")
    open(p_txt, "w").write(txt)
    pd.DataFrame(rows).to_csv(os.path.join(a.output_dir, f"{a.out_name}.csv"), index=False)
    print("\n".join(L[-8:]))
    print(f"\nwrote {p_txt} and {a.out_name}.csv")


if __name__ == "__main__":
    main()
