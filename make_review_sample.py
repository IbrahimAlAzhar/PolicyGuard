#!/usr/bin/env python
"""
Expert validation helper (improvement 1).

--mode sample : draw a stratified, blinded sample of generated critiques (equal per model x condition),
                write one sheet per reviewer (same items, independent copies) + a hidden key file.
                Reviewers fill: valid (1 = scientifically valid limitation of this paper, 0 = not),
                evidence_supported (1 = the cited passage / data line supports it, 0 = not), notes.
--mode score  : read the filled sheets, report Cohen's kappa between the two reviewers, validity and
                evidence support per condition, and how many VALID critiques the authors did not mention
                (judge found no author match), next to the pseudo-label based numbers.

  python make_review_sample.py --mode sample --per_cell 5
  python make_review_sample.py --mode score --sheets outputs/review/reviewer_1.csv,outputs/review/reviewer_2.csv
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medlim import config  # noqa: E402
from medlim.io_utils import read_csv, done  # noqa: E402


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ok = ~(pd.isna(a) | pd.isna(b))
    a, b = a[ok].astype(int), b[ok].astype(int)
    if len(a) == 0:
        return float("nan")
    po = (a == b).mean()
    pe = sum((a == v).mean() * (b == v).mean() for v in set(a) | set(b))
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sample", "score"], required=True)
    ap.add_argument("--input_csv", default=config.DEFAULT_INPUT_CSV)
    ap.add_argument("--output_dir", default=config.OUTPUT_DIR)
    ap.add_argument("--table", default=None, help="default <output_dir>/eval/df_eval.csv")
    ap.add_argument("--eval_type", default="list")
    ap.add_argument("--per_cell", type=int, default=5, help="critiques per model x condition")
    ap.add_argument("--reviewers", type=int, default=2)
    ap.add_argument("--sheets", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rdir = os.path.join(a.output_dir, "review")
    os.makedirs(rdir, exist_ok=True)
    table = a.table or os.path.join(a.output_dir, "eval", "df_eval.csv")
    df = read_csv(table)
    if a.mode == "sample":
        rng = random.Random(a.seed)
        cols = [c for c in df.columns if c.startswith("lim_") and c.count("__") == 0]
        items = []
        for c in cols:
            pool = []
            for i in df.index:
                if not done(df.at[i, c + "__json"]):
                    continue
                M = json.loads(df.at[i, f"{c}__{a.eval_type}__matches"]) if done(df.at[i, f"{c}__{a.eval_type}__matches"]) else []
                ref = json.loads(df.at[i, "full_reference_json"])
                for k, it in enumerate(json.loads(df.at[i, c + "__json"])):
                    hit = [ref[r] for r, g in M if g == k and r < len(ref)]
                    pool.append({"pmid": df.at[i, config.COL_ID], "title": df.at[i, config.COL_TITLE],
                                 "critique": it.get("limitation"), "evidence": it.get("evidence_span") or it.get("evidence", ""),
                                 "ph_tags": ",".join(it.get("ph_tags", []) or []), "_column": c,
                                 "_judge_author_match": any(h["origin"] == "author" for h in hit),
                                 "_judge_pseudo_match": any(h["origin"] == "pseudo" for h in hit)})
            items += rng.sample(pool, min(a.per_cell, len(pool)))
        rng.shuffle(items)
        for n, it in enumerate(items):
            it["item_id"] = f"R{n + 1:04d}"
        key = pd.DataFrame(items)
        key.to_csv(os.path.join(rdir, "review_key_HIDDEN.csv"), index=False)
        sheet = key[["item_id", "pmid", "title", "critique", "evidence", "ph_tags"]].copy()
        for c in ("valid", "evidence_supported", "notes"):
            sheet[c] = ""
        for r in range(a.reviewers):
            sheet.to_csv(os.path.join(rdir, f"reviewer_{r + 1}.csv"), index=False)
        print(f"{len(items)} critiques -> {rdir}/reviewer_1..{a.reviewers}.csv (key: review_key_HIDDEN.csv)")
        return
    key = pd.read_csv(os.path.join(rdir, "review_key_HIDDEN.csv"))
    sheets = [pd.read_csv(p) for p in a.sheets.split(",")]
    m = key.copy()
    for n, s in enumerate(sheets):
        m = m.merge(s[["item_id", "valid", "evidence_supported"]].rename(
            columns={"valid": f"valid_{n + 1}", "evidence_supported": f"ev_{n + 1}"}), on="item_id")
    L = [f"EXPERT REVIEW  {len(m)} critiques, {len(sheets)} reviewers"]
    if len(sheets) >= 2:
        L.append(f"Cohen's kappa validity {kappa(m.valid_1, m.valid_2):.3f}; evidence support {kappa(m.ev_1, m.ev_2):.3f}")
    vcols = [c for c in m.columns if c.startswith("valid_")]
    ecols = [c for c in m.columns if c.startswith("ev_")]
    m["valid_mean"] = m[vcols].mean(axis=1)
    m["ev_mean"] = m[ecols].mean(axis=1)
    m["valid_both"] = (m[vcols] == 1).all(axis=1)
    g = m.groupby("_column").agg(n=("item_id", "size"), valid=("valid_mean", "mean"), evidence=("ev_mean", "mean"),
                                 valid_not_by_authors=("valid_both", lambda s: int((s & ~m.loc[s.index, "_judge_author_match"]).sum())),
                                 judge_pseudo_match=("_judge_pseudo_match", "mean"))
    L.append(g.to_string())
    pm = m[m._judge_pseudo_match & ~m._judge_author_match]
    if len(pm):
        L.append(f"precision of pseudo matches (critiques matched only to pseudo labels, judged valid by all "
                 f"reviewers): {pm.valid_both.mean():.3f} over {len(pm)}")
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(rdir, "review_scores.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
