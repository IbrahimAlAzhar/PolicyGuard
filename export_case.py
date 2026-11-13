#!/usr/bin/env python
"""
Export ONE complete, auditable case: manuscript passage, exact public-data release and rows,
generated critique, judge decision, reviewer judgment (if available) and simulation inputs/outputs
(if you pass them). Writes case_<pmid>_<column>.json and .md into <output_dir>/cases/.

  python export_case.py --row 3 --column lim_qwen_agents_rag_ph
  python export_case.py --pmid 31234567 --column lim_gemma_agents_rag_ph --item 2 \
      --review_csv outputs/review/review_filled.csv --simulation_json my_sim.json
"""
import argparse
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medlim import config  # noqa: E402
from medlim.io_utils import read_csv, done  # noqa: E402
from medlim.publichealth import fingerprint, _load  # noqa: E402


def locate(span, text, pad=350):
    if not span or not text:
        return None
    key = re.sub(r"\s+", " ", span.strip())[:60].lower()
    flat = re.sub(r"\s+", " ", text)
    j = flat.lower().find(key)
    if j < 0:
        words = key.split()[:6]
        j = flat.lower().find(" ".join(words)) if words else -1
    if j < 0:
        return None
    return {"char_offset_in_full_text_excl_lim": j, "context": flat[max(0, j - pad): j + len(span) + pad]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default=config.DEFAULT_INPUT_CSV)
    ap.add_argument("--output_dir", default=config.OUTPUT_DIR)
    ap.add_argument("--table", default=None, help="df_eval.csv or <stem>__reference.csv (default: eval if present)")
    ap.add_argument("--row", type=int, default=None)
    ap.add_argument("--pmid", default=None)
    ap.add_argument("--column", default="lim_qwen_agents_rag_ph")
    ap.add_argument("--item", type=int, default=None, help="1-based critique number (default: first one citing [PH-n])")
    ap.add_argument("--eval_type", default="list")
    ap.add_argument("--review_csv", default=None, help="filled review sheet from make_review_sample.py")
    ap.add_argument("--simulation_json", default=None, help="population-scenario inputs/outputs to attach")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.input_csv))[0]
    cand = [args.table, os.path.join(args.output_dir, "eval", "df_eval.csv"),
            os.path.join(args.output_dir, f"{stem}__reference.csv")]
    path = next(p for p in cand if p and os.path.exists(p))
    df = read_csv(path)
    if args.pmid is not None:
        i = df.index[df[config.COL_ID].astype(str) == str(args.pmid)][0]
    else:
        i = args.row if args.row is not None else df.index[0]
    c = args.column
    items = json.loads(df.at[i, c + "__json"])
    if not items:
        raise SystemExit("no critiques in this cell")
    if args.item:
        k = args.item - 1
    else:
        k = next((n for n, x in enumerate(items) if x.get("ph_tags")), 0)
    it = items[k]
    prov = json.loads(df.at[i, "ph_provenance_json"]) if done(df.at[i, "ph_provenance_json"]) else []
    tags = set(it.get("ph_tags") or re.findall(r"PH-\d+", json.dumps(it)))
    pub = []
    for p in prov:
        if p["tag"] in tags:
            rec = dict(p)
            rec["current_file_fingerprint"] = fingerprint(p.get("file", ""))
            rec["fingerprint_matches_run"] = rec["current_file_fingerprint"].get("sha256") == p.get("sha256")
            t = _load(p.get("file", ""))
            if t is not None and p.get("row_indices"):
                idx = [r for r in p["row_indices"][:10] if r in t.index]
                rec["rows_verbatim"] = json.loads(t.loc[idx].to_json(orient="records", default_handler=str))
            pub.append(rec)
    ref = json.loads(df.at[i, "full_reference_json"]) if "full_reference_json" in df.columns and done(df.at[i, "full_reference_json"]) else []
    judge = None
    mcol = f"{c}__{args.eval_type}__matches"
    if mcol in df.columns and done(df.at[i, mcol]):
        M = json.loads(df.at[i, mcol])
        judge = {"eval_type": args.eval_type, "matched_reference_items": [ref[r] for r, g in M if g == k and r < len(ref)]}
    review = None
    if args.review_csv and os.path.exists(args.review_csv):
        rv = pd.read_csv(args.review_csv)
        hit = rv[(rv.get("pmid").astype(str) == str(df.at[i, config.COL_ID])) & (rv.get("critique") == it["limitation"])] \
            if "pmid" in rv.columns else rv.iloc[0:0]
        review = json.loads(hit.to_json(orient="records")) if len(hit) else "no reviewer judgment for this critique"
    case = {
        "paper": {"row": int(i), "pmid": str(df.at[i, config.COL_ID]) if config.COL_ID in df.columns else None,
                  "title": str(df.at[i, config.COL_TITLE])},
        "system": {"column": c, "meta": json.loads(df.at[i, c + "__meta"]) if done(df.at[i, c + "__meta"]) else None},
        "generated_critique": it,
        "manuscript_passage": locate(it.get("evidence_span") or it.get("evidence", ""), str(df.at[i, config.COL_TEXT])),
        "public_data": pub,
        "public_health_context_shown_to_model": df.at[i, "ph_context"],
        "reference_set": ref,
        "judge": judge,
        "reviewer_judgment": review,
        "simulation": json.load(open(args.simulation_json)) if args.simulation_json else
        "not run (attach with --simulation_json: inputs, assumptions, outputs)",
    }
    out = os.path.join(args.output_dir, "cases")
    os.makedirs(out, exist_ok=True)
    base = os.path.join(out, f"case_{case['paper']['pmid']}_{c}")
    json.dump(case, open(base + ".json", "w"), indent=2, ensure_ascii=False, default=str)
    md = [f"# Traceable case: {case['paper']['title']}", f"PMID {case['paper']['pmid']} - system `{c}` - critique {k + 1}",
          "\n## Generated critique", it["limitation"], f"\nseverity {it.get('severity')}, agents {it.get('agents')}, "
          f"anchored {it.get('anchored')}", "\n## Manuscript passage",
          f"> {(case['manuscript_passage'] or {}).get('context', it.get('evidence_span') or '(none)')}",
          "\n## Public data used"]
    for p in pub:
        md.append(f"- [{p['tag']}] {p['text']}\n  - file `{p.get('file')}` sha256 `{p.get('sha256')}` "
                  f"(mtime {p.get('mtime')}); release {p.get('release')}; filter `{p.get('filter')}`; rows "
                  f"{p.get('row_indices', [])[:10]}; unchanged since run: {p.get('fingerprint_matches_run')}")
    if not pub:
        md.append("(this critique cites no public-data line)")
    md += ["\n## Judge", json.dumps(judge, indent=1, ensure_ascii=False) if judge else "(not evaluated yet)",
           "\n## Reviewer judgment", json.dumps(review, indent=1) if review else "(none)",
           "\n## Simulation", json.dumps(case["simulation"], indent=1) if isinstance(case["simulation"], dict) else case["simulation"]]
    open(base + ".md", "w").write("\n".join(md) + "\n")
    print(f"wrote {base}.json and {base}.md")


if __name__ == "__main__":
    main()
