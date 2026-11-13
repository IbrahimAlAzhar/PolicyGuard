#!/usr/bin/env python
"""
STAGE 1b - majority vote with gpt-4o-mini -> pseudo-reference -> full reference set.

1. Merge the three per-model generation files (<stem>__gen_<model>.csv) into one table.
2. For every paper, gpt-4o-mini clusters the ADDITIONAL limitations proposed by qwen / gemma / mistral
   (anonymised as M1/M2/M3, order shuffled per paper) and flags clusters that restate an author limitation.
3. votes = number of DIFFERENT models in a cluster (counted in code, not by the LLM).
   A cluster becomes a pseudo label when votes >= --min_votes (default 2 = majority of 3) and it does
   not duplicate an author limitation. Weight: 3 votes -> 0.7, 2 votes -> 0.5 (--vote_weights).
4. full reference = author limitations (weight 1.0, split on '\\n') + pseudo labels (weight 0.5 / 0.7).

Output: <output_dir>/<stem>__reference.csv  (all generation columns + reference columns), rewritten every
--save_every rows; resumable. Also writes reference_report.txt.

  python build_reference.py --n_rows 10
  python build_reference.py --n_rows -1 --min_votes 3            # unanimous only
  python build_reference.py --n_rows -1 --then_evaluate --eval_type list
"""
import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medlim import config  # noqa: E402
from medlim import prompts as P  # noqa: E402
from medlim.io_utils import setup_logging, log, read_csv, save_csv, select_rows, ensure_cols, done  # noqa: E402
from medlim.llm import ChatLLM, FatalAPIError  # noqa: E402
from medlim.units import extract_json, format_limitations, author_units  # noqa: E402

REF_COLS = ["pseudo_vote_clusters_json", "pseudo_reference_json", "n_pseudo_reference", "n_pseudo_votes3",
            "n_pseudo_votes2", "full_reference_json", "full_reference_text", "n_full_reference", "pseudo_vote_meta"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_csv", default=config.DEFAULT_INPUT_CSV)
    ap.add_argument("--output_dir", default=config.OUTPUT_DIR)
    ap.add_argument("--models", default="qwen,gemma,mistral")
    ap.add_argument("--gen_csvs", default=None, help="comma list; default <output_dir>/<stem>__gen_<model>.csv")
    ap.add_argument("--output_csv", default=None)
    ap.add_argument("--n_rows", type=int, default=-1)
    ap.add_argument("--start_row", type=int, default=0)
    ap.add_argument("--min_votes", type=int, default=2)
    ap.add_argument("--vote_weights", default="3:0.7,2:0.5,1:0.0", help="votes:weight of a pseudo label")
    ap.add_argument("--judge_model", default=config.JUDGE_MODEL)
    ap.add_argument("--backend", default="openai", choices=["openai", "mock"])
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--then_evaluate", action="store_true", help="run evaluate.py on the result afterwards")
    ap.add_argument("--eval_type", default="list", choices=["list", "pairwise", "both"])
    return ap.parse_args()


def merge_generation_files(args, models):
    stem = os.path.splitext(os.path.basename(args.input_csv))[0]
    paths = (args.gen_csvs.split(",") if args.gen_csvs else
             [os.path.join(args.output_dir, f"{stem}__gen_{m}.csv") for m in models])
    base = None
    for m, p in zip(models, paths):
        if not os.path.exists(p):
            log(f"WARNING: no generation file for {m}: {p}")
            continue
        g = read_csv(p)
        if base is None:
            base = g
            log(f"base table from {p} ({len(g)} rows)")
            continue
        if len(g) != len(base) or (config.COL_ID in g.columns and
                                   not g[config.COL_ID].astype(str).equals(base[config.COL_ID].astype(str))):
            raise SystemExit(f"{p} does not have the same rows (same order) as the other generation files")
        new = [c for c in g.columns if c.startswith((f"lim_{m}_", f"pseudo_cand_{m}"))]
        base = base.drop(columns=[c for c in new if c in base.columns])
        base = pd.concat([base, g[new]], axis=1)
        # shared columns (author units / public-health context) are identical; fill gaps from this file
        for c in ("author_limitation_units_json", "n_author_limitations", "ph_context", "ph_provenance_json",
                  "ph_disease_code", "ph_states"):
            if c in g.columns:
                base[c] = base[c].where(base[c].notna(), g[c]) if c in base.columns else g[c]
        log(f"+ {len(new)} columns from {p}")
    if base is None:
        raise SystemExit("no generation files found - run generate.py first")
    return base, stem


async def vote_row(llm, df, i, models, args, weights, rng):
    title = str(df.at[i, config.COL_TITLE] or "")
    auth = json.loads(df.at[i, "author_limitation_units_json"]) if done(df.at[i, "author_limitation_units_json"]) \
        else author_units(df.at[i, config.COL_AUTHOR_LIM] if config.COL_AUTHOR_LIM in df.columns else "")
    lists = {}
    for m in models:
        c = f"pseudo_cand_{m}__json"
        if c in df.columns and done(df.at[i, c]):
            lst = [x.get("limitation", "") for x in json.loads(df.at[i, c]) if isinstance(x, dict)]
            lists[m] = [x for x in lst if x]
    avail = [m for m in models if m in lists]
    order = avail[:]
    rng.shuffle(order)
    anon = {f"M{j + 1}": m for j, m in enumerate(order)}
    ids = {f"{a}_{k + 1}": (m, t) for a, m in anon.items() for k, t in enumerate(lists[m])}
    meta = {"models_available": avail, "anon": anon, "n_candidates": len(ids)}
    clusters = []
    n_nonempty = sum(1 for m in avail if lists[m])
    if ids and n_nonempty >= min(args.min_votes, 2):
        raw = await llm.chat(P.VOTE_SYSTEM, P.vote_user(title, auth, {a: lists[m] for a, m in anon.items()}),
                             max_tokens=2500, json_mode=True)
        obj = extract_json(raw)
        got = obj.get("clusters", []) if isinstance(obj, dict) else []
        seen = set()
        for cl in got if isinstance(got, list) else []:
            if not isinstance(cl, dict):
                continue
            mem = [x for x in cl.get("members", []) if isinstance(x, str) and x in ids and x not in seen]
            if not mem:
                continue
            seen.update(mem)
            clusters.append({"canonical": str(cl.get("canonical") or ids[mem[0]][1]).strip(), "members": mem,
                             "member_texts": [ids[x][1] for x in mem],
                             "models": sorted({ids[x][0] for x in mem}),
                             "duplicates_author": bool(cl.get("duplicates_author", False)),
                             "author_ids": cl.get("author_ids", [])})
        for x in ids:                                   # anything the LLM forgot is a singleton
            if x not in seen:
                clusters.append({"canonical": ids[x][1], "members": [x], "member_texts": [ids[x][1]],
                                 "models": [ids[x][0]], "duplicates_author": False, "author_ids": [], "orphan": True})
        meta["parsed"] = bool(got)
    elif ids:                                           # a single model has candidates: no majority possible
        clusters = [{"canonical": t, "members": [x], "member_texts": [t], "models": [m], "duplicates_author": False,
                     "author_ids": []} for x, (m, t) in ids.items()]
        meta["parsed"] = "skipped (fewer than 2 models with candidates)"
    for cl in clusters:
        cl["votes"] = len(cl["models"])
    pseudo = [{"text": cl["canonical"], "votes": cl["votes"], "weight": weights.get(cl["votes"], 0.0),
               "models": cl["models"]}
              for cl in clusters if cl["votes"] >= args.min_votes and not cl["duplicates_author"]
              and weights.get(cl["votes"], 0.0) > 0]
    pseudo.sort(key=lambda x: -x["votes"])
    full = [{"id": f"A{k + 1}", "text": t, "origin": "author", "weight": 1.0, "votes": None} for k, t in enumerate(auth)]
    full += [{"id": f"P{k + 1}", "text": p["text"], "origin": "pseudo", "weight": p["weight"], "votes": p["votes"]}
             for k, p in enumerate(pseudo)]
    meta["n_duplicates_author"] = sum(1 for c in clusters if c["duplicates_author"])
    df.at[i, "pseudo_vote_clusters_json"] = json.dumps(clusters, ensure_ascii=False)
    df.at[i, "pseudo_reference_json"] = json.dumps(pseudo, ensure_ascii=False)
    df.at[i, "n_pseudo_reference"] = len(pseudo)
    df.at[i, "n_pseudo_votes3"] = sum(1 for p in pseudo if p["votes"] >= 3)
    df.at[i, "n_pseudo_votes2"] = sum(1 for p in pseudo if p["votes"] == 2)
    df.at[i, "full_reference_json"] = json.dumps(full, ensure_ascii=False)
    df.at[i, "full_reference_text"] = format_limitations([x["text"] for x in full]) or "(none)"
    df.at[i, "n_full_reference"] = len(full)
    df.at[i, "pseudo_vote_meta"] = json.dumps(meta, default=str)


def report(df, rows, models, args, out_csv, llm):
    L = []
    W = L.append
    W("=" * 110)
    W(f" REFERENCE SET (author + pseudo labels)   {time.strftime('%Y-%m-%d %H:%M')}")
    W("=" * 110)
    W(f" table: {out_csv}")
    W(f" rows: {len(rows)}   min_votes: {args.min_votes}   weights: {args.vote_weights}   judge: {args.judge_model}")
    na = np.array([len(json.loads(df.at[i, "author_limitation_units_json"])) for i in rows])
    W(f"\n AUTHOR-MENTIONED LIMITATIONS (split on '\\n'): total {na.sum()}, mean {na.mean():.2f}, median "
      f"{np.median(na):.0f}, min {na.min()}, max {na.max()}; papers with none: {(na == 0).sum()}")
    W("   distribution: " + ", ".join(f"{k}:{v}" for k, v in sorted(Counter(na.tolist()).items())))
    for m in models:
        c = f"pseudo_cand_{m}__json"
        if c in df.columns:
            n = [len(json.loads(df.at[i, c])) for i in rows if done(df.at[i, c])]
            W(f" pseudo candidates from {m:<8}: {len(n)} papers, mean {np.mean(n) if n else 0:.2f} per paper")
    npl = np.array([int(float(df.at[i, "n_pseudo_reference"] or 0)) for i in rows if done(df.at[i, "full_reference_json"])])
    v3 = sum(int(float(df.at[i, "n_pseudo_votes3"] or 0)) for i in rows if done(df.at[i, "full_reference_json"]))
    v2 = sum(int(float(df.at[i, "n_pseudo_votes2"] or 0)) for i in rows if done(df.at[i, "full_reference_json"]))
    W(f"\n PSEUDO LABELS: total {npl.sum()} (3 votes: {v3}, 2 votes: {v2}); mean {npl.mean() if len(npl) else 0:.2f} "
      f"per paper; papers with none: {(npl == 0).sum()}")
    no_auth_with_pseudo = sum(1 for i, a in zip(rows, na) if a == 0 and int(float(df.at[i, 'n_pseudo_reference'] or 0)) > 0)
    W(f" papers WITHOUT author limitations that now have a pseudo reference: {no_auth_with_pseudo} of {(na == 0).sum()}")
    dup = sum(json.loads(df.at[i, "pseudo_vote_meta"]).get("n_duplicates_author", 0) for i in rows
              if done(df.at[i, "pseudo_vote_meta"]))
    W(f" clusters dropped because they restate an author limitation: {dup}")
    # pairwise agreement: share of model A's candidates whose cluster also contains model B
    W("\n MODEL AGREEMENT (share of A's candidates whose cluster also contains B)")
    pair = {(a, b): [0, 0] for a in models for b in models if a != b}
    for i in rows:
        if not done(df.at[i, "pseudo_vote_clusters_json"]):
            continue
        for cl in json.loads(df.at[i, "pseudo_vote_clusters_json"]):
            for x in cl["members"]:
                a = next((m for k, m in json.loads(df.at[i, "pseudo_vote_meta"])["anon"].items()
                          if x.startswith(k + "_")), None)
                for b in models:
                    if a and b != a:
                        pair[(a, b)][1] += 1
                        pair[(a, b)][0] += b in cl["models"]
    for (a, b), (x, n) in pair.items():
        if n:
            W(f"   {a:>8} -> {b:<8} {x / n:.3f}  ({x}/{n})")
    nf = [int(float(df.at[i, "n_full_reference"] or 0)) for i in rows if done(df.at[i, "full_reference_json"])]
    W(f"\n FULL REFERENCE: mean {np.mean(nf) if nf else 0:.2f} items per paper (author weight 1.0 + pseudo weights)")
    W(f" gpt usage: {llm.stats()}")
    W("=" * 110)
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(os.path.dirname(out_csv), "reference_report.txt"), "w") as f:
        f.write(txt + "\n")


async def main_async(args):
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    weights = {int(k): float(v) for k, v in (x.split(":") for x in args.vote_weights.split(","))}
    df, stem = merge_generation_files(args, models)
    out_csv = args.output_csv or os.path.join(args.output_dir, f"{stem}__reference.csv")
    ensure_cols(df, REF_COLS)
    if os.path.exists(out_csv) and not args.overwrite:             # keep finished votes from a previous run
        old = read_csv(out_csv)
        if len(old) == len(df):
            for c in REF_COLS:
                if c in old.columns:
                    df[c] = old[c].astype(object)
            log(f"resuming votes from {out_csv}")
    rows = select_rows(df, args.start_row, args.n_rows)
    llm = ChatLLM(args.judge_model, backend=args.backend, max_concurrency=args.concurrency, seed=args.seed)
    todo = [i for i in rows if args.overwrite or not done(df.at[i, "full_reference_json"])]
    log(f"{len(rows)} rows selected, {len(todo)} need a vote; models {models}")
    t0 = time.time()
    for s in range(0, len(todo), args.save_every):
        batch = todo[s:s + args.save_every]
        await asyncio.gather(*[vote_row(llm, df, i, models, args, weights, random.Random(args.seed * 100003 + int(i)))
                               for i in batch])
        save_csv(df, out_csv)
        log(f"voted {s + len(batch)}/{len(todo)} -> {out_csv}  ({(time.time() - t0) / 60:.1f} min) {llm.stats()}")
    save_csv(df, out_csv)
    report(df, rows, models, args, out_csv, llm)
    return out_csv


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, "logs"), "build_reference")
    try:
        out_csv = asyncio.run(main_async(args))
    except FatalAPIError as e:
        raise SystemExit(f"OpenAI refused the request: {e}")
    if args.then_evaluate:
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.py"),
               "--reference_csv", out_csv, "--eval_type", args.eval_type, "--n_rows", str(args.n_rows),
               "--start_row", str(args.start_row), "--backend", args.backend, "--models", args.models,
               "--output_dir", args.output_dir]
        log("running " + " ".join(cmd))
        sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
