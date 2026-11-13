#!/usr/bin/env python
"""
200-paper experiment on the regex dataset (pubmed_oa_regex_lim_10000.csv).

Stages (--stages, comma list; 'all' = extract,generate,evaluate)
  extract    [GPU, this model]  a) author-mentioned limitations from the FULL paper as atomic units (verbatim
             quote checked against the text)  b) pseudo-label candidates: additional limitations with societal
             impact the authors did not mention (model sees its own extracted list + public-health data)
  generate   [GPU, this model]  limitations with societal impact from full_text_excl_lim under the ablation grid
             zs / zs_ph / rag / rag_ph / agents / agents_ph / agents_rag / agents_rag_ph
             agents = 11 specialists run SEQUENTIALLY (each sees teammates' findings) + leader/master agent
  reference  [CPU, gpt-4o-mini, shared]  waits until qwen / gemma / mistral finished 'extract', then ONE job builds
             (a) consensus author limitations: gpt-4o-mini clusters the three extractions, votes = #models
             (b) pseudo labels: majority vote over the three pseudo lists, minus anything the authors stated
             (c) full reference = author (weight 1.0) + pseudo (0.7 if 3 votes, 0.5 if 2)
  evaluate   [CPU, gpt-4o-mini]  this model's generations vs the full reference:
             list judge (one call: which reference / generated items match) and pairwise judge (every pair Yes/No)
             -> coverage of reference (recall), coverage of generated (precision), F1; also 1:1 matching and a
             societal-impact rating; then rebuilds final_text.txt from every model evaluated so far
  combine    rebuild final_text.txt only

Modes:  --mode dry  no LLM at all: prepares the sample (200 random papers + local RAG + public-health context)
                    and writes dry_check.txt (GT split, lengths, token estimates, prompt previews)
        --mode run  the stages above

  python run_experiment.py --mode dry
  python run_experiment.py --model qwen --stages extract --n_rows 5 --run_name test5
  python run_experiment.py --model qwen --stages all --n_rows 5 --run_name test5
  python run_experiment.py --model gemma --stages extract,generate,evaluate
  python run_experiment.py --model none --stages reference,evaluate --eval_models qwen,gemma,mistral   # CPU job
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medlim import config  # noqa: E402
from medlim import prompts as P0  # noqa: E402
from medlim import prompts_exp as P  # noqa: E402
from medlim import publichealth as PH  # noqa: E402
from medlim import sync  # noqa: E402
from medlim.io_utils import setup_logging, log, save_csv, ensure_cols, done  # noqa: E402
from medlim.llm import ChatLLM, FatalAPIError  # noqa: E402
from medlim.exp_methods import run_extract, run_pseudo, run_zero_shot, run_agents_seq  # noqa: E402
from medlim.paper import extraction_view, generation_text  # noqa: E402
from medlim.units import author_units, format_limitations, extract_json  # noqa: E402

ALL_MODELS = ["qwen", "gemma", "mistral"]
POLL_S = int(os.environ.get("MEDLIM_POLL_S", "120"))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="run", choices=["run", "dry"])
    ap.add_argument("--model", default="none", help="qwen | gemma | mistral (= vLLM served name) | none")
    ap.add_argument("--served_name", default=None)
    ap.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--backend", default="openai", choices=["openai", "mock"])
    ap.add_argument("--stages", default="extract", help="extract,generate,evaluate | all | reference | combine")
    # data / columns
    ap.add_argument("--data_csv", default=os.path.join(config.PROJECT_DIR, "data/pubmed_oa_regex_lim_10000.csv"))
    ap.add_argument("--id_col", default="pmid")
    ap.add_argument("--title_col", default="title")
    ap.add_argument("--abstract_col", default="abstract")
    ap.add_argument("--fulltext_col", default="full_text", help="text the EXTRACTOR reads (contains the limitations)")
    ap.add_argument("--input_col", default="full_text_excl_lim", help="text the GENERATORS read")
    ap.add_argument("--gt_col", default="limitations_section", help="regex ground truth ('\\n'-separated)")
    # sample
    ap.add_argument("--n_sample", type=int, default=200, help="random papers drawn from --data_csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_quality", default="none", choices=["none", "low", "medium", "high"],
                    help="only sample papers whose regex lim_quality is at least this")
    ap.add_argument("--n_rows", type=int, default=-1, help="process the first N rows of the sample (-1 = all)")
    ap.add_argument("--start_row", type=int, default=0)
    ap.add_argument("--run_name", default=None, help="default sample<N>_seed<S>; use e.g. test5 for a smoke test")
    ap.add_argument("--out_root", default=os.path.join(config.PROJECT_DIR, "outputs_exp"))
    # generation
    ap.add_argument("--models", default="qwen,gemma,mistral", help="the extractors / voters for the reference")
    ap.add_argument("--conditions", default="all")
    ap.add_argument("--k", type=int, default=8, help="output budget per condition")
    ap.add_argument("--extract_k", type=int, default=15)
    ap.add_argument("--pseudo_k", type=int, default=8)
    ap.add_argument("--paper_chars", type=int, default=14000)
    ap.add_argument("--extract_chars", type=int, default=16000)
    ap.add_argument("--rag_chars", type=int, default=2000)
    ap.add_argument("--max_tokens", type=int, default=1800)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--agent_mode", default="sequential", choices=["sequential", "parallel"])
    ap.add_argument("--gate", default="strict", choices=["strict", "off"])
    ap.add_argument("--leader", default="self", choices=["self", "gpt"], help="leader agent: same open model or gpt-4o-mini")
    ap.add_argument("--json_mode", type=int, default=0)
    ap.add_argument("--rag_model", default=os.path.join(config.MODELS_DIR, "all-MiniLM-L6-v2"))
    # reference / evaluation
    ap.add_argument("--judge_model", default=config.JUDGE_MODEL)
    ap.add_argument("--judge_backend", default=None, help="default = --backend")
    ap.add_argument("--pseudo", type=int, default=1,
                    help="1 = also create pseudo labels (additional limitations); 0 = author limitations only")
    ap.add_argument("--gt_min_votes", type=int, default=3, help="author GT: extractors that must agree (3 = all)")
    ap.add_argument("--pseudo_min_votes", type=int, default=2)
    ap.add_argument("--vote_weights", default="3:0.7,2:0.5")
    ap.add_argument("--gt_source", default="consensus", choices=["consensus", "majority", "unanimous", "regex"],
                    help="author part of the reference at evaluation time (consensus = --gt_min_votes)")
    ap.add_argument("--eval_type", default="both", choices=["list", "pairwise", "both"])
    ap.add_argument("--impact_eval", type=int, default=1)
    ap.add_argument("--eval_models", default=None, help="models to evaluate (default: --model)")
    ap.add_argument("--baseline", default="zs")
    ap.add_argument("--workers", type=int, default=16, help="parallel gpt-4o-mini requests")
    ap.add_argument("--wait_minutes", type=int, default=1440, help="how long 'reference' waits for the other models")
    return ap.parse_args()


# ============================================================================ sample preparation
def prepare_sample(args, path):
    log(f"preparing sample of {args.n_sample} papers from {args.data_csv}")
    light_cols = [args.id_col, args.title_col, args.abstract_col, args.gt_col, "lim_quality"]
    head = pd.read_csv(args.data_csv, nrows=0).columns.tolist()
    miss = [c for c in (args.id_col, args.title_col, args.fulltext_col, args.input_col, args.gt_col) if c not in head]
    if miss:
        raise SystemExit(f"columns {miss} not in {args.data_csv}; available: {head}")
    light = pd.read_csv(args.data_csv, usecols=[c for c in light_cols if c in head], low_memory=False)
    light = light.drop_duplicates(subset=[args.id_col]).reset_index(drop=True)
    pool = light
    order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    if args.min_quality != "none" and "lim_quality" in light.columns:
        pool = light[light.lim_quality.map(order).fillna(0) >= order[args.min_quality]]
    n = min(args.n_sample, len(pool))
    ids = pool.sample(n, random_state=args.seed)[args.id_col].astype(str).tolist()
    idset = set(ids)
    parts = []
    for ch in pd.read_csv(args.data_csv, chunksize=1000, low_memory=False):
        parts.append(ch[ch[args.id_col].astype(str).isin(idset)])
    df = pd.concat(parts).drop_duplicates(subset=[args.id_col])
    df = df.set_index(df[args.id_col].astype(str)).loc[ids].reset_index(drop=True)   # keep the random order
    # local RAG over the other papers
    corpus = light[~light[args.id_col].astype(str).isin(idset)].reset_index(drop=True)
    docs = (corpus[args.title_col].fillna("").astype(str) + ". " + corpus[args.abstract_col].fillna("").astype(str)).tolist()
    queries = (df[args.title_col].fillna("").astype(str) + ". " + df[args.abstract_col].fillna("").astype(str)).tolist()
    from medlim.local_rag import hybrid_topk
    top, how = hybrid_topk(queries, docs, 3, args.rag_model if os.path.isdir(args.rag_model) else None, log=log)
    rag = []
    for hits in top:
        ps = []
        for i in hits:
            r = corpus.iloc[i]
            lim = " ".join(str(r.get(args.gt_col, "") or "").split("\n"))[:1200]
            ps.append(f"Title: {r[args.title_col]}\nAbstract: {str(r[args.abstract_col])[:1500]}\n"
                      f"Author-stated limitations: {lim or '(none found)'}")
        rag.append(json.dumps(ps, ensure_ascii=False))
    df["rag_top3_json"] = rag
    df["rag_method"] = how
    # public-health context (deterministic, computed once for all three models)
    ctxs = [PH.build_context(str(r[args.title_col] or ""), str(r.get(args.abstract_col, "") or ""),
                             str(r[args.input_col] or "")) for _, r in df.iterrows()]
    df["ph_context"] = [c["text"] for c in ctxs]
    df["ph_provenance_json"] = [json.dumps(c["provenance"], default=str) for c in ctxs]
    df["ph_disease_code"] = [c["disease_code"] or "" for c in ctxs]
    df["ph_states"] = [";".join(c["states"]) for c in ctxs]
    df["ph_notes"] = [json.dumps(c["notes"] + [c["setting"]], default=str) for c in ctxs]
    units = [author_units(t) for t in df[args.gt_col]]
    df["regex_units_json"] = [json.dumps(u, ensure_ascii=False) for u in units]
    df["n_regex_units"] = [len(u) for u in units]
    save_csv(df, path)
    log(f"sample saved: {path} ({len(df)} papers; retrieval {how}; public-health inventory {PH.data_inventory()})")


def load_sample(args, run_dir):
    path = os.path.join(run_dir, "sample.csv")
    if not os.path.exists(path):
        lock = path + ".lock"
        if sync.try_lock(lock):
            try:
                prepare_sample(args, path)
            finally:
                sync.release(lock)
        else:
            sync.wait_for(lambda: os.path.exists(path) and not os.path.exists(lock), "sample.csv (other job)", 120,
                          30, log, lock, 60)
    df = pd.read_csv(path, low_memory=False)
    # papers without a title: use the first sentence of the abstract so prompts never say 'nan'
    t = df[args.title_col] if args.title_col in df.columns else pd.Series([None] * len(df))
    miss = t.isna() | t.astype(str).str.strip().isin(["", "nan", "None"])
    if miss.any():
        df[args.title_col] = df[args.title_col].astype(object)   # an all-empty column is float64
        ab = df[args.abstract_col].fillna("").astype(str) if args.abstract_col in df.columns else pd.Series([""] * len(df))
        df.loc[miss, args.title_col] = ["(no title) " + (a.split(". ")[0][:200] if a.strip() else "untitled paper")
                                        for a in ab[miss]]
        log(f"{int(miss.sum())} papers have no title -> first sentence of the abstract used instead")
    if df[args.id_col].dtype.kind == "f":
        df[args.id_col] = df[args.id_col].map(lambda x: str(int(x)) if pd.notna(x) else "")
    return df, path


def select(df, args):
    end = None if args.n_rows < 0 else args.start_row + args.n_rows
    return list(df.index[args.start_row:end])


# ============================================================================ dry check
def dry_check(df, rows, args, run_dir):
    L = []
    W = L.append
    W("=" * 110)
    W(f" DRY CHECK (no LLM)   {time.strftime('%Y-%m-%d %H:%M')}")
    W(f" data   : {args.data_csv}")
    W(f" sample : {os.path.join(run_dir, 'sample.csv')}  ({len(df)} papers, seed {args.seed}); rows checked: {len(rows)}")
    W(f" columns: id={args.id_col} title={args.title_col} extractor input={args.fulltext_col} generator input="
      f"{args.input_col} regex GT={args.gt_col}")
    W(f" retrieval: {df['rag_method'].iloc[0] if 'rag_method' in df else '?'};  public-health inventory: {PH.data_inventory()}")
    W("=" * 110)
    n = np.array([int(df.at[i, "n_regex_units"]) for i in rows])
    W(f" regex author limitations (split on '\\n'): total {n.sum()}, mean {n.mean():.2f}, median {np.median(n):.0f}, "
      f"range {n.min()}-{n.max()}, papers with 0: {(n == 0).sum()}")
    W("   distribution: " + ", ".join(f"{k}:{v}" for k, v in sorted(Counter(n.tolist()).items())))
    tok = lambda s: len(s) / 4  # noqa: E731
    ex, gen, ag = [], [], []
    for i in rows:
        r = df.loc[i]
        v = extraction_view(r[args.fulltext_col], r.get("lim_start_char", -1), r.get("lim_end_char", -1), args.extract_chars)
        ex.append(tok(P.extract_user(r[args.title_col], v, args.extract_k)))
        txt = generation_text(r[args.input_col], args.paper_chars)
        ps = json.loads(r["rag_top3_json"]) if done(r.get("rag_top3_json")) else []
        gen.append(tok(P.zs_user(r[args.title_col], txt, args.k, [p[:args.rag_chars] for p in ps], r.get("ph_context") or "")))
        ag.append(tok(P.agent_user(r[args.title_col], txt, P.AGENTS[0][3], [], [p[:args.rag_chars] for p in ps],
                                   r.get("ph_context") or "")))
    W(f"\n PROMPT SIZE ESTIMATES (tokens ~ chars/4; keep below the vLLM --max-model-len minus ~1.8k output)")
    for name, x in (("extraction", ex), ("zero-shot rag_ph", gen), ("agent rag_ph", ag)):
        W(f"   {name:<18} mean {np.mean(x):,.0f}  max {np.max(x):,.0f}")
    calls = len(rows) * (2 + 4 + 4 * 12)
    W(f" LLM calls per model for these rows ~ {calls:,} (2 extract/pseudo + 4 zero-shot + 4 x (11 agents + leader))")
    W(f" public-health context present: {sum(1 for i in rows if done(df.at[i, 'ph_context']))}/{len(rows)}; "
      f"disease codes: {dict(Counter(df.loc[rows, 'ph_disease_code'].fillna('').tolist()).most_common(8))}")
    for i in rows[: min(len(rows), 100)]:
        r = df.loc[i]
        W(f"\n{'#' * 110}\n# ROW {i}  {args.id_col}={r[args.id_col]}  regex GT units: {r['n_regex_units']}  "
          f"quality={r.get('lim_quality', '?')}  method={r.get('lim_method', '?')}")
        W(f"# {str(r[args.title_col])[:140]}\n{'#' * 110}")
        for k, u in enumerate(json.loads(r["regex_units_json"])):
            W(f"   GT{k + 1}: {u}")
        W(f"   extractor input chars: {len(str(r[args.fulltext_col])):,} -> view {len(extraction_view(r[args.fulltext_col], r.get('lim_start_char', -1), r.get('lim_end_char', -1), args.extract_chars)):,}; "
          f"generator input chars: {len(str(r[args.input_col])):,}")
        W(f"   public-health: {(str(r.get('ph_context') or '(none)')[:400]).replace(chr(10), ' | ')}")
    r = df.loc[rows[0]]
    v = extraction_view(r[args.fulltext_col], r.get("lim_start_char", -1), r.get("lim_end_char", -1), args.extract_chars)
    W(f"\n{'=' * 110}\nPROMPT PREVIEW (row {rows[0]}), extraction (first 1500 chars of the paper part omitted):\n")
    pr = P.extract_user(r[args.title_col], v, args.extract_k)
    W(pr[:600] + "\n   [...]\n" + pr[-1600:])
    W(f"\n{'=' * 110}\nPROMPT PREVIEW, agent (last 2500 chars):\n")
    ps = json.loads(r["rag_top3_json"]) if done(r.get("rag_top3_json")) else []
    W(P.agent_user(r[args.title_col], generation_text(r[args.input_col], args.paper_chars), P.AGENTS[0][3], [],
                   [p[:args.rag_chars] for p in ps], r.get("ph_context") or "")[-2500:])
    path = os.path.join(run_dir, "dry_check.txt")
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:22]))
    log(f"dry check written: {path}")


# ============================================================================ GPU stages
def model_file(run_dir, m):
    return os.path.join(run_dir, f"model_{m}.csv")


async def gpu_stages(args, df_sample, rows, run_dir, stages):
    m = args.model
    path = model_file(run_dir, m)
    if os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
        if len(df) != len(df_sample):
            raise SystemExit(f"{path} has {len(df)} rows, sample has {len(df_sample)}; use another --run_name")
        log(f"resuming {path}")
    else:
        df = df_sample.copy()
    think_off = m == "qwen"
    llm = ChatLLM(args.served_name or m, base_url=args.base_url, backend=args.backend, max_concurrency=args.concurrency,
                  disable_thinking=think_off, json_mode=bool(args.json_mode), seed=args.seed)
    leader = ChatLLM(args.judge_model, backend=args.judge_backend or args.backend, max_concurrency=8) \
        if args.leader == "gpt" else llm
    conds = list(config.CONDITIONS) if args.conditions == "all" else [c.strip() for c in args.conditions.split(",")]
    cols = []
    if "extract" in stages:
        cols += [f"ext_{m}", f"ext_{m}__json", f"ext_{m}__meta", f"ext_{m}__trace"]
        if args.pseudo:
            cols += [f"pseudo_cand_{m}", f"pseudo_cand_{m}__json", f"pseudo_cand_{m}__meta"]
    if "generate" in stages:
        for c in conds:
            b = f"lim_{m}_{c}"
            cols += [b, b + "__json", b + "__meta"] + ([b + "__trace"] if config.CONDITIONS[c][0] == "agents" else [])
    ensure_cols(df, cols)

    def store(i, col, r):
        if isinstance(r, Exception):
            if isinstance(r, FatalAPIError):
                raise r
            df.at[i, col + "__meta"] = json.dumps({"error": f"{type(r).__name__}: {r}"})
            return
        items, meta, extra = r
        meta.update(model=args.served_name or m, prompt_version=P.PROMPT_VERSION)
        df.at[i, col] = format_limitations(items) or "(none)"
        df.at[i, col + "__json"] = json.dumps(items, ensure_ascii=False, default=str)
        df.at[i, col + "__meta"] = json.dumps(meta, default=str)
        if col + "__trace" in df.columns:
            df.at[i, col + "__trace"] = extra

    async def row_extract(i):
        r = df.loc[i]
        title = str(r[args.title_col] or "")
        if args.overwrite or not done(df.at[i, f"ext_{m}"]):
            view = extraction_view(r[args.fulltext_col], r.get("lim_start_char", -1), r.get("lim_end_char", -1),
                                   args.extract_chars)
            res = await asyncio.gather(run_extract(llm, title, view, args.extract_k, args.max_tokens),
                                       return_exceptions=True)
            store(i, f"ext_{m}", res[0])
        if args.pseudo and (args.overwrite or not done(df.at[i, f"pseudo_cand_{m}"])):
            own = [x["limitation"] for x in json.loads(df.at[i, f"ext_{m}__json"])] \
                if done(df.at[i, f"ext_{m}__json"]) else json.loads(r["regex_units_json"])
            text = generation_text(r[args.input_col], args.paper_chars)
            res = await asyncio.gather(run_pseudo(llm, title, text, own, args.pseudo_k, r.get("ph_context") or "",
                                                  args.max_tokens), return_exceptions=True)
            store(i, f"pseudo_cand_{m}", res[0])

    async def row_generate(i):
        r = df.loc[i]
        title, abstract = str(r[args.title_col] or ""), str(r.get(args.abstract_col, "") or "")
        text = generation_text(r[args.input_col], args.paper_chars)
        ps = [p[:args.rag_chars] for p in json.loads(r["rag_top3_json"])] if done(r.get("rag_top3_json")) else []
        ph_text = r.get("ph_context") if done(r.get("ph_context")) else ""
        ph_prov = json.loads(r["ph_provenance_json"]) if done(r.get("ph_provenance_json")) else []
        tasks = {}
        for c in conds:
            b = f"lim_{m}_{c}"
            if not args.overwrite and done(df.at[i, b]):
                continue
            method, use_rag, use_ph = config.CONDITIONS[c]
            if method == "zero_shot":
                tasks[b] = run_zero_shot(llm, title, text, args.k, ps if use_rag else None, ph_text if use_ph else "",
                                         args.max_tokens)
            else:
                tasks[b] = run_agents_seq(llm, title, abstract, text, args.k, ps if use_rag else None,
                                          ph_text if use_ph else "", ph_prov if use_ph else [], leader,
                                          min(args.max_tokens, 1000), args.gate, args.agent_mode == "parallel")
        res = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for col, r_ in zip(tasks, res):
            store(i, col, r_)

    for stage, fn in (("extract", row_extract), ("generate", row_generate)):
        if stage not in stages:
            continue
        t0 = time.time()
        for s in range(0, len(rows), args.save_every):
            batch = rows[s:s + args.save_every]
            await asyncio.gather(*[fn(i) for i in batch])
            save_csv(df, path)
            el = time.time() - t0
            log(f"[{m}:{stage}] {s + len(batch)}/{len(rows)} papers saved -> {path}  {el / 60:.1f} min, eta "
                f"{(len(rows) - s - len(batch)) * el / max(1, s + len(batch)) / 60:.1f} min  {llm.stats()}")
        if stage == "extract":
            n_ext = [len(json.loads(df.at[i, f"ext_{m}__json"])) for i in rows if done(df.at[i, f"ext_{m}__json"])]
            n_rx = [int(df.at[i, "n_regex_units"]) for i in rows]
            log(f"[{m}] author limitations: LLM-extracted mean {np.mean(n_ext) if n_ext else 0:.2f}/paper "
                f"(total {sum(n_ext)}), regex mean {np.mean(n_rx):.2f}/paper (total {sum(n_rx)})")
            for i in rows[:200]:
                print(f"    row {i:>4} {args.id_col}={df.at[i, args.id_col]}  regex units {df.at[i, 'n_regex_units']}"
                      f"  {m} extracted {len(json.loads(df.at[i, f'ext_{m}__json'])) if done(df.at[i, f'ext_{m}__json']) else 'ERR'}")
    return df


# ============================================================================ reference (shared, gpt-4o-mini)
def ref_path(run_dir):
    return os.path.join(run_dir, "reference.csv")


def extract_ready(run_dir, models, ids, need_pseudo=True):
    for m in models:
        p = model_file(run_dir, m)
        if not os.path.exists(p):
            return False
        d = pd.read_csv(p, usecols=lambda c: c in ("pmid", f"ext_{m}", f"pseudo_cand_{m}"), low_memory=False)
        if f"ext_{m}" not in d.columns or (need_pseudo and f"pseudo_cand_{m}" not in d.columns):
            return False
        if not all(done(d.at[i, f"ext_{m}"]) and (not need_pseudo or done(d.at[i, f"pseudo_cand_{m}"])) for i in ids):
            return False
    return True


def reference_ready(run_dir, ids):
    p = ref_path(run_dir)
    if not os.path.exists(p) or os.path.exists(p + ".lock"):
        return False
    d = pd.read_csv(p, usecols=lambda c: c == "full_reference_json", low_memory=False)
    return "full_reference_json" in d.columns and all(done(d.at[i, "full_reference_json"]) for i in ids)


async def cluster_call(llm, system, user, ids):
    raw = await llm.chat(system, user, max_tokens=2500, json_mode=True)
    obj = extract_json(raw)
    got = obj.get("clusters", []) if isinstance(obj, dict) else []
    clusters, seen = [], set()
    for cl in got if isinstance(got, list) else []:
        if not isinstance(cl, dict):
            continue
        mem = [x for x in cl.get("members", []) if isinstance(x, str) and x in ids and x not in seen]
        if not mem:
            continue
        seen.update(mem)
        clusters.append({"canonical": str(cl.get("canonical") or ids[mem[0]][1]).strip(), "members": mem,
                         "member_texts": [ids[x][1] for x in mem], "models": sorted({ids[x][0] for x in mem}),
                         "duplicates_author": bool(cl.get("duplicates_author", False))})
    for x in ids:
        if x not in seen:
            clusters.append({"canonical": ids[x][1], "members": [x], "member_texts": [ids[x][1]],
                             "models": [ids[x][0]], "duplicates_author": False, "orphan": True})
    for c in clusters:
        c["votes"] = len(c["models"])
    return clusters, bool(got)


async def build_reference(args, df_sample, rows, run_dir):
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    weights = {int(k): float(v) for k, v in (x.split(":") for x in args.vote_weights.split(","))}
    path = ref_path(run_dir)
    ref = pd.read_csv(path, low_memory=False) if os.path.exists(path) else \
        df_sample[[args.id_col, args.title_col, args.gt_col, "regex_units_json", "n_regex_units"]].copy()
    mdfs = {m: pd.read_csv(model_file(run_dir, m), low_memory=False) for m in models}
    for m, d in mdfs.items():
        for c in (f"ext_{m}", f"ext_{m}__json", f"pseudo_cand_{m}", f"pseudo_cand_{m}__json"):
            if c in d.columns:
                ref[c] = d[c].astype(object)
    out_cols = ["gt_clusters_json", "gt_author_unanimous_json", "gt_author_majority_json", "gt_author_json",
                "gt_author_text", "n_gt_author", "n_gt_unanimous", "n_gt_majority", "pseudo_clusters_json",
                "pseudo_reference_json", "n_pseudo", "full_reference_json", "full_reference_text", "n_full_reference",
                "reference_meta"]
    ensure_cols(ref, out_cols)
    judge = ChatLLM(args.judge_model, backend=args.judge_backend or args.backend, max_concurrency=args.workers,
                    seed=args.seed)
    gt_min = min(args.gt_min_votes, len(models))

    async def one(i):
        rng = random.Random(args.seed * 100003 + int(i))
        title = str(ref.at[i, args.title_col] or "")
        lists = {m: [x["limitation"] for x in json.loads(ref.at[i, f"ext_{m}__json"])]
                 if done(ref.at[i, f"ext_{m}__json"]) else [] for m in models}
        order = models[:]
        rng.shuffle(order)
        anon = {f"E{j + 1}": m for j, m in enumerate(order)}
        ids = {f"{a}_{k + 1}": (m, t) for a, m in anon.items() for k, t in enumerate(lists[m])}
        meta = {"anon_extract": anon}
        if sum(1 for m in models if lists[m]) >= 2:
            clusters, ok = await cluster_call(judge, P.CONSENSUS_SYSTEM, P.consensus_user(
                title, {a: lists[m] for a, m in anon.items()}, str(ref.at[i, args.gt_col] or "")), ids)
            meta["consensus_parsed"] = ok
        else:
            clusters = [{"canonical": t, "members": [x], "member_texts": [t], "models": [m], "votes": 1}
                        for x, (m, t) in ids.items()]
        unan = [c["canonical"] for c in clusters if c["votes"] >= len(models)]
        major = [c["canonical"] for c in clusters if c["votes"] >= 2]
        final = [c["canonical"] for c in clusters if c["votes"] >= gt_min]
        ref.at[i, "gt_clusters_json"] = json.dumps(clusters, ensure_ascii=False)
        ref.at[i, "gt_author_unanimous_json"] = json.dumps(unan, ensure_ascii=False)
        ref.at[i, "gt_author_majority_json"] = json.dumps(major, ensure_ascii=False)
        ref.at[i, "gt_author_json"] = json.dumps(final, ensure_ascii=False)
        ref.at[i, "gt_author_text"] = format_limitations(final) or "(none)"
        ref.at[i, "n_gt_author"], ref.at[i, "n_gt_unanimous"], ref.at[i, "n_gt_majority"] = len(final), len(unan), len(major)
        # ---- pseudo labels
        plists = {m: [x["limitation"] for x in json.loads(ref.at[i, f"pseudo_cand_{m}__json"])]
                  if args.pseudo and f"pseudo_cand_{m}__json" in ref.columns and done(ref.at[i, f"pseudo_cand_{m}__json"])
                  else [] for m in models}
        order = models[:]
        rng.shuffle(order)
        panon = {f"M{j + 1}": m for j, m in enumerate(order)}
        pids = {f"{a}_{k + 1}": (m, t) for a, m in panon.items() for k, t in enumerate(plists[m])}
        authors = list(dict.fromkeys(major + json.loads(ref.at[i, "regex_units_json"])))
        if sum(1 for m in models if plists[m]) >= 2:
            pcl, ok = await cluster_call(judge, P0.VOTE_SYSTEM, P0.vote_user(
                title, authors, {a: plists[m] for a, m in panon.items()}), pids)
            meta["pseudo_parsed"] = ok
        else:
            pcl = [{"canonical": t, "members": [x], "member_texts": [t], "models": [m], "votes": 1,
                    "duplicates_author": False} for x, (m, t) in pids.items()]
        pseudo = [{"text": c["canonical"], "votes": c["votes"], "weight": weights.get(min(c["votes"], 3), 0.0),
                   "models": c["models"]} for c in pcl
                  if c["votes"] >= args.pseudo_min_votes and not c.get("duplicates_author")
                  and weights.get(min(c["votes"], 3), 0.0) > 0]
        full = [{"id": f"A{k + 1}", "text": t, "origin": "author", "weight": 1.0} for k, t in enumerate(final)]
        full += [{"id": f"P{k + 1}", "text": p["text"], "origin": "pseudo", "weight": p["weight"], "votes": p["votes"]}
                 for k, p in enumerate(pseudo)]
        meta.update(anon_pseudo=panon, n_pseudo_dup_author=sum(1 for c in pcl if c.get("duplicates_author")))
        ref.at[i, "pseudo_clusters_json"] = json.dumps(pcl, ensure_ascii=False)
        ref.at[i, "pseudo_reference_json"] = json.dumps(pseudo, ensure_ascii=False)
        ref.at[i, "n_pseudo"] = len(pseudo)
        ref.at[i, "full_reference_json"] = json.dumps(full, ensure_ascii=False)
        ref.at[i, "full_reference_text"] = format_limitations([x["text"] for x in full]) or "(none)"
        ref.at[i, "n_full_reference"] = len(full)
        ref.at[i, "reference_meta"] = json.dumps(meta)

    todo = [i for i in rows if args.overwrite or not done(ref.at[i, "full_reference_json"])]
    log(f"[reference] {len(todo)} papers to build (gt_min_votes={gt_min}, pseudo_min_votes={args.pseudo_min_votes})")
    for s in range(0, len(todo), args.save_every):
        await asyncio.gather(*[one(i) for i in todo[s:s + args.save_every]])
        save_csv(ref, path + ".partial.csv")
        log(f"[reference] {s + len(todo[s:s + args.save_every])}/{len(todo)}  {judge.stats()}")
    save_csv(ref, path)
    if os.path.exists(path + ".partial.csv"):
        os.remove(path + ".partial.csv")
    na = [int(ref.at[i, "n_regex_units"]) for i in rows]
    ng = [int(float(ref.at[i, "n_gt_author"])) for i in rows]
    log(f"[reference] author limitations per paper: regex {np.mean(na):.2f}, LLM consensus (>= {gt_min} votes) "
        f"{np.mean(ng):.2f}; pseudo labels {np.mean([int(float(ref.at[i, 'n_pseudo'])) for i in rows]):.2f}")
    for i in rows[:200]:
        print(f"    row {i:>4} {args.id_col}={ref.at[i, args.id_col]}  regex {ref.at[i, 'n_regex_units']}  "
              + "  ".join(f"{m} {len(json.loads(ref.at[i, f'ext_{m}__json'])) if done(ref.at[i, f'ext_{m}__json']) else 0}"
                          for m in models)
              + f"  -> consensus {ref.at[i, 'n_gt_author']} (unanimous {ref.at[i, 'n_gt_unanimous']}, "
                f"majority {ref.at[i, 'n_gt_majority']})  pseudo {ref.at[i, 'n_pseudo']}")


async def reference_stage(args, df_sample, rows, run_dir):
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if reference_ready(run_dir, rows) and not args.overwrite:
        log("[reference] already built for these rows")
        return True
    lock = ref_path(run_dir) + ".lock"
    if sync.try_lock(lock):
        try:
            ok = sync.wait_for(lambda: extract_ready(run_dir, models, rows, bool(args.pseudo)), f"'extract' of {models}",
                               args.wait_minutes, POLL_S, log)
            if not ok:
                return False
            await build_reference(args, df_sample, rows, run_dir)
        finally:
            sync.release(lock)
        return True
    log("[reference] another job is building it; waiting")
    return sync.wait_for(lambda: reference_ready(run_dir, rows), "reference.csv", args.wait_minutes, POLL_S, log,
                         lock, stale_min=max(240, args.wait_minutes + 120))


# ============================================================================ main
async def main_async(args):
    run_name = args.run_name or f"sample{args.n_sample}_seed{args.seed}"
    run_dir = os.path.join(args.out_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    stages = {"all": ["extract", "generate", "evaluate"]}.get(args.stages, [s.strip() for s in args.stages.split(",")])
    df, sample_path = load_sample(args, run_dir)
    rows = select(df, args)
    log(f"run dir {run_dir}; sample {len(df)} papers; processing rows {rows[0] if rows else '-'}..{rows[-1] if rows else '-'} "
        f"({len(rows)}); stages {stages}; model {args.model}")
    if args.mode == "dry":
        dry_check(df, rows, args, run_dir)
        return
    if ("extract" in stages or "generate" in stages):
        if args.model not in ALL_MODELS and args.backend != "mock":
            raise SystemExit("--model qwen|gemma|mistral is required for extract / generate")
        await gpu_stages(args, df, rows, run_dir, stages)
        open(os.path.join(run_dir, f"gpu_done_{args.model}.flag"), "w").write(time.ctime())
    if "evaluate" in stages or "reference" in stages:
        ok = await reference_stage(args, df, rows, run_dir)
        if not ok:
            raise SystemExit("reference not available - run the other models' extract stage, then rerun with "
                             "--stages reference,evaluate")
    if "evaluate" in stages:
        from medlim import exp_eval
        models = [m.strip() for m in (args.eval_models or args.model).split(",") if m.strip() in ALL_MODELS]
        for m in models:
            await exp_eval.evaluate_model(args, m, rows, run_dir)
        exp_eval.combine(args, run_dir, rows)
    elif "combine" in stages:
        from medlim import exp_eval
        exp_eval.combine(args, run_dir, rows)


def main():
    args = parse_args()
    run_name = args.run_name or f"sample{args.n_sample}_seed{args.seed}"
    os.makedirs(os.path.join(args.out_root, run_name, "logs"), exist_ok=True)
    setup_logging(os.path.join(args.out_root, run_name, "logs"), f"exp_{args.model}_{args.mode}")
    try:
        asyncio.run(main_async(args))
    except FatalAPIError as e:
        raise SystemExit(f"API refused the request: {e}")


if __name__ == "__main__":
    main()