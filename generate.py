#!/usr/bin/env python
"""
STAGE 1a - run ONE open LLM (served by vLLM) over the papers:

  task 'pseudo'    : pseudo-reference candidates = ADDITIONAL limitations the authors did not mention
                     (the model sees the author list + public-health data, and is told not to repeat it)
  task 'generate'  : limitation generation under the ablation grid (the model never sees the author list;
                     it gets full_text_excl_lim only):
                       zs            zero-shot, paper only
                       zs_ph         zero-shot + public-health data (CDC/Census/CMS/HRSA)
                       rag           zero-shot + retrieved literature (rag_top3_json)
                       rag_ph        zero-shot + literature + public-health data
                       agents        11 specialist agents + evidence gate + leader merge
                       agents_ph     agents + public-health data
                       agents_rag    agents + literature
                       agents_rag_ph full PolicyGuard (agents + literature + public-health data)
                     every condition has the same output budget (--k limitations at most)
  task 'all'       : both

Output: <output_dir>/<input_stem>__gen_<model>.csv = the input csv + new columns, rewritten every
--save_every rows (resumable: finished cells are skipped on a rerun).

  new shared columns : n_author_limitations, author_limitation_units_json, ph_context, ph_provenance_json,
                       ph_disease_code, ph_states
  per model m        : pseudo_cand_<m>, pseudo_cand_<m>__json, pseudo_cand_<m>__meta
                       lim_<m>_<cond>, lim_<m>_<cond>__json, lim_<m>_<cond>__meta (+ __trace for agents)
  text columns use 'Limitation 1: ...\\nLimitation 2: ...'

  python generate.py --model qwen --task all --n_rows 10
  python generate.py --model gemma --task generate --conditions zs,agents_rag_ph --n_rows -1
  python generate.py --model qwen --backend mock --n_rows 3      # offline plumbing test
"""
import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medlim import config  # noqa: E402
from medlim import publichealth as PH  # noqa: E402
from medlim.io_utils import (setup_logging, log, read_csv, save_csv, select_rows, rag_passages,  # noqa: E402
                             ensure_cols, done)
from medlim.llm import ChatLLM, FatalAPIError  # noqa: E402
from medlim.methods import run_zero_shot, run_pseudo, run_agents  # noqa: E402
from medlim.prompts import PROMPT_VERSION  # noqa: E402
from medlim.units import author_units, format_limitations  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="short name: qwen | gemma | mistral (= vLLM served name)")
    ap.add_argument("--served_name", default=None, help="vLLM --served-model-name (default = --model)")
    ap.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--backend", default="openai", choices=["openai", "mock"])
    ap.add_argument("--input_csv", default=config.DEFAULT_INPUT_CSV)
    ap.add_argument("--output_dir", default=config.OUTPUT_DIR)
    ap.add_argument("--output_csv", default=None, help="default <output_dir>/<stem>__gen_<model>.csv")
    ap.add_argument("--task", default="all", choices=["pseudo", "generate", "all"])
    ap.add_argument("--conditions", default="all", help="'all' or comma list of " + ",".join(config.CONDITIONS))
    ap.add_argument("--n_rows", type=int, default=-1, help="-1 = all rows")
    ap.add_argument("--start_row", type=int, default=0)
    ap.add_argument("--k", type=int, default=8, help="output budget: max limitations per paper per condition")
    ap.add_argument("--pseudo_k", type=int, default=8, help="max additional limitations per model for the pseudo set")
    ap.add_argument("--pseudo_use_ph", type=int, default=1, help="1 = pseudo-reference prompt sees public-health data")
    ap.add_argument("--pseudo_use_rag", type=int, default=0, help="1 = pseudo-reference prompt sees retrieved papers")
    ap.add_argument("--paper_chars", type=int, default=16000)
    ap.add_argument("--rag_chars", type=int, default=2500, help="chars kept per retrieved paper")
    ap.add_argument("--ph_chars", type=int, default=2600)
    ap.add_argument("--max_tokens", type=int, default=1500)
    ap.add_argument("--concurrency", type=int, default=16, help="max simultaneous requests to the server")
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true", help="recompute cells that are already filled")
    ap.add_argument("--json_mode", type=int, default=0, help="1 = ask vLLM for guided JSON (response_format)")
    ap.add_argument("--disable_thinking", type=int, default=-1, help="-1 auto (on for qwen), 0/1")
    ap.add_argument("--gate", default="strict", choices=["strict", "off"], help="agents evidence-anchoring gate")
    ap.add_argument("--merger", default="self", choices=["self", "gpt"],
                    help="who merges the agents' critiques: the same open model (clean ablation) or gpt-4o-mini")
    ap.add_argument("--live_rag", action="store_true", help="retrieve with HybridRAG when rag_top3_json is missing")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def add_shared_columns(df, rows, args):
    ensure_cols(df, ["author_limitation_units_json", "n_author_limitations", "ph_context", "ph_provenance_json",
                     "ph_disease_code", "ph_states", "ph_notes"])
    for i in rows:
        if not done(df.at[i, "author_limitation_units_json"]):
            u = author_units(df.at[i, config.COL_AUTHOR_LIM] if config.COL_AUTHOR_LIM in df.columns else "")
            df.at[i, "author_limitation_units_json"] = json.dumps(u, ensure_ascii=False)
            df.at[i, "n_author_limitations"] = len(u)
        if df.at[i, "ph_provenance_json"] is None or not done(df.at[i, "ph_provenance_json"]):
            ctx = PH.build_context(str(df.at[i, config.COL_TITLE] or ""), str(df.at[i, config.COL_ABSTRACT] or "")
                                   if config.COL_ABSTRACT in df.columns else "",
                                   str(df.at[i, config.COL_TEXT] or ""), args.ph_chars)
            df.at[i, "ph_context"] = ctx["text"]
            df.at[i, "ph_provenance_json"] = json.dumps(ctx["provenance"], ensure_ascii=False, default=str)
            df.at[i, "ph_disease_code"] = ctx["disease_code"] or ""
            df.at[i, "ph_states"] = ";".join(ctx["states"])
            df.at[i, "ph_notes"] = json.dumps(ctx["notes"] + [ctx["setting"]], default=str)


def print_author_counts(df, rows):
    n = [len(json.loads(df.at[i, "author_limitation_units_json"])) for i in rows]
    log(f"author-mentioned limitations (limitations_section split on '\\n'): {sum(n)} units over {len(n)} papers; "
        f"mean {np.mean(n):.2f}, median {np.median(n):.0f}, min {min(n)}, max {max(n)}; "
        f"papers with 0 author limitations: {sum(1 for x in n if x == 0)}")
    for i, x in zip(rows, n):
        pid = df.at[i, config.COL_ID] if config.COL_ID in df.columns else i
        print(f"    row {i:>4}  {config.COL_ID}={pid}  author limitations: {x}")


async def main_async(args):
    model = args.model
    served = args.served_name or model
    think_off = (model == "qwen") if args.disable_thinking < 0 else bool(args.disable_thinking)
    llm = ChatLLM(served, base_url=args.base_url, backend=args.backend, max_concurrency=args.concurrency,
                  disable_thinking=think_off, json_mode=bool(args.json_mode), seed=args.seed)
    merger = llm
    if args.merger == "gpt":
        merger = ChatLLM(config.JUDGE_MODEL, backend=args.backend, max_concurrency=8)

    stem = os.path.splitext(os.path.basename(args.input_csv))[0]
    out_csv = args.output_csv or os.path.join(args.output_dir, f"{stem}__gen_{model}.csv")
    if os.path.exists(out_csv):
        df = read_csv(out_csv)
        log(f"resuming from {out_csv} ({len(df)} rows)")
    else:
        df = read_csv(args.input_csv)
        log(f"input {args.input_csv}: {len(df)} rows")
    for c in (config.COL_TITLE, config.COL_TEXT):
        if c not in df.columns:
            raise SystemExit(f"input has no column '{c}'")
    rows = select_rows(df, args.start_row, args.n_rows)
    log(f"rows {args.start_row}..{args.start_row + len(rows) - 1} ({len(rows)} papers), model={served} "
        f"@ {args.base_url if args.backend == 'openai' else 'MOCK'}, thinking_off={think_off}")
    log(f"public-health data inventory: {PH.data_inventory()}")

    conds = list(config.CONDITIONS) if args.conditions == "all" else [c.strip() for c in args.conditions.split(",")]
    bad = [c for c in conds if c not in config.CONDITIONS]
    if bad:
        raise SystemExit(f"unknown conditions {bad}; choose from {list(config.CONDITIONS)}")
    do_pseudo = args.task in ("pseudo", "all")
    do_gen = args.task in ("generate", "all")

    rag = None
    if args.live_rag and "rag_top3_json" not in df.columns:
        from medlim.rag_hybrid import HybridRAG
        rag = HybridRAG()

    add_shared_columns(df, rows, args)
    print_author_counts(df, rows)
    pcol = f"pseudo_cand_{model}"
    cols = []
    if do_pseudo:
        cols += [pcol, pcol + "__json", pcol + "__meta"]
    if do_gen:
        for c in conds:
            b = f"lim_{model}_{c}"
            cols += [b, b + "__json", b + "__meta"] + ([b + "__trace"] if config.CONDITIONS[c][0] == "agents" else [])
    ensure_cols(df, cols)
    save_csv(df, out_csv)

    async def do_row(i):
        title = str(df.at[i, config.COL_TITLE] or "")
        text = str(df.at[i, config.COL_TEXT] or "")
        if not text.strip() and config.COL_ABSTRACT in df.columns:
            text = str(df.at[i, config.COL_ABSTRACT] or "")
        text = text[:args.paper_chars]
        passages = rag_passages(df.loc[i], args.rag_chars)
        if not passages and rag is not None:
            passages = [p[:args.rag_chars] for p in await asyncio.to_thread(rag.retrieve, f"Title: {title}\n{text[:3000]}", 3)]
        ph_text = df.at[i, "ph_context"] if done(df.at[i, "ph_context"]) else ""
        ph_prov = json.loads(df.at[i, "ph_provenance_json"]) if done(df.at[i, "ph_provenance_json"]) else []
        tasks = {}
        if do_pseudo and (args.overwrite or not done(df.at[i, pcol])):
            auth = json.loads(df.at[i, "author_limitation_units_json"])
            tasks[pcol] = run_pseudo(llm, title, text, auth, args.pseudo_k, ph_text if args.pseudo_use_ph else "",
                                     passages if args.pseudo_use_rag else None, args.max_tokens)
        if do_gen:
            for c in conds:
                b = f"lim_{model}_{c}"
                if not args.overwrite and done(df.at[i, b]):
                    continue
                method, use_rag, use_ph = config.CONDITIONS[c]
                psg = passages if use_rag else None
                pht = ph_text if use_ph else ""
                if method == "zero_shot":
                    tasks[b] = run_zero_shot(llm, title, text, args.k, psg, pht, args.max_tokens)
                else:
                    tasks[b] = run_agents(llm, title, text, args.k, psg, pht, ph_prov if use_ph else [], merger,
                                          min(args.max_tokens, 1200), args.gate)
        if not tasks:
            return 0
        res = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for col, r in zip(tasks, res):
            if isinstance(r, FatalAPIError):
                raise r
            if isinstance(r, Exception):
                log(f"row {i} {col}: {type(r).__name__}: {r}")
                df.at[i, col + "__meta"] = json.dumps({"error": f"{type(r).__name__}: {r}"})
                continue
            items, meta, extra = r
            cond = col.split(f"lim_{model}_", 1)[1] if col.startswith("lim_") else None
            used_rag = config.CONDITIONS[cond][1] if cond else bool(args.pseudo_use_rag)
            meta.update({"rag_passages": len(passages) if used_rag else 0, "prompt_version": PROMPT_VERSION,
                         "model": served})
            # an empty list is a valid answer: store a marker so the cell counts as done
            df.at[i, col] = format_limitations(items) or "(none)"
            df.at[i, col + "__json"] = json.dumps(items, ensure_ascii=False, default=str)
            df.at[i, col + "__meta"] = json.dumps(meta, default=str)
            if col + "__trace" in df.columns:
                df.at[i, col + "__trace"] = extra
        return len(tasks)

    t0, n_done = time.time(), 0
    for s in range(0, len(rows), args.save_every):
        batch = rows[s:s + args.save_every]
        counts = await asyncio.gather(*[do_row(i) for i in batch])
        n_done += len(batch)
        save_csv(df, out_csv)
        el = time.time() - t0
        log(f"[{model}] {n_done}/{len(rows)} papers ({sum(counts)} cells this batch) saved -> {out_csv}  "
            f"{el / 60:.1f} min, eta {(len(rows) - n_done) * el / max(1, n_done) / 60:.1f} min  {llm.stats()}")
    write_summary(df, rows, model, conds if do_gen else [], do_pseudo, out_csv, llm)


def write_summary(df, rows, model, conds, do_pseudo, out_csv, llm):
    L = [f"GENERATION SUMMARY  model={model}  {time.strftime('%Y-%m-%d %H:%M')}  rows={len(rows)}",
         f"output: {out_csv}", f"llm: {llm.stats()}", ""]
    cols = ([f"pseudo_cand_{model}"] if do_pseudo else []) + [f"lim_{model}_{c}" for c in conds]
    L.append(f"{'column':<34} {'done':>5} {'mean#':>6} {'zero':>5}  parse modes / errors")
    for c in cols:
        metas = [json.loads(df.at[i, c + "__meta"]) for i in rows if done(df.at[i, c + "__meta"])]
        ns = [m.get("n", 0) for m in metas if "error" not in m]
        modes = Counter(m.get("parse_mode", "?") for m in metas)
        errs = sum(1 for m in metas if "error" in m)
        extra = ""
        if metas and "n_emitted" in metas[0]:
            em = np.mean([m.get("n_emitted", 0) for m in metas if "error" not in m] or [0])
            ga = np.mean([m.get("n_after_gate", 0) for m in metas if "error" not in m] or [0])
            fb = sum(1 for m in metas if m.get("gate_fallback"))
            extra = f" | agents: emitted {em:.1f}, after gate {ga:.1f}, gate-fallback rows {fb}"
        L.append(f"{c:<34} {len(metas):>5} {np.mean(ns) if ns else 0:>6.2f} {sum(1 for x in ns if x == 0):>5}  "
                 f"{dict(modes)} errors={errs}{extra}")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(os.path.dirname(out_csv), f"gen_summary_{model}.txt"), "w") as f:
        f.write(txt + "\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, "logs"), f"generate_{args.model}")
    try:
        asyncio.run(main_async(args))
    except FatalAPIError as e:
        raise SystemExit(f"API refused the request: {e}")


if __name__ == "__main__":
    main()
