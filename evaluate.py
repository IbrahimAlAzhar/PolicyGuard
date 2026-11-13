#!/usr/bin/env python
"""
STAGE 2 - gpt-4o-mini as judge of generated limitations vs the FULL reference set
(author limitations, weight 1.0  +  majority-vote pseudo labels, weight 0.5 / 0.7).

Judging (--eval_type)
  list      one call per (paper, generated column): numbered reference list (author and pseudo items
            shuffled together, origin hidden) + numbered generated list -> JSON matches
  pairwise  your archived Yes/No prompt on every (reference item, generated item) pair
  both      run both and report them side by side (+ agreement between the two judges)

Scores per paper (matches M between reference items r and generated items g)
  weighted recall    = sum_{r matched} w_r / sum_r w_r                        (coverage of the reference)
  weighted precision = mean_g  max_{r matched to g} w_r   (1 = matches an author limitation,
                                                            0.7 / 0.5 = matches only a pseudo label, 0 = none)
  weighted F1        = harmonic mean
  author-only P/R/F1 (secondary, only papers with >= 1 author limitation), pseudo recall,
  novel_valid        = generated items that match a pseudo label but no author limitation
                       (issues the system found that the authors did not mention)

Outputs in --output_dir: df_eval<TAG>.csv, summary<TAG>.csv, final_text<TAG>.txt, judge_cache.jsonl

  python evaluate.py --eval_type list --n_rows 10
  python evaluate.py --eval_type both --n_rows -1 --models qwen,gemma,mistral
  python evaluate.py --eval_type list --backend mock --n_rows 3       # offline test
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medlim import config  # noqa: E402
from medlim import prompts as P  # noqa: E402
from medlim.io_utils import setup_logging, log, read_csv, save_csv, select_rows, done  # noqa: E402
from medlim.llm import ChatLLM, FatalAPIError  # noqa: E402
from medlim.units import extract_json, parse_formatted  # noqa: E402

JUDGE_VERSION = "judge_v2"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference_csv", default=None, help="default <output_dir>/<stem>__reference.csv")
    ap.add_argument("--input_csv", default=config.DEFAULT_INPUT_CSV, help="only used to derive the default stem")
    ap.add_argument("--output_dir", default=config.OUTPUT_DIR)
    ap.add_argument("--eval_dir", default=None, help="default <output_dir>/eval")
    ap.add_argument("--models", default="qwen,gemma,mistral")
    ap.add_argument("--conditions", default="all")
    ap.add_argument("--columns", default=None, help="explicit comma list of generated columns (overrides models x conditions)")
    ap.add_argument("--n_rows", type=int, default=-1)
    ap.add_argument("--start_row", type=int, default=0)
    ap.add_argument("--eval_type", default="list", choices=["list", "pairwise", "both"])
    ap.add_argument("--judge_model", default=config.JUDGE_MODEL)
    ap.add_argument("--backend", default="openai", choices=["openai", "mock"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--baseline", default="zs", help="condition every other condition is compared with")
    ap.add_argument("--pseudo_weight_override", default=None, help="e.g. '3:0.7,2:0.5' to re-weight without re-voting")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# ----------------------------------------------------------------------------- cached judge
class Cache:
    def __init__(self, path):
        self.path, self.d, self.lock = path, {}, threading.Lock()
        if os.path.exists(path):
            for line in open(path):
                try:
                    r = json.loads(line)
                    self.d[r["k"]] = r["a"]
                except Exception:
                    continue
        self.fh = open(path, "a", buffering=1)

    def get(self, k):
        return self.d.get(k)

    def put(self, k, a, extra=None):
        with self.lock:
            self.d[k] = a
            self.fh.write(json.dumps({"k": k, "a": a, **(extra or {})}, ensure_ascii=False) + "\n")


def key(*parts):
    return hashlib.sha1("\x00".join(map(str, parts)).encode()).hexdigest()


def is_yes(a):
    m = re.match(r"\W*(yes|no)\b", str(a), re.I)
    return m.group(1).lower() == "yes" if m else "YES" in str(a).upper()


async def judge_list(llm, cache, refs, gens, seed):
    """-> set of (ref_idx, gen_idx). References are shuffled so the judge cannot tell author from pseudo."""
    if not refs or not gens:
        return set(), 0
    order = list(range(len(refs)))
    random.Random(seed).shuffle(order)
    shown = [refs[j] for j in order]
    user = P.judge_list_user(shown, gens)
    k = key(JUDGE_VERSION, "list", llm.model, user)
    a = cache.get(k)
    if a is None:
        a = await llm.chat(P.JUDGE_LIST_SYSTEM, user, max_tokens=60 + 25 * len(gens), json_mode=True)
        if not a:
            return set(), 1
        cache.put(k, a)
    obj = extract_json(a)
    out = set()
    for m in (obj.get("matches", []) if isinstance(obj, dict) else []):
        if not isinstance(m, dict):
            continue
        g = re.fullmatch(r"G(\d+)", str(m.get("g", "")).strip())
        if not g or not (0 < int(g.group(1)) <= len(gens)):
            continue
        for r in m.get("r", []) or []:
            rr = re.fullmatch(r"R(\d+)", str(r).strip())
            if rr and 0 < int(rr.group(1)) <= len(refs):
                out.add((order[int(rr.group(1)) - 1], int(g.group(1)) - 1))
    return out, 0


async def judge_pairs(llm, cache, refs, gens):
    todo, out, err = [], set(), 0
    for ri, r in enumerate(refs):
        for gi, g in enumerate(gens):
            prompt = P.judge_pair_prompt(r, g)
            k = key(JUDGE_VERSION, "pair", llm.model, prompt)
            a = cache.get(k)
            if a is None:
                todo.append((ri, gi, k, prompt))
            elif is_yes(a):
                out.add((ri, gi))
    if todo:
        ans = await asyncio.gather(*[llm.chat("", p, max_tokens=5) for *_, p in todo])
        for (ri, gi, k, p), a in zip(todo, ans):
            if not a:
                err += 1
                continue
            cache.put(k, a)
            if is_yes(a):
                out.add((ri, gi))
    return out, err


# ----------------------------------------------------------------------------- scoring
def score(ref, gens, M):
    w = np.array([x["weight"] for x in ref], float)
    is_auth = np.array([x["origin"] == "author" for x in ref])
    nG = len(gens)
    r_hit = np.zeros(len(ref), bool)
    g_best = np.zeros(nG)
    g_auth = np.zeros(nG, bool)
    g_pseudo = np.zeros(nG, bool)
    for r, g in M:
        r_hit[r] = True
        g_best[g] = max(g_best[g], w[r])
        if is_auth[r]:
            g_auth[g] = True
        else:
            g_pseudo[g] = True
    f = lambda p, r: 2 * p * r / (p + r) if p + r > 0 else 0.0  # noqa: E731
    out = {"n_gen": nG, "n_ref": len(ref), "n_author": int(is_auth.sum()), "n_pseudo": int((~is_auth).sum()),
           "n_matched_pairs": len(M)}
    out["w_recall"] = float((w * r_hit).sum() / w.sum()) if w.sum() > 0 else np.nan
    out["w_precision"] = float(g_best.mean()) if nG else 0.0
    out["w_f1"] = f(out["w_precision"], out["w_recall"]) if not np.isnan(out["w_recall"]) else np.nan
    # flat (every reference item weight 1)
    out["flat_recall"] = float(r_hit.mean()) if len(ref) else np.nan
    out["flat_precision"] = float((g_auth | g_pseudo).mean()) if nG else 0.0
    out["flat_f1"] = f(out["flat_precision"], out["flat_recall"]) if len(ref) else np.nan
    if is_auth.any():
        out["auth_recall"] = float(r_hit[is_auth].mean())
        out["auth_precision"] = float(g_auth.mean()) if nG else 0.0
        out["auth_f1"] = f(out["auth_precision"], out["auth_recall"])
    else:
        out["auth_recall"] = out["auth_precision"] = out["auth_f1"] = np.nan
    out["pseudo_recall"] = float(r_hit[~is_auth].mean()) if (~is_auth).any() else np.nan
    out["novel_valid"] = int((g_pseudo & ~g_auth).sum())
    out["matched_author"] = int(g_auth.sum())
    out["unmatched"] = int((~(g_auth | g_pseudo)).sum())
    return out


# ----------------------------------------------------------------------------- stats
def boot_ci(x, n=2000, seed=0):
    x = np.asarray([v for v in x if not (isinstance(v, float) and math.isnan(v))], float)
    if len(x) < 2:
        return (np.nan, np.nan)
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_p(a, b, n=5000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if len(d) < 3 or np.allclose(d, 0):
        return 1.0
    sims = np.abs((d[None, :] * np.random.default_rng(seed).choice([-1, 1], (n, len(d)))).mean(1))
    return float((1 + (sims >= abs(d.mean()) - 1e-12).sum()) / (n + 1))


def nanmean(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(x)) if x else float("nan")


# ----------------------------------------------------------------------------- NLP similarity of matched pairs
TOK = re.compile(r"(?u)\b\w\w+\b")


def tfidf_cos(a, b):
    from collections import Counter as C
    ta, tb = TOK.findall(a.lower()), TOK.findall(b.lower())
    sa, sb = set(ta), set(tb)
    if not sa or not sb:
        return 0.0
    idf = {w: math.log(3 / (1 + (w in sa) + (w in sb))) + 1 for w in sa | sb}
    va = {w: c * idf[w] for w, c in C(ta).items()}
    vb = {w: c * idf[w] for w, c in C(tb).items()}
    na, nb = math.sqrt(sum(v * v for v in va.values())), math.sqrt(sum(v * v for v in vb.values()))
    return sum(va[w] * vb.get(w, 0) for w in va) / (na * nb)


def rouge_l(a, b):
    r, c = re.findall(r"[a-z0-9]+", a.lower()), re.findall(r"[a-z0-9]+", b.lower())
    if not r or not c:
        return 0.0
    prev = [0] * (len(c) + 1)
    for x in r:
        cur = [0]
        for j, y in enumerate(c):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if not lcs:
        return 0.0
    p, rr = lcs / len(c), lcs / len(r)
    return 2 * p * rr / (p + rr)


# ----------------------------------------------------------------------------- main
def load_reference(df, i, override):
    ref = json.loads(df.at[i, "full_reference_json"])
    if override:
        for x in ref:
            if x["origin"] == "pseudo":
                x["weight"] = override.get(int(x["votes"]), x["weight"])
        ref = [x for x in ref if x["weight"] > 0]
    return ref


def cond_of(col):
    m = re.match(r"lim_([a-z0-9]+)_(.+)$", col)
    return (m.group(1), m.group(2)) if m else ("?", col)


async def main_async(args):
    stem = os.path.splitext(os.path.basename(args.input_csv))[0]
    ref_csv = args.reference_csv or os.path.join(args.output_dir, f"{stem}__reference.csv")
    eval_dir = args.eval_dir or os.path.join(args.output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    df = read_csv(ref_csv)
    log(f"reference table {ref_csv}: {len(df)} rows")
    if "full_reference_json" not in df.columns:
        raise SystemExit("no full_reference_json column - run build_reference.py first")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    conds = list(config.CONDITIONS) if args.conditions == "all" else [c.strip() for c in args.conditions.split(",")]
    cols = ([c.strip() for c in args.columns.split(",")] if args.columns else
            [f"lim_{m}_{c}" for m in models for c in conds if f"lim_{m}_{c}" in df.columns])
    if not cols:
        raise SystemExit("no generated columns found")
    override = ({int(k): float(v) for k, v in (x.split(":") for x in args.pseudo_weight_override.split(","))}
                if args.pseudo_weight_override else None)
    rows = [i for i in select_rows(df, args.start_row, args.n_rows) if done(df.at[i, "full_reference_json"])]
    log(f"{len(rows)} rows with a reference x {len(cols)} columns; eval_type={args.eval_type}")
    types = ["list", "pairwise"] if args.eval_type == "both" else [args.eval_type]
    llm = ChatLLM(args.judge_model, backend=args.backend, max_concurrency=args.workers, seed=args.seed)
    cache = Cache(os.path.join(eval_dir, "judge_cache.jsonl"))

    res = {t: {c: {} for c in cols} for t in types}       # res[type][col][row] = metrics
    matches = {t: {c: {} for c in cols} for t in types}
    errors = Counter()
    t0 = time.time()
    jobs = []
    for i in rows:
        ref = load_reference(df, i, override)
        refs = [x["text"] for x in ref]
        for c in cols:
            if not done(df.at[i, c]):
                continue
            gens = [] if str(df.at[i, c]).strip() == "(none)" else parse_formatted(df.at[i, c])
            for t in types:
                jobs.append((t, c, i, ref, refs, gens))

    async def run(job):
        t, c, i, ref, refs, gens = job
        if t == "list":
            M, e = await judge_list(llm, cache, refs, gens, seed=args.seed * 7919 + int(i))
        else:
            M, e = await judge_pairs(llm, cache, refs, gens)
        errors[t] += e
        if ref:
            res[t][c][i] = score(ref, gens, M)
            matches[t][c][i] = sorted(M)
        return 1

    B = 200
    for s in range(0, len(jobs), B):
        await asyncio.gather(*[run(j) for j in jobs[s:s + B]])
        log(f"judged {min(s + B, len(jobs))}/{len(jobs)} (paper, column, type) jobs  "
            f"{(time.time() - t0) / 60:.1f} min  {llm.stats()}")

    # per-row columns
    new = {}
    for t in types:
        for c in cols:
            for kname in ("w_precision", "w_recall", "w_f1", "auth_f1", "novel_valid", "n_gen"):
                new[f"{c}__{t}__{kname}"] = [res[t][c].get(i, {}).get(kname, np.nan) for i in df.index]
            new[f"{c}__{t}__matches"] = [json.dumps(matches[t][c].get(i, [])) if i in matches[t][c] else None
                                         for i in df.index]
    out = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    tag = args.tag
    save_csv(out, os.path.join(eval_dir, f"df_eval{tag}.csv"))
    write_final(args, df, rows, cols, types, res, matches, errors, llm, eval_dir, tag, ref_csv, override)


def write_final(args, df, rows, cols, types, res, matches, errors, llm, eval_dir, tag, ref_csv, override):
    L = []
    W = L.append

    def f3(x):
        return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"

    def table(headers, data):
        w = [max(len(str(h)), *(len(str(r[k])) for r in data)) if data else len(str(h)) for k, h in enumerate(headers)]
        line = "+" + "+".join("-" * (x + 2) for x in w) + "+"
        o = [line, "| " + " | ".join(str(h).ljust(w[k]) for k, h in enumerate(headers)) + " |", line.replace("-", "=")]
        for r in data:
            o.append("| " + " | ".join(str(x).ljust(w[k]) if k == 0 else str(x).rjust(w[k]) for k, x in enumerate(r)) + " |")
        o.append(line)
        return "\n".join(o)

    W("=" * 120)
    W(f" MED_LIMAGENTS v2 - FINAL RESULTS    {time.strftime('%Y-%m-%d %H:%M')}")
    W("=" * 120)
    W(f" reference table : {ref_csv}")
    W(f" judge           : {args.judge_model} ({args.backend}); eval types: {types}; temperature 0; cache {eval_dir}/judge_cache.jsonl")
    W(f" rows            : {len(rows)} papers with a reference set; columns: {len(cols)}")
    W(f" scoring         : match to an AUTHOR limitation = 1.0; match to a PSEUDO label = its weight "
      f"(3 votes 0.7, 2 votes 0.5{'; override ' + str(override) if override else ''}); no match = 0")
    W("                   wP = mean best weight per generated item; wR = matched reference weight / total weight")
    W("                   author-only P/R/F1 uses only papers with >=1 author limitation (secondary measure)")
    W("                   novel = generated items matching a pseudo label but no author limitation (per paper)")
    W(f" failed judge calls: {dict(errors)} (counted as no match; rerun to retry - answers are cached)")

    # ---------- A. dataset / reference
    na = np.array([len(json.loads(df.at[i, "author_limitation_units_json"])) for i in rows])
    npl = np.array([int(float(df.at[i, "n_pseudo_reference"] or 0)) for i in rows])
    v3 = np.array([int(float(df.at[i, "n_pseudo_votes3"] or 0)) for i in rows])
    W("\nA. REFERENCE SET")
    W(f"   author-mentioned limitations (split on '\\n'): total {na.sum()}, mean {na.mean():.2f}/paper, median "
      f"{np.median(na):.0f}, range {na.min()}-{na.max()}; papers with none: {(na == 0).sum()}")
    W("   distribution (#limitations:#papers): " + ", ".join(f"{k}:{v}" for k, v in sorted(Counter(na.tolist()).items())))
    W(f"   pseudo labels: total {npl.sum()} ({v3.sum()} with 3 votes, {npl.sum() - v3.sum()} with 2 votes), mean "
      f"{npl.mean():.2f}/paper; papers with none: {(npl == 0).sum()}")
    W(f"   papers with no author limitations but >=1 pseudo label (now evaluable): {((na == 0) & (npl > 0)).sum()}")

    summ = []
    for t in types:
        W(f"\nB[{t}]. MAIN RESULTS PER MODEL x CONDITION   (mean over papers; wF1 95% bootstrap CI)")
        data = []
        for c in cols:
            R = res[t][c]
            if not R:
                continue
            m, cond = cond_of(c)
            v = lambda k: [R[i][k] for i in R]  # noqa: E731
            lo, hi = boot_ci(v("w_f1"))
            d = {"eval_type": t, "column": c, "model": m, "condition": cond, "n_papers": len(R),
                 "n_gen": nanmean(v("n_gen")), "w_precision": nanmean(v("w_precision")), "w_recall": nanmean(v("w_recall")),
                 "w_f1": nanmean(v("w_f1")), "w_f1_lo": lo, "w_f1_hi": hi,
                 "flat_precision": nanmean(v("flat_precision")), "flat_recall": nanmean(v("flat_recall")),
                 "flat_f1": nanmean(v("flat_f1")), "auth_precision": nanmean(v("auth_precision")),
                 "auth_recall": nanmean(v("auth_recall")), "auth_f1": nanmean(v("auth_f1")),
                 "n_auth_papers": sum(1 for i in R if not math.isnan(R[i]["auth_f1"])),
                 "pseudo_recall": nanmean(v("pseudo_recall")), "novel_valid": nanmean(v("novel_valid")),
                 "novel_valid_total": int(np.nansum(v("novel_valid"))), "unmatched": nanmean(v("unmatched"))}
            summ.append(d)
            data.append([f"{m}/{cond}", d["n_papers"], f"{d['n_gen']:.1f}", f3(d["w_precision"]), f3(d["w_recall"]),
                         f3(d["w_f1"]), f"[{f3(lo)},{f3(hi)}]", f"{f3(d['auth_precision'])}/{f3(d['auth_recall'])}/"
                         f"{f3(d['auth_f1'])} ({d['n_auth_papers']})", f3(d["pseudo_recall"]),
                         f"{d['novel_valid']:.2f} ({d['novel_valid_total']})", f"{d['unmatched']:.2f}"])
        W(table(["model/condition", "papers", "#gen", "wP", "wR", "wF1", "wF1 95% CI", "author P/R/F1 (papers)",
                 "pseudo R", "novel/paper (total)", "unmatched/paper"], data))

        # ---------- C. ablations
        W(f"\nC[{t}]. ABLATIONS (paired over the same papers; d = after - before; * p < 0.05)")
        pairs = [("+public-health data", "zs", "zs_ph"), ("+public-health data", "rag", "rag_ph"),
                 ("+public-health data", "agents", "agents_ph"), ("+public-health data", "agents_rag", "agents_rag_ph"),
                 ("+literature RAG", "zs", "rag"), ("+literature RAG", "zs_ph", "rag_ph"),
                 ("+literature RAG", "agents", "agents_rag"), ("+literature RAG", "agents_ph", "agents_rag_ph"),
                 ("agents vs single model", "zs", "agents"), ("agents vs single model", "rag", "agents_rag"),
                 ("agents vs single model", "zs_ph", "agents_ph"), ("agents vs single model", "rag_ph", "agents_rag_ph"),
                 ("full system vs zero-shot", "zs", "agents_rag_ph")]
        models = sorted({cond_of(c)[0] for c in cols})
        data = []
        for m in models:
            for lab, a, b in pairs:
                ca, cb = f"lim_{m}_{a}", f"lim_{m}_{b}"
                if ca not in res[t] or cb not in res[t]:
                    continue
                common = sorted(set(res[t][ca]) & set(res[t][cb]))
                if len(common) < 2:
                    continue
                cells = [f"{m}", lab, f"{a} -> {b}", len(common)]
                for k in ("w_f1", "auth_f1", "novel_valid"):
                    x = [res[t][ca][i][k] for i in common]
                    y = [res[t][cb][i][k] for i in common]
                    dd = nanmean(np.array(y, float) - np.array(x, float))
                    p = paired_p(y, x)
                    cells.append(f"{dd:+.3f}{'*' if p < 0.05 else ''} (p={p:.2g})")
                data.append(cells)
        if data:
            W(table(["model", "effect", "conditions", "papers", "d wF1", "d author F1", "d novel/paper"], data))
        else:
            W("   (need at least two conditions of the same model)")

        # ---------- D. average over models
        W(f"\nD[{t}]. CONDITION AVERAGED OVER MODELS")
        S = pd.DataFrame([d for d in summ if d["eval_type"] == t])
        if len(S):
            g = S.groupby("condition", sort=False)[["w_precision", "w_recall", "w_f1", "auth_f1", "pseudo_recall",
                                                    "novel_valid", "n_gen"]].mean()
            g = g.reindex([c for c in config.CONDITIONS if c in g.index] + [c for c in g.index if c not in config.CONDITIONS])
            W(table(["condition", "wP", "wR", "wF1", "author F1", "pseudo R", "novel/paper", "#gen"],
                    [[k, f3(r.w_precision), f3(r.w_recall), f3(r.w_f1), f3(r.auth_f1), f3(r.pseudo_recall),
                      f"{r.novel_valid:.2f}", f"{r.n_gen:.1f}"] for k, r in g.iterrows()]))

        # ---------- E. papers without author limitations
        no_auth = [i for i in rows if len(json.loads(df.at[i, "author_limitation_units_json"])) == 0]
        if no_auth:
            W(f"\nE[{t}]. PAPERS WITHOUT AUTHOR LIMITATIONS ({len(no_auth)}; scored against pseudo labels only)")
            data = []
            for c in cols:
                R = {i: res[t][c][i] for i in no_auth if i in res[t][c]}
                if R:
                    data.append([c, len(R), f3(nanmean([r["w_precision"] for r in R.values()])),
                                 f3(nanmean([r["w_recall"] for r in R.values()])),
                                 f3(nanmean([r["w_f1"] for r in R.values()]))])
            W(table(["column", "papers", "wP", "wR", "wF1"], data))

        # ---------- H. text similarity of matched pairs
        W(f"\nH[{t}]. TEXT SIMILARITY OF MATCHED PAIRS (TF-IDF cosine / ROUGE-L F1, mean over matched pairs)")
        data = []
        for c in cols:
            sims_c, sims_r = [], []
            for i, M in matches[t][c].items():
                ref = json.loads(df.at[i, "full_reference_json"])
                gens = parse_formatted(df.at[i, c]) if str(df.at[i, c]).strip() != "(none)" else []
                for r, g in M:
                    if r < len(ref) and g < len(gens):
                        sims_c.append(tfidf_cos(ref[r]["text"], gens[g]))
                        sims_r.append(rouge_l(ref[r]["text"], gens[g]))
            data.append([c, len(sims_c), f3(nanmean(sims_c)), f3(nanmean(sims_r))])
        W(table(["column", "matched pairs", "tfidf cos", "ROUGE-L"], data))

    if len(types) == 2:
        W("\nG. AGREEMENT BETWEEN LIST AND PAIRWISE JUDGES (same papers/columns)")
        agree, tot, jac = 0, 0, []
        for c in cols:
            for i in set(matches["list"][c]) & set(matches["pairwise"][c]):
                a, b = set(map(tuple, matches["list"][c][i])), set(map(tuple, matches["pairwise"][c][i]))
                if a | b:
                    jac.append(len(a & b) / len(a | b))
                ga = np.array([res["list"][c][i]["w_f1"], res["pairwise"][c][i]["w_f1"]], float)
                tot += 1
                agree += abs(ga[0] - ga[1]) < 0.1 if not np.isnan(ga).any() else 0
        W(f"   match-set Jaccard {nanmean(jac):.3f} over {len(jac)} (paper, column) cells; |wF1 difference| < 0.1 "
          f"in {agree}/{tot}")

    # ---------- F. generation diagnostics
    W("\nF. GENERATION DIAGNOSTICS (from the __meta columns)")
    data = []
    for c in cols:
        metas = [json.loads(df.at[i, c + "__meta"]) for i in rows if c + "__meta" in df.columns and done(df.at[i, c + "__meta"])]
        if not metas:
            continue
        modes = Counter(m.get("parse_mode", "?") for m in metas)
        errs = sum(1 for m in metas if "error" in m)
        ag = ""
        if any("n_emitted" in m for m in metas):
            ag = (f"emitted {nanmean([m.get('n_emitted') for m in metas]):.1f} -> gate "
                  f"{nanmean([m.get('n_after_gate') for m in metas]):.1f} -> leader "
                  f"{nanmean([m.get('n_after_leader') for m in metas]):.1f}; gate-fallback "
                  f"{sum(1 for m in metas if m.get('gate_fallback'))}")
        data.append([c, len(metas), f"{nanmean([m.get('n', 0) for m in metas]):.1f}",
                     " ".join(f"{k}:{v}" for k, v in modes.items()), errs, ag])
    if data:
        W(table(["column", "papers", "#items", "parse modes", "errors", "agents pipeline"], data))
    W("\nNOTES")
    W(" - Pseudo labels come from the same three open models that are evaluated, prompted differently (they see the")
    W("   author list and are asked for ADDITIONAL limitations, with public-health data). Conditions that use")
    W("   public-health data may therefore be favoured on pseudo matches; compare with the author-only columns,")
    W("   which do not depend on the pseudo labels.")
    W(" - Author-limitation matching is kept as a secondary measure; 'novel/paper' counts valid-looking issues")
    W("   beyond the authors' list. Expert validation of a sample (make_review_sample.py) is still needed to")
    W("   confirm that pseudo matches are scientifically valid.")
    W(f" - judge usage: {llm.stats()}")
    W("=" * 120)
    txt = "\n".join(L) + "\n"
    print(txt)
    pd.DataFrame(summ).to_csv(os.path.join(eval_dir, f"summary{tag}.csv"), index=False)
    for p in (os.path.join(eval_dir, f"final_text{tag}.txt"), os.path.join(args.output_dir, f"final_text{tag}.txt")):
        with open(p, "w") as f:
            f.write(txt)
    log(f"wrote {args.output_dir}/final_text{tag}.txt, {eval_dir}/summary{tag}.csv, df_eval{tag}.csv")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, "logs"), "evaluate")
    try:
        asyncio.run(main_async(args))
    except FatalAPIError as e:
        raise SystemExit(f"OpenAI refused the request: {e}")


if __name__ == "__main__":
    main()
