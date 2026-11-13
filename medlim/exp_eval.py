"""
Evaluation for the experiment.

Reference R of a paper = author limitations (LLM consensus, weight 1.0) + pseudo labels (0.7 / 0.5).
Generated G = the condition's final limitation units.

(a) list judge     : ONE gpt-4o-mini call per (paper, condition): numbered R (author/pseudo shuffled, origin hidden)
                     and numbered G -> which R each G matches.
(b) pairwise judge : EVERY (r, g) pair sent separately with the archived Yes/No prompt.
From the match matrix M:
    any-match      recall    = sum w(r matched) / sum w(r)        (coverage of the reference)
                   precision = mean_g max w(r matched to g)        (coverage of generated; 1 author, .7/.5 pseudo)
    1:1 matching   maximum one-to-one assignment on M (author items first): every r and g used once
                   recall = sum w(matched r)/sum w ; precision = sum w(matched r)/|G|
    author-only P/R/F1 (secondary), pseudo recall, novel = G matching a pseudo label but no author limitation
(c) societal impact: one call per (paper, condition) rates each G 0/1/2 (+ data_grounded)

Outputs  <run_dir>/eval/<model>/per_row.csv, summary.csv, judge_cache.jsonl ; <run_dir>/final_text.txt
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import time
from collections import Counter

import numpy as np
import pandas as pd

from . import config
from . import prompts as P0
from . import prompts_exp as P
from .io_utils import log, save_csv, done
from .llm import ChatLLM
from .units import extract_json

JV = "exp_judge_v1"


# ----------------------------------------------------------------------------- helpers (shared with evaluate.py)
class Cache:
    def __init__(self, path):
        self.path, self.d = path, {}
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

    def put(self, k, a):
        self.d[k] = a
        self.fh.write(json.dumps({"k": k, "a": a}, ensure_ascii=False) + "\n")


def key(*parts):
    import hashlib
    return hashlib.sha1("\x00".join(map(str, parts)).encode()).hexdigest()


def is_yes(a):
    m = re.match(r"\W*(yes|no)\b", str(a), re.I)
    return m.group(1).lower() == "yes" if m else "YES" in str(a).upper()


def nanmean(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(x)) if x else float("nan")


def boot_ci(x, n=2000, seed=0):
    x = np.asarray([v for v in x if not (isinstance(v, float) and math.isnan(v))], float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_p(a, b, n=5000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if len(d) < 3 or np.allclose(d, 0):
        return 1.0
    sims = np.abs((d[None, :] * np.random.default_rng(seed).choice([-1, 1], (n, len(d)))).mean(1))
    return float((1 + (sims >= abs(d.mean()) - 1e-12).sum()) / (n + 1))


# ----------------------------------------------------------------------------- judges
async def judge_list(llm, cache, refs, gens, seed):
    if not refs or not gens:
        return set(), 0
    order = list(range(len(refs)))
    random.Random(seed).shuffle(order)
    user = P.judge_list_user([refs[j] for j in order], gens)
    k = key(JV, "list", llm.model, user)
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
        if not g or not 0 < int(g.group(1)) <= len(gens):
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
            prompt = P0.judge_pair_prompt(r, g)
            k = key(JV, "pair", llm.model, prompt)
            a = cache.get(k)
            if a is None:
                todo.append((ri, gi, k, prompt))
            elif is_yes(a):
                out.add((ri, gi))
    if todo:
        ans = await asyncio.gather(*[llm.chat("", p, max_tokens=5) for *_, p in todo])
        for (ri, gi, k, _), a in zip(todo, ans):
            if not a:
                err += 1
                continue
            cache.put(k, a)
            if is_yes(a):
                out.add((ri, gi))
    return out, err


async def judge_impact(llm, cache, items):
    if not items:
        return []
    user = P.impact_user(items)
    k = key(JV, "impact", llm.model, user)
    a = cache.get(k)
    if a is None:
        a = await llm.chat(P.IMPACT_SYSTEM, user, max_tokens=40 + 30 * len(items), json_mode=True)
        if not a:
            return []
        cache.put(k, a)
    obj = extract_json(a)
    sc = {}
    for s in (obj.get("scores", []) if isinstance(obj, dict) else []):
        g = re.fullmatch(r"G(\d+)", str(s.get("g", "")).strip()) if isinstance(s, dict) else None
        if g and 0 < int(g.group(1)) <= len(items):
            try:
                sc[int(g.group(1)) - 1] = (max(0, min(2, int(s.get("score", 0)))), bool(s.get("data_grounded", False)))
            except (TypeError, ValueError):
                continue
    return [sc.get(i, (0, False)) for i in range(len(items))]


# ----------------------------------------------------------------------------- scoring
def one_to_one(M, ref, nG):
    """maximum bipartite matching (Kuhn), trying author (heavier) references first."""
    adj = {}
    for r, g in M:
        adj.setdefault(r, []).append(g)
    match_g = {}

    def aug(r, seen):
        for g in adj.get(r, []):
            if g in seen:
                continue
            seen.add(g)
            if g not in match_g or aug(match_g[g], seen):
                match_g[g] = r
                return True
        return False

    for r in sorted(adj, key=lambda x: -ref[x]["weight"]):
        aug(r, set())
    return {(r, g) for g, r in match_g.items()}


def score(ref, nG, M):
    f = lambda p, r: 2 * p * r / (p + r) if p + r > 0 else 0.0  # noqa: E731
    w = np.array([x["weight"] for x in ref], float)
    auth = np.array([x["origin"] == "author" for x in ref])
    o = {"n_gen": nG, "n_ref": len(ref), "n_author": int(auth.sum()), "n_pseudo": int((~auth).sum())}
    if not len(ref):
        return o
    r_hit = np.zeros(len(ref), bool)
    g_best, g_auth, g_ps = np.zeros(nG), np.zeros(nG, bool), np.zeros(nG, bool)
    for r, g in M:
        r_hit[r] = True
        g_best[g] = max(g_best[g], w[r])
        g_auth[g] |= auth[r]
        g_ps[g] |= not auth[r]
    o["recall"] = float((w * r_hit).sum() / w.sum())
    o["precision"] = float(g_best.mean()) if nG else 0.0
    o["f1"] = f(o["precision"], o["recall"])
    one = one_to_one(M, ref, nG)
    mw = sum(w[r] for r, _ in one)
    o["recall_1to1"] = float(mw / w.sum())
    o["precision_1to1"] = float(mw / nG) if nG else 0.0
    o["f1_1to1"] = f(o["precision_1to1"], o["recall_1to1"])
    o["n_matched_ref"] = int(r_hit.sum())
    o["n_matched_gen"] = int((g_auth | g_ps).sum())
    if auth.any():
        o["auth_recall"] = float(r_hit[auth].mean())
        o["auth_precision"] = float(g_auth.mean()) if nG else 0.0
        o["auth_f1"] = f(o["auth_precision"], o["auth_recall"])
    o["pseudo_recall"] = float(r_hit[~auth].mean()) if (~auth).any() else float("nan")
    o["novel"] = int((g_ps & ~g_auth).sum())
    o["unmatched"] = int((~(g_auth | g_ps)).sum())
    return o


def build_ref(refrow, gt_source, weights_ok=True):
    full = json.loads(refrow["full_reference_json"])
    pseudo = [x for x in full if x["origin"] == "pseudo"]
    if gt_source == "consensus":
        auth = json.loads(refrow["gt_author_json"])
    elif gt_source == "majority":
        auth = json.loads(refrow["gt_author_majority_json"])
    elif gt_source == "unanimous":
        auth = json.loads(refrow["gt_author_unanimous_json"])
    else:
        auth = json.loads(refrow["regex_units_json"])
    return [{"text": t, "origin": "author", "weight": 1.0} for t in auth] + pseudo


# ----------------------------------------------------------------------------- per-model evaluation
async def evaluate_model(args, m, rows, run_dir):
    mf = os.path.join(run_dir, f"model_{m}.csv")
    if not os.path.exists(mf):
        log(f"[eval] no {mf}; skipping {m}")
        return
    df = pd.read_csv(mf, low_memory=False)
    ref = pd.read_csv(os.path.join(run_dir, "reference.csv"), low_memory=False)
    conds = list(config.CONDITIONS) if args.conditions == "all" else [c.strip() for c in args.conditions.split(",")]
    cols = [f"lim_{m}_{c}" for c in conds if f"lim_{m}_{c}__json" in df.columns]
    if not cols:
        log(f"[eval] {m}: no generation columns - run the 'generate' stage first")
        return
    edir = os.path.join(run_dir, "eval", m)
    os.makedirs(edir, exist_ok=True)
    llm = ChatLLM(args.judge_model, backend=args.judge_backend or args.backend, max_concurrency=args.workers,
                  seed=args.seed)
    cache = Cache(os.path.join(edir, "judge_cache.jsonl"))
    types = ["list", "pairwise"] if args.eval_type == "both" else [args.eval_type]
    recs, errors = [], Counter()

    async def job(i, c):
        if not done(df.at[i, c + "__json"]) or not done(ref.at[i, "full_reference_json"]):
            return
        items = [x for x in json.loads(df.at[i, c + "__json"]) if isinstance(x, dict) and x.get("limitation")]
        gens = [x["limitation"] for x in items]
        R = build_ref(ref.loc[i], args.gt_source)
        refs = [x["text"] for x in R]
        base = {"model": m, "condition": c.split(f"lim_{m}_", 1)[1], "row": int(i), "pmid": str(df.at[i, args.id_col])}
        imp = await judge_impact(llm, cache, items) if args.impact_eval else []
        ib = {"impact_mean": nanmean([s for s, _ in imp]) if imp else float("nan"),
              "impact_specific": nanmean([s == 2 for s, _ in imp]) if imp else float("nan"),
              "impact_data_grounded": nanmean([g for _, g in imp]) if imp else float("nan"),
              "ph_cited": nanmean([bool(x.get("ph_tags")) for x in items]) if items else float("nan")}
        for t in types:
            if t == "list":
                M, e = await judge_list(llm, cache, refs, gens, args.seed * 7919 + int(i))
            else:
                M, e = await judge_pairs(llm, cache, refs, gens)
            errors[t] += e
            recs.append({**base, "eval_type": t, **score(R, len(gens), M), **ib, "matches": json.dumps(sorted(M))})

    jobs = [(i, c) for i in rows for c in cols]
    t0 = time.time()
    for s in range(0, len(jobs), 100):
        await asyncio.gather(*[job(i, c) for i, c in jobs[s:s + 100]])
        log(f"[eval:{m}] {min(s + 100, len(jobs))}/{len(jobs)} (paper, condition) jobs  "
            f"{(time.time() - t0) / 60:.1f} min  {llm.stats()}")
    per = pd.DataFrame(recs)
    per.to_csv(os.path.join(edir, "per_row.csv"), index=False)
    if len(per):
        num = [c for c in per.columns if c not in ("model", "condition", "row", "pmid", "eval_type", "matches")]
        per.groupby(["eval_type", "condition"])[num].mean().reset_index().to_csv(os.path.join(edir, "summary.csv"), index=False)
    json.dump({"judge": llm.stats(), "failed_calls": dict(errors), "gt_source": args.gt_source,
               "time": time.ctime()}, open(os.path.join(edir, "eval_meta.json"), "w"))
    log(f"[eval:{m}] wrote {edir}/per_row.csv ({len(per)} records); failed judge calls {dict(errors)}")


# ----------------------------------------------------------------------------- final_text.txt
def _table(headers, data):
    w = [max(len(str(h)), *(len(str(r[k])) for r in data)) if data else len(str(h)) for k, h in enumerate(headers)]
    line = "+" + "+".join("-" * (x + 2) for x in w) + "+"
    o = [line, "| " + " | ".join(str(h).ljust(w[k]) for k, h in enumerate(headers)) + " |", line.replace("-", "=")]
    for r in data:
        o.append("| " + " | ".join(str(x).ljust(w[k]) if k == 0 else str(x).rjust(w[k]) for k, x in enumerate(r)) + " |")
    o.append(line)
    return "\n".join(o)


def f3(x):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def combine(args, run_dir, rows):
    L = []
    W = L.append
    W("=" * 120)
    W(f" LIMITATIONS WITH SOCIETAL IMPACT - EXPERIMENT RESULTS   {time.strftime('%Y-%m-%d %H:%M')}")
    W("=" * 120)
    W(f" run dir   : {run_dir}")
    W(f" judge     : {args.judge_model}; author GT source: {args.gt_source} (gt_min_votes={args.gt_min_votes}); "
      f"pseudo weights {args.vote_weights}")
    rp = os.path.join(run_dir, "reference.csv")
    if os.path.exists(rp):
        ref = pd.read_csv(rp, low_memory=False)
        rr = [i for i in rows if i in ref.index and done(ref.at[i, "full_reference_json"])]
        models = [m for m in args.models.split(",") if f"ext_{m}__json" in ref.columns]
        W(f"\nA. GROUND TRUTH ({len(rr)} papers)")
        nx = np.array([int(ref.at[i, "n_regex_units"]) for i in rr])
        data = [["regex (limitations_section split on '\\n')", f"{nx.sum()}", f"{nx.mean():.2f}", f"{np.median(nx):.0f}",
                 f"{(nx == 0).sum()}"]]
        for m in models:
            n = np.array([len(json.loads(ref.at[i, f"ext_{m}__json"])) if done(ref.at[i, f"ext_{m}__json"]) else 0 for i in rr])
            data.append([f"LLM extraction: {m}", f"{n.sum()}", f"{n.mean():.2f}", f"{np.median(n):.0f}", f"{(n == 0).sum()}"])
        for lab, c in (("consensus, all extractors agree (unanimous)", "n_gt_unanimous"),
                       ("consensus, >= 2 extractors (majority)", "n_gt_majority"),
                       (f"AUTHOR GT used (>= {min(args.gt_min_votes, len(models))} votes)", "n_gt_author"),
                       ("pseudo labels (additional, majority vote)", "n_pseudo"),
                       ("FULL REFERENCE (author + pseudo)", "n_full_reference")):
            n = np.array([int(float(ref.at[i, c] or 0)) for i in rr])
            data.append([lab, f"{n.sum()}", f"{n.mean():.2f}", f"{np.median(n):.0f}", f"{(n == 0).sum()}"])
        W(_table(["source", "total", "mean/paper", "median", "papers with 0"], data))
        agree = {}
        for i in rr:
            for cl in json.loads(ref.at[i, "gt_clusters_json"]):
                for a in cl["models"]:
                    for b in models:
                        if a != b:
                            x = agree.setdefault((a, b), [0, 0])
                            x[1] += 1
                            x[0] += b in cl["models"]
        if agree:
            W("   extractor agreement (share of A's extracted units that B also extracted): "
              + "; ".join(f"{a}->{b} {x / n:.2f}" for (a, b), (x, n) in sorted(agree.items()) if n))
    per = []
    for m in ("qwen", "gemma", "mistral"):
        p = os.path.join(run_dir, "eval", m, "per_row.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            per.append(d[d.row.isin(rows)])
    if not per:
        W("\n(no evaluation results yet)")
    else:
        per = pd.concat(per, ignore_index=True)
        order = {c: k for k, c in enumerate(config.CONDITIONS)}
        for t in sorted(per.eval_type.unique()):
            D = per[per.eval_type == t]
            W(f"\nB[{t}]. MAIN RESULTS (any-match; mean over papers; F1 95% CI)  "
              + ("one gpt call per paper x condition" if t == "list" else "every reference x generated pair judged Yes/No"))
            data = []
            for (m, c), g in sorted(D.groupby(["model", "condition"]), key=lambda x: (x[0][0], order.get(x[0][1], 99))):
                lo, hi = boot_ci(g.f1.tolist())
                data.append([f"{m}/{c}", len(g), f"{g.n_gen.mean():.1f}", f3(g.precision.mean()), f3(g.recall.mean()),
                             f3(g.f1.mean()), f"[{f3(lo)},{f3(hi)}]", f3(g.f1_1to1.mean()),
                             f"{f3(g.get('auth_precision', pd.Series(dtype=float)).mean())}/{f3(g.get('auth_recall', pd.Series(dtype=float)).mean())}/{f3(g.get('auth_f1', pd.Series(dtype=float)).mean())}",
                             f3(g.pseudo_recall.mean()), f"{g.novel.mean():.2f}", f3(g.impact_mean.mean()),
                             f3(g.impact_specific.mean())])
            W(_table(["model/condition", "papers", "#gen", "P (gen cov.)", "R (ref cov.)", "F1", "F1 95% CI", "F1 1:1",
                      "author P/R/F1", "pseudo R", "novel/paper", "impact 0-2", "impact=2"], data))
            W(f"\nC[{t}]. ABLATIONS per model (paired over the same papers; d = after - before; * p < 0.05)")
            pairs = [("+public-health data", "zs", "zs_ph"), ("+public-health data", "rag", "rag_ph"),
                     ("+public-health data", "agents", "agents_ph"), ("+public-health data", "agents_rag", "agents_rag_ph"),
                     ("+literature RAG", "zs", "rag"), ("+literature RAG", "agents", "agents_rag"),
                     ("+literature RAG", "agents_ph", "agents_rag_ph"),
                     ("agents+leader vs single model", "zs", "agents"), ("agents+leader vs single model", "rag", "agents_rag"),
                     ("agents+leader vs single model", "zs_ph", "agents_ph"),
                     ("agents+leader vs single model", "rag_ph", "agents_rag_ph"),
                     ("full system vs zero-shot", "zs", "agents_rag_ph")]
            data = []
            for m in sorted(D.model.unique()):
                Dm = D[D.model == m].set_index(["condition", "row"])
                for lab, a, b in pairs:
                    if a not in Dm.index.get_level_values(0) or b not in Dm.index.get_level_values(0):
                        continue
                    A, B = Dm.loc[a], Dm.loc[b]
                    common = A.index.intersection(B.index)
                    if len(common) < 2:
                        continue
                    cells = [m, lab, f"{a} -> {b}", len(common)]
                    for kk in ("f1", "auth_f1", "novel", "impact_mean"):
                        if kk not in A.columns:
                            cells.append("-")
                            continue
                        x, y = A.loc[common, kk].astype(float).values, B.loc[common, kk].astype(float).values
                        dd, p = nanmean((y - x).tolist()), paired_p(y, x)
                        cells.append(f"{dd:+.3f}{'*' if p < 0.05 else ''} (p={p:.2g})")
                    data.append(cells)
            if data:
                W(_table(["model", "effect", "conditions", "papers", "d F1", "d author F1", "d novel", "d impact"], data))
            W(f"\nD[{t}]. CONDITION AVERAGED OVER MODELS")
            g = D.groupby("condition")[["precision", "recall", "f1", "f1_1to1", "pseudo_recall", "novel", "impact_mean",
                                        "impact_data_grounded", "ph_cited"]].mean()
            g = g.reindex([c for c in config.CONDITIONS if c in g.index])
            W(_table(["condition", "P", "R", "F1", "F1 1:1", "pseudo R", "novel", "impact", "impact grounded", "cites PH"],
                     [[k, f3(r.precision), f3(r.recall), f3(r.f1), f3(r.f1_1to1), f3(r.pseudo_recall), f"{r.novel:.2f}",
                       f3(r.impact_mean), f3(r.impact_data_grounded), f3(r.ph_cited)] for k, r in g.iterrows()]))
        if per.eval_type.nunique() == 2:
            a = per[per.eval_type == "list"].set_index(["model", "condition", "row"])
            b = per[per.eval_type == "pairwise"].set_index(["model", "condition", "row"])
            c = a.index.intersection(b.index)
            if len(c):
                r = np.corrcoef(a.loc[c, "f1"].astype(float), b.loc[c, "f1"].astype(float))[0, 1] if len(c) > 2 else float("nan")
                W(f"\nE. LIST vs PAIRWISE JUDGE: {len(c)} (paper, condition) cells; F1 correlation {f3(r)}; mean F1 "
                  f"list {f3(a.loc[c, 'f1'].mean())} vs pairwise {f3(b.loc[c, 'f1'].mean())}")
    W("\nF. GENERATION DIAGNOSTICS")
    data = []
    for m in ("qwen", "gemma", "mistral"):
        p = os.path.join(run_dir, f"model_{m}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p, low_memory=False)
        for c in [x for x in d.columns if (x.startswith(f"lim_{m}_") or x.startswith(f"ext_{m}") or
                                         x.startswith(f"pseudo_cand_{m}")) and x.endswith("__meta")]:
            metas = [json.loads(d.at[i, c]) for i in rows if i in d.index and done(d.at[i, c])]
            if not metas:
                continue
            extra = ""
            if any("n_emitted" in x for x in metas):
                extra = (f"agents emitted {nanmean([x.get('n_emitted') for x in metas]):.1f} -> gate "
                         f"{nanmean([x.get('n_after_gate') for x in metas]):.1f} -> leader ok "
                         f"{sum(1 for x in metas if x.get('leader_ok'))}/{len(metas)}")
            if any("n_unanchored_dropped" in x for x in metas):
                extra = f"unanchored quotes dropped {sum(x.get('n_unanchored_dropped', 0) for x in metas)}"
            data.append([c[:-6], len(metas), f"{nanmean([x.get('n', 0) for x in metas]):.1f}",
                         " ".join(f"{k}:{v}" for k, v in Counter(x.get('parse_mode', '?') for x in metas).items()),
                         sum(1 for x in metas if "error" in x), extra])
    if data:
        W(_table(["column", "papers", "#items", "parse", "errors", "notes"], data))
    W("\nNOTES")
    W(" - Author GT = limitations the three open LLMs extracted from the full paper and gpt-4o-mini clustered;")
    W("   only clusters found by >= gt_min_votes extractors are kept. The regex units are shown for comparison.")
    W(" - Pseudo labels come from the same three models (prompted for ADDITIONAL limitations with public-health")
    W("   data), so _ph conditions may gain on pseudo matches: read 'author P/R/F1' next to the weighted scores.")
    W(" - 'novel' = generated items matching a pseudo label but no author limitation; expert review is still")
    W("   needed to confirm they are valid (make_review_sample.py).")
    W("=" * 120)
    txt = "\n".join(L) + "\n"
    open(os.path.join(run_dir, "final_text.txt"), "w").write(txt)
    print(txt)
    log(f"wrote {os.path.join(run_dir, 'final_text.txt')}")
