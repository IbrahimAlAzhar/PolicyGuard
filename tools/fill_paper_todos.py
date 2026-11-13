#!/usr/bin/env python
"""
Collects every fact the PolicyGuard manuscript's TO DO items need from the v2 run, the data folder, the model
folders and the web app, and writes:
    <out_dir>/todo_report.txt     everything found, item by item, with notes on what still needs a human
    <out_dir>/latex_snippets.tex  ready-to-paste LaTeX (sentences and tables) for items a-h

No LLM is called (optional: --probe_openai 1 makes ONE 1-token gpt-4o-mini call to read the snapshot name).

  python tools/fill_paper_todos.py
  python tools/fill_paper_todos.py --run_dir outputs_exp/sample200_seed42 --n_rows 50 --trace_model gemma
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from medlim import config  # noqa: E402
from medlim.publichealth import fingerprint  # noqa: E402

CONDS = ["zs", "zs_ph", "rag", "rag_ph", "agents", "agents_ph", "agents_rag", "agents_rag_ph"]
BASE = os.path.dirname(ROOT)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default=os.path.join(ROOT, "outputs_exp/sample200_seed42"))
    ap.add_argument("--n_rows", type=int, default=50)
    ap.add_argument("--models", default="qwen,gemma,mistral")
    ap.add_argument("--eval_type", default="list")
    ap.add_argument("--gt_source", default="consensus")
    ap.add_argument("--data_csv", default=os.path.join(ROOT, "data/pubmed_oa_regex_lim_10000.csv"))
    ap.add_argument("--web_dirs", default=f"{BASE}/med_limagents_web,{BASE}/med_equity")
    ap.add_argument("--model_dirs", default="qwen=/lstr/sahara/datalab-ml/models/qwen3_8b,"
                                            "gemma=/lstr/sahara/datalab-ml/models/gemma3_27b_it,"
                                            "mistral=/lstr/sahara/datalab-ml/models/mistral_small_3_1_24b_instruct")
    ap.add_argument("--trace_model", default="gemma", help="model whose agents_rag_ph output is traced in Table trace")
    ap.add_argument("--probe_openai", type=int, default=0)
    ap.add_argument("--distinct_papers", type=int, default=1, help="take the two trace examples from different papers")
    ap.add_argument("--out_dir", default=None)
    return ap.parse_args()


# ----------------------------------------------------------------------------- helpers
def tex(s, n=None):
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    s = re.sub(r"\s+", " ", s).strip()
    if n and len(s) > n:
        s = s[:n].rsplit(" ", 1)[0] + " \\ldots"
        tail = True
    else:
        tail = False
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    out = "".join(rep.get(ch, ch) for ch in (s[:-len(" \\ldots")] if tail else s))
    return out + (r" \ldots" if tail else "")


def done(v):
    return isinstance(v, str) and v.strip() not in ("", "nan")


def sign_flip_p(d, n=5000, seed=0):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) < 3 or np.allclose(d, 0):
        return 1.0
    sims = np.abs((d[None, :] * np.random.default_rng(seed).choice([-1, 1], (n, len(d)))).mean(1))
    return float((1 + (sims >= abs(d.mean()) - 1e-12).sum()) / (n + 1))


def boot_ci(d, n=2000, seed=0):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return (np.nan, np.nan)
    m = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


class Out:
    def __init__(self):
        self.rep, self.tex = [], []

    def h(self, t):
        self.rep += ["", "=" * 100, t, "=" * 100]
        self.tex += ["", "% " + "=" * 96, "% " + t, "% " + "=" * 96]

    def r(self, *a):
        self.rep.append(" ".join(str(x) for x in a))

    def t(self, s):
        self.tex.append(s)


# ----------------------------------------------------------------------------- (a) public-data releases
def item_a(o, S):
    o.h("(a) PUBLIC-DATA RELEASES: identifier, file date, hash (Section 'Public-health context')")
    used = {}
    if S is not None and "ph_provenance_json" in S:
        for v in S["ph_provenance_json"].dropna():
            try:
                for p in json.loads(v):
                    used.setdefault(p.get("file"), {"source": p.get("source"), "release": p.get("release"),
                                                    "sha256": p.get("sha256"), "mtime": p.get("mtime")})
            except Exception:
                continue
    o.r(f"data root: {config.DATA_ROOT}")
    o.r("files actually cited in the run's provenance records:")
    rows = []
    for f, info in sorted(used.items(), key=lambda x: str(x[0])):
        fp = fingerprint(f) if f else {}
        rel = info.get("release") or {}
        yrs = sorted(set(re.findall(r"20\d\d", os.path.basename(str(f)))))
        o.r(f"  - {info['source']:<22} {f}\n      release fields {rel}; years in file name {yrs}; size "
            f"{fp.get('size_bytes')}; file modified {fp.get('mtime')}; sha256 {str(fp.get('sha256'))[:16]}; "
            f"unchanged since run: {fp.get('sha256') == info.get('sha256')}")
        rows.append((info["source"], f, rel, yrs, fp))
    o.r("\nall files in the data folders (for the deposit list):")
    for k, p in config.PH_PATHS.items():
        files = sorted(glob.glob(os.path.join(p, "**", "*.*"), recursive=True)) if os.path.isdir(p) else ([p] if os.path.exists(p) else [])
        for f in files[:12]:
            st = os.stat(f)
            o.r(f"  {k:<22} {os.path.relpath(f, config.DATA_ROOT)}  {st.st_size:,} B  modified "
                f"{pd.Timestamp(st.st_mtime, unit='s'):%Y-%m-%d}")
        if not files:
            o.r(f"  {k:<22} (missing)")
    # PLACES release year from the data itself
    pl = config.PH_PATHS["cdc_places_county"]
    if os.path.exists(pl):
        try:
            h = pd.read_csv(pl, nrows=0).columns
            cols = [c for c in ("Year", "DataSource") if c in h]
            if cols:
                d = pd.read_csv(pl, usecols=cols, low_memory=False)
                o.r(f"  PLACES county file: {', '.join(f'{c}={sorted(d[c].dropna().astype(str).unique())[:6]}' for c in cols)}")
        except Exception as e:  # noqa: BLE001
            o.r(f"  PLACES read failed: {e}")
    for pat in ("README*", "*manifest*", "*download*", "*.log", "*source*"):
        for f in glob.glob(os.path.join(config.DATA_ROOT, "**", pat), recursive=True)[:10]:
            o.r(f"  note file: {f}")
            try:
                o.r("     " + " | ".join(open(f, errors="ignore").read().splitlines()[:6]))
            except Exception:
                pass
    o.r("\nNOTE: 'file modified' is the local file time, usually the download date; confirm against your download "
        "records. Release identifiers come from the data's Year/DataSource fields and file names; check them.")
    o.t("% Replace the \\fix in 'Public-health context with provenance tags' with this sentence + table:")
    o.t("Table~\\ref{tab:releases} lists the release, local file date, and SHA-256 prefix of each file used; the files "
        "will be deposited with the run traces at \\fix{repository/DOI}.")
    o.t("\\begin{table}[t]\n\\caption{Public-data files used in the experiments.}\\label{tab:releases}\n\\centering\\scriptsize")
    o.t("\\begin{tabular}{@{}llll@{}}\n\\toprule\nSource & File & Release & File date / SHA-256 \\\\\n\\midrule")
    for src, f, rel, yrs, fp in rows:
        relstr = "; ".join(f"{k} {', '.join(map(str, v if isinstance(v, list) else [v]))}" for k, v in (rel or {}).items()) or ", ".join(yrs)
        relcell = tex(relstr, 40) if relstr else "\\fix{release}"
        o.t(f"{tex(src)} & {tex(os.path.basename(str(f)))} & {relcell} & {tex(str(fp.get('mtime', ''))[:10])} / "
            f"\\texttt{{{str(fp.get('sha256', ''))[:10]}}} \\\\")
    o.t("\\bottomrule\n\\end{tabular}\n\\end{table}")


# ----------------------------------------------------------------------------- (b) adult population in the simulator
def item_b(o, web_dirs):
    o.h("(b) SCENARIO SIMULATOR: which population term is used? (Interface section, 'Scenarios')")
    pats = re.compile(r"TotalPop18plus|TotalPopulation|E_TOTPOP|E_AGE17|E_AGE65|adult|18\s*\+|pop(ulation)?_?(col|term|count)?\s*[\*=]|"
                      r"monte|n_draws|draws|efficacy_loss|simulat", re.I)
    hits = 0
    for d in [x for x in web_dirs.split(",") if x]:
        if not os.path.isdir(d):
            o.r(f"  (not found: {d})")
            continue
        for f in sorted(glob.glob(os.path.join(d, "**", "*.py"), recursive=True)):
            try:
                lines = open(f, errors="ignore").read().splitlines()
            except Exception:
                continue
            for n, line in enumerate(lines):
                if pats.search(line) and hits < 80:
                    o.r(f"  {os.path.relpath(f, d)}:{n + 1}: {line.strip()[:160]}")
                    hits += 1
    o.r("\nREAD THE LINES ABOVE: if the population multiplied by prevalence is total population (TotalPopulation, "
        "E_TOTPOP), the term is NOT adult-only. PLACES also ships TotalPop18plus in recent releases.")
    o.t("% (b) choose ONE sentence, depending on what the code above shows, and delete the \\fix:")
    o.t("% if adult population is used:")
    o.t("% The population term is the adult (18+) population of the matching tracts, consistent with the PLACES estimates.")
    o.t("% if total population is used:")
    o.t("% The current tool multiplies prevalence by the total tract population; because PLACES estimates refer to "
        "adults, this overstates the affected population and will be replaced by the adult population.")


# ----------------------------------------------------------------------------- (e) sample order + duplicates
def item_e(o, S, M, args):
    o.h("(e) DATASET: first 50 rows? duplicate versions in the retrieval corpus?")
    rows_done = None
    for m, df in M.items():
        ok = [i for i in df.index if all(done(df.at[i, f"lim_{m}_{c}"]) for c in CONDS if f"lim_{m}_{c}" in df)]
        o.r(f"  {m}: rows with all 8 configurations = {len(ok)}; indices {ok[:3]}...{ok[-3:] if ok else ''}")
        rows_done = set(ok) if rows_done is None else rows_done & set(ok)
    first = rows_done == set(range(args.n_rows)) if rows_done is not None else False
    o.r(f"  rows done by all models == sample rows 0..{args.n_rows - 1}: {first}")
    o.r("  (sample.csv keeps the random seed-42 order, so rows 0..49 are the first 50 of the sample)")
    dup_line = "\\fix{duplicate check}"
    if os.path.exists(args.data_csv) and S is not None:
        head = pd.read_csv(args.data_csv, nrows=0).columns
        cols = [c for c in ("pmid", "pmc", "doi", "abstract", "title") if c in head]
        D = pd.read_csv(args.data_csv, usecols=cols, low_memory=False)
        D["pmid"] = D["pmid"].astype(str).str.replace(r"\.0$", "", regex=True)
        sid = set(S["pmid"].astype(str).str.replace(r"\.0$", "", regex=True))
        D["_abs"] = D["abstract"].fillna("").map(lambda x: norm(x)[:300]) if "abstract" in D else ""
        n_pmid_dup = int(D.duplicated("pmid").sum())
        n_doi_dup = int(D.dropna(subset=["doi"]).duplicated("doi").sum()) if "doi" in D else 0
        n_abs_dup = int(D[D._abs.str.len() > 50].duplicated("_abs").sum())
        samp = D[D.pmid.isin(sid)]
        corp = D[~D.pmid.isin(sid)]
        leak_abs = int(corp[corp._abs.str.len() > 50]._abs.isin(set(samp._abs)).sum())
        leak_doi = int(corp.doi.dropna().isin(set(samp.doi.dropna())).sum()) if "doi" in D else 0
        o.r(f"  collection: {len(D):,} rows; duplicate PMIDs {n_pmid_dup}; duplicate DOIs {n_doi_dup}; "
            f"identical abstracts (first 300 chars) {n_abs_dup}")
        o.r(f"  corpus entries duplicating a sampled paper: same DOI {leak_doi}, same abstract {leak_abs}")
        self_hits = 0
        for _, r in S.head(args.n_rows).iterrows():
            a = norm(r.get("abstract", ""))[:200]
            if a and done(r.get("rag_top3_json")) and any(a[:150] in norm(p) for p in json.loads(r["rag_top3_json"])):
                self_hits += 1
        o.r(f"  evaluated papers whose retrieved passages contain their own abstract: {self_hits}")
        o.r("  NOTE: the code removes duplicates by PMID only; preprint/published pairs have different PMIDs and are "
            "not removed unless they share text.")
        dup_line = (f"Duplicates were removed by PMID only; preprint and published versions of the same work were not "
                    f"merged. In the collection, {leak_doi} corpus entries share a DOI and {leak_abs} share an abstract "
                    f"with a sampled publication, and {self_hits} of the {args.n_rows} evaluated publications retrieved "
                    f"a passage containing their own abstract.")
    o.t("% (e) replace the two \\fix items in 'Dataset and computational requirements':")
    o.t(f"% first-50 check: {'CONFIRMED' if first else 'NOT CONFIRMED - see todo_report.txt'}; then delete that \\fix.")
    o.t(dup_line)


# ----------------------------------------------------------------------------- (f) checkpoints, versions, GPUs, dates
def item_f(o, args, M):
    o.h("(f) MODELS, SOFTWARE VERSIONS, GPU, WALL-CLOCK, RUN DATES")
    names = {}
    for kv in args.model_dirs.split(","):
        m, p = kv.split("=", 1)
        info = {}
        for fn in ("config.json", "generation_config.json"):
            f = os.path.join(p, fn)
            if os.path.exists(f):
                try:
                    j = json.load(open(f))
                    info.update({k: j[k] for k in ("_name_or_path", "architectures", "torch_dtype", "transformers_version", "model_type") if k in j})
                except Exception:
                    pass
        for fn in ("README.md", "model_info.txt"):
            f = os.path.join(p, fn)
            if os.path.exists(f):
                txt = open(f, errors="ignore").read()
                ids = re.findall(r"(?:huggingface\.co/|base_model:\s*|model_id\s*[:=]\s*)([\w\-.]+/[\w\-.]+)", txt)
                if ids:
                    info["hub_ids_in_" + fn] = sorted(set(ids))[:5]
        names[m] = info
        o.r(f"  {m}: {p}\n     {info}")
    for pkg in ("vllm", "transformers", "torch", "bitsandbytes", "openai"):
        try:
            mod = __import__(pkg)
            o.r(f"  {pkg} {getattr(mod, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001
            o.r(f"  {pkg}: not importable ({type(e).__name__})")
    snap = "\\fix{gpt-4o-mini snapshot}"
    if args.probe_openai:
        try:
            from openai import OpenAI
            r = OpenAI().chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "ok"}], max_tokens=1)
            snap = r.model
            o.r(f"  gpt-4o-mini resolves to snapshot: {snap} (today; check it matches the run period)")
        except Exception as e:  # noqa: BLE001
            o.r(f"  OpenAI probe failed: {e}")
    gpus, spans = Counter(), []
    for f in sorted(glob.glob(os.path.join(ROOT, "outputs_exp", "pbs_exp_*.out")) + glob.glob(os.path.join(args.run_dir, "logs", "*"))):
        try:
            txt = open(f, errors="ignore").read()
        except Exception:
            continue
        for g in re.findall(r"NVIDIA [A-Za-z0-9\- ]+?\d+GB", txt):
            gpus[g.strip()] += 1
        starts = re.findall(r"=== (\w{3} \w{3} +\d+ [\d:]+ \w+ \d{4})", txt)
        if len(starts) >= 2:
            try:
                a, b = pd.Timestamp(" ".join(starts[0].split()[:4] + starts[0].split()[-1:])), \
                       pd.Timestamp(" ".join(starts[-1].split()[:4] + starts[-1].split()[-1:]))
                spans.append((os.path.basename(f), a, b))
            except Exception:
                pass
    o.r(f"  GPUs seen in logs: {dict(gpus)}")
    tot = 0.0
    for f, a, b in spans:
        h = (b - a).total_seconds() / 3600
        tot += h
        o.r(f"  {f}: {a} -> {b}  ({h:.1f} h)")
    dates = []
    for m in M:
        for f in [os.path.join(args.run_dir, f"model_{m}.csv"), os.path.join(args.run_dir, "eval", m, "per_row.csv")]:
            if os.path.exists(f):
                dates.append(pd.Timestamp(os.path.getmtime(f), unit="s"))
    for f in glob.glob(os.path.join(args.run_dir, "logs", "*_output.txt")):
        dates.append(pd.Timestamp(os.path.getmtime(f), unit="s"))
    o.r(f"  run files last modified between {min(dates):%Y-%m-%d} and {max(dates):%Y-%m-%d}" if dates else "  no run files")
    o.r("  NOTE: wall-clock from PBS .out keeps only the LAST submission of each job; add earlier runs if you resubmitted.")
    o.t("% (f) replace the two \\fix items in the model paragraph (check checkpoint names against the hub ids above):")
    o.t("The checkpoints are Qwen3-8B (\\texttt{Qwen/Qwen3-8B}), Gemma~3 27B (\\texttt{google/gemma-3-27b-it}), and Mistral "
        "Small~3.1 24B (\\texttt{mistralai/Mistral-Small-3.1-24B-Instruct-2503}); Gemma and Mistral were loaded with "
        "4-bit bitsandbytes quantization to fit one GPU. We used vLLM \\fix{version} and the GPT-4o-mini snapshot "
        f"\\texttt{{{tex(snap)}}}; runs took place between {min(dates):%B %-d} and {max(dates):%B %-d, %Y}." if dates else
        "% run dates not found")
    g = ", ".join(f"{k}" for k in gpus) or "\\fix{GPU type}"
    o.t(f"Each model ran on one {tex(g)}; generation, extraction, and evaluation took about {tot:.0f} GPU-hours in total "
        "\\fix{check against your job history}.")


# ----------------------------------------------------------------------------- (g) ablation table with all 12 retrieval contrasts
def item_g(o, P, models, n_rows):
    o.h("(g) ABLATION TABLE: paired tests for all contrasts, incl. the missing zs_ph -> rag_ph")
    effects = [("+ literature retrieval", [("zs", "rag"), ("zs_ph", "rag_ph"), ("agents", "agents_rag"), ("agents_ph", "agents_rag_ph")]),
               ("+ public-health context", [("zs", "zs_ph"), ("rag", "rag_ph"), ("agents", "agents_ph"), ("agents_rag", "agents_rag_ph")]),
               ("agent pipeline vs.\\ single model", [("zs", "agents"), ("zs_ph", "agents_ph"), ("rag", "agents_rag"), ("rag_ph", "agents_rag_ph")]),
               ("full system vs.\\ zs", [("zs", "agents_rag_ph")])]
    o.r(f"  {'effect':<34} {'model':<8} {'contrast':<28} {'n':>3} {'dF1':>8} {'95% CI':>18} {'p':>7}")
    agg = []
    for lab, pairs in effects:
        deltas, dec, sig_dec, sig_inc, n = [], 0, 0, 0, 0
        for m in models:
            for a, b in pairs:
                x = P[(P.model == m) & (P.condition == a)].set_index("row").f1
                y = P[(P.model == m) & (P.condition == b)].set_index("row").f1
                common = x.index.intersection(y.index)
                common = [i for i in common if i < n_rows]
                if len(common) < 2:
                    continue
                d = (y.loc[common] - x.loc[common]).values.astype(float)
                p = sign_flip_p(d)
                lo, hi = boot_ci(d)
                n += 1
                deltas.append(np.nanmean(d))
                dec += np.nanmean(d) < 0
                sig_dec += (p < 0.05) and np.nanmean(d) < 0
                sig_inc += (p < 0.05) and np.nanmean(d) > 0
                o.r(f"  {lab.replace(chr(92), ''):<34} {m:<8} {a + ' -> ' + b:<28} {len(common):>3} {np.nanmean(d):>+8.3f} "
                    f"[{lo:+.3f},{hi:+.3f}] {p:>7.3f}{' *' if p < 0.05 else ''}")
        agg.append((lab, n, np.mean(deltas) if deltas else np.nan, dec, sig_dec, sig_inc, deltas))
    o.t("% (g) replacement for Table ablation (all contrasts tested; delete the \\fix below the table)")
    o.t("\\begin{tabular}{@{}lccccc@{}}\n\\toprule\nEffect & Contrasts & $\\bar{\\Delta}$F1 & Dec. & Sig.\\ dec. & Sig.\\ inc. \\\\\n\\midrule")
    for lab, n, mean, dec, sd, si, _ in agg:
        o.t(f"{lab} & {n} & ${mean:+.3f}$ & {dec} & {sd}/{n} & {si}/{n} \\\\")
    o.t("\\bottomrule\n\\end{tabular}")
    o.t("% caption: change 'Sig.' to: Sig. dec./inc.: contrasts with a significant decrease/increase "
        "(sign-flip permutation test, 5,000 permutations, p<0.05, uncorrected, exploratory).")
    o.r("\n  per-contrast 95% bootstrap CIs are listed above; add them as an appendix table if space allows.")


# ----------------------------------------------------------------------------- (h) gate pass rates by evidence source
def src_of(c):
    s = str(c.get("evidence_source", "manuscript")).lower()
    return "public data" if ("public" in s or "data" in s) else "corpus" if any(k in s for k in ("corpus", "literature", "reference")) else "manuscript"


def item_h(o, S, M, args):
    o.h("(h) EVIDENCE GATE BY SOURCE (Table gate) + candidates for an 'unsupported inference' example")
    tab = defaultdict(lambda: Counter())
    examples = []
    for m, df in M.items():
        for c in ("agents", "agents_ph", "agents_rag", "agents_rag_ph"):
            col = f"lim_{m}_{c}__trace"
            if col not in df:
                continue
            for i in df.index[:args.n_rows]:
                if not done(df.at[i, col]):
                    continue
                tr = json.loads(df.at[i, col])
                for cand in tr.get("candidates", []):
                    if "anchored" not in cand:          # removed earlier by the boilerplate filter
                        tab[(m, c)]["boilerplate"] += 1
                        continue
                    s = src_of(cand)
                    tab[(m, c)][s + "_n"] += 1
                    tab[(m, c)][s + "_pass"] += bool(cand["anchored"])
                    if s == "public data" and cand["anchored"] and c.endswith("_ph"):
                        notes = str(S.at[i, "ph_notes"]) if S is not None and "ph_notes" in S else ""
                        nonus = '"non_us_mentions": []' not in notes and "non_us_mentions" in notes
                        states = str(S.at[i, "ph_states"]) if S is not None and "ph_states" in S else ""
                        examples.append((not nonus and states not in ("", "nan"), m, c, i, cand, notes, states))
                meta = df.at[i, f"lim_{m}_{c}__meta"]
                if done(meta) and json.loads(meta).get("gate_fallback"):
                    tab[(m, c)]["gate_fallback_rows"] += 1
    o.r(f"  {'model':<8} {'config':<15} {'manuscript':>18} {'corpus':>14} {'public data':>16} {'boilerpl.':>9} {'fallback':>8}")
    o.t("% (h) new table: gate pass rate by evidence source (passed / emitted)")
    o.t("\\begin{tabular}{@{}llccc@{}}\n\\toprule\nModel & Configuration & Manuscript & Corpus & Public data \\\\\n\\midrule")
    for (m, c), t in sorted(tab.items()):
        def cell(s):
            n, p = t[s + "_n"], t[s + "_pass"]
            return f"{p}/{n} ({p / n:.0%})" if n else "--"
        o.r(f"  {m:<8} {c:<15} {cell('manuscript'):>18} {cell('corpus'):>14} {cell('public data'):>16} "
            f"{t['boilerplate']:>9} {t['gate_fallback_rows']:>8}")
        o.t(f"{m} & {tex(c)} & {cell('manuscript').replace('%', chr(92) + '%')} & {cell('corpus').replace('%', chr(92) + '%')} & "
            f"{cell('public data').replace('%', chr(92) + '%')} \\\\")
    o.t("\\bottomrule\n\\end{tabular}")
    o.r("\n  'fallback' = papers where no candidate passed and all were kept (flagged) - mention if non-zero.")
    o.r("\n  CANDIDATES FOR 'resolvable tag + unsupported inference' (passed by tag resolution; papers outside the US")
    o.r("  or with no US state listed first - YOU must judge which inference is unsupported):")
    examples.sort(key=lambda x: x[0])
    for _, m, c, i, cand, notes, states in examples[:10]:
        o.r(f"   - {m}/{c} row {i} pmid {S.at[i, 'pmid'] if S is not None else '?'}  states [{states}]  setting {notes[:120]}")
        o.r(f"       claim: {cand.get('claim', '')[:220]}")
        o.r(f"       evidence: {cand.get('evidence_span', '')[:160]}  tags {cand.get('ph_tags')}")


# ----------------------------------------------------------------------------- (c) provenance trace table
def item_c(o, S, M, R, P, args):
    o.h("(c) PROVENANCE TRACE (Table trace) from the full system agents_rag_ph")
    m = args.trace_model
    df = M.get(m)
    if df is None or f"lim_{m}_agents_rag_ph__json" not in df:
        o.r(f"  no agents_rag_ph output for {m}")
        return
    from medlim.exp_eval import build_ref
    picks = {"matched": None, "unmatched": None}
    for i in df.index[:args.n_rows]:
        if not done(df.at[i, f"lim_{m}_agents_rag_ph__json"]) or R is None or not done(R.at[i, "full_reference_json"]):
            continue
        items = [x for x in json.loads(df.at[i, f"lim_{m}_agents_rag_ph__json"]) if isinstance(x, dict) and x.get("limitation")]
        pr = P[(P.model == m) & (P.condition == "agents_rag_ph") & (P.row == i)]
        M_ = json.loads(pr.matches.iloc[0]) if len(pr) and done(pr.matches.iloc[0]) else []
        ref = build_ref(R.loc[i], args.gt_source)
        for g, it in enumerate(items):
            if not it.get("ph_tags"):
                continue
            hit = [ref[r]["text"] for r, gg in M_ if gg == g and r < len(ref)]
            k = "matched" if hit else "unmatched"
            other = picks["unmatched" if k == "matched" else "matched"]
            if picks[k] is None and (other is None or other[0] != i or not args.distinct_papers):
                picks[k] = (i, g, it, hit)
        if all(picks.values()):
            break
    for k, v in picks.items():
        if v is None:
            o.r(f"  no {k} final limitation with a public-data tag found")
            continue
        i, g, it, hit = v
        tr = json.loads(df.at[i, f"lim_{m}_agents_rag_ph__trace"]) if done(df.at[i, f"lim_{m}_agents_rag_ph__trace"]) else {}
        cands = tr.get("candidates", [])
        toks = set(norm(it["limitation"]).split())
        mc = [c for c in cands if c.get("agent") in it.get("agents", []) and src_of(c) == "manuscript" and c.get("anchored")]
        mc.sort(key=lambda c: -len(toks & set(norm(c.get("claim", "")).split())))
        pc = [c for c in cands if c.get("agent") in it.get("agents", []) and src_of(c) == "public data"]
        pc.sort(key=lambda c: -len(toks & set(norm(c.get("claim", "")).split())))
        cand = pc[0] if pc else (mc[0] if mc else None)
        span = mc[0]["evidence_span"] if mc else ""
        text = str(S.at[i, "full_text_excl_lim"]) if S is not None else ""
        off = norm(text).find(norm(span)[:60]) if span else -1
        raw_off = text.lower().find(span.strip()[:40].lower()) if span else -1
        prov = {p["tag"]: p for p in json.loads(S.at[i, "ph_provenance_json"])} if S is not None and done(S.at[i, "ph_provenance_json"]) else {}
        tag = (it.get("ph_tags") or [None])[0]
        p = prov.get(tag, {})
        o.r(f"\n  [{k}] row {i}  pmid {S.at[i, 'pmid'] if S is not None else '?'}  model {m}  final item G{g + 1}")
        o.r(f"   final: {it['limitation']}\n   source ids {it.get('source_ids')}  agents {it.get('agents')}  tags {it.get('ph_tags')}")
        o.r(f"   candidate: {cand.get('agent') if cand else '-'}: {cand.get('claim') if cand else '-'}  "
            f"gate: {'passed' if cand and cand.get('anchored') else 'failed'} by "
            f"{'tag resolution' if cand and src_of(cand) == 'public data' else 'span location'}")
        o.r(f"   manuscript span: {span[:200]!r}  char offset {raw_off if raw_off >= 0 else 'not found verbatim'}")
        o.r(f"   [{tag}] {p.get('text', '')}\n   file {p.get('file')}  sha256 {str(p.get('sha256'))[:12]}  release {p.get('release')}")
        o.r(f"   filter {p.get('filter')}  rows {p.get('row_indices', [])[:10]}  statistic {p.get('statistic')}")
        o.r(f"   judge: {'matched author unit: ' + hit[0] if hit else 'unmatched'}")
        o.r(f"   study setting: states [{S.at[i, 'ph_states'] if S is not None else ''}]  notes {str(S.at[i, 'ph_notes'])[:200] if S is not None else ''}")
        o.t(f"% (c) Table trace, {k} example (row {i}); the Assessment row must be written by you")
        rows = [("Paper", f"PMID {S.at[i, 'pmid'] if S is not None else '?'}; {m}"),
                ("Manuscript span", (f"``{span[:160]}'' (offset {raw_off})" if span and raw_off >= 0 else (f"``{span[:160]}''" if span else "none (public-data evidence only)"))),
                ("Candidate", f"{cand.get('agent') if cand else '-'}: {(cand or {}).get('claim', '')[:160]}; gate passed by "
                              f"{'tag resolution' if cand and src_of(cand) == 'public data' else 'span location'}"),
                ("Final limitation", f"{it['limitation'][:200]}; source IDs {', '.join(it.get('source_ids') or [])}"),
                ("Public-data tag", f"[{tag}] {p.get('text', '')[:200]}"),
                ("Source file", f"{os.path.basename(str(p.get('file', '')))}; SHA-256 {str(p.get('sha256', ''))[:12]}; release "
                                f"{'; '.join(f'{a} ' + ', '.join(map(str, b if isinstance(b, list) else [b])) for a, b in (p.get('release') or {}).items()) or 'n/a'}"),
                ("Filter and rows", f"{p.get('filter', '')}; rows {', '.join(map(str, (p.get('row_indices') or [])[:6]))}"
                                    f"{' ...' if len(p.get('row_indices') or []) > 6 else ''}; {p.get('statistic', '')}"),
                ("Judge outcome", f"matched: {hit[0][:160]}" if hit else "unmatched")]
        for a, b in rows:
            o.t(f"{a} & {tex(b, 260)} \\\\")
        o.t("Assessment & \\fix{one sentence: does the record support the claim?} \\\\")


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.run_dir, "paper_todos")
    os.makedirs(out_dir, exist_ok=True)
    models = [m for m in args.models.split(",") if m]
    rd = args.run_dir
    S = pd.read_csv(os.path.join(rd, "sample.csv"), low_memory=False) if os.path.exists(os.path.join(rd, "sample.csv")) else None
    if S is not None:
        S["pmid"] = S["pmid"].astype(str).str.replace(r"\.0$", "", regex=True)
    M = {m: pd.read_csv(os.path.join(rd, f"model_{m}.csv"), low_memory=False) for m in models if os.path.exists(os.path.join(rd, f"model_{m}.csv"))}
    R = pd.read_csv(os.path.join(rd, "reference.csv"), low_memory=False) if os.path.exists(os.path.join(rd, "reference.csv")) else None
    parts = [pd.read_csv(f) for f in glob.glob(os.path.join(rd, "eval", "*", "per_row.csv"))]
    P = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["model", "condition", "row", "eval_type", "f1", "matches"])
    P = P[P.eval_type == args.eval_type]
    o = Out()
    o.r(f"run dir {rd}; models {list(M)}; rows evaluated per model: "
        + ", ".join(f"{m} {P[P.model == m].row.nunique()}" for m in models))
    for fn, a in ((item_a, (o, S)), (item_b, (o, args.web_dirs)), (item_c, (o, S, M, R, P, args)),
                  (item_e, (o, S, M, args)), (item_f, (o, args, M)), (item_g, (o, P, models, args.n_rows)),
                  (item_h, (o, S, M, args))):
        try:
            fn(*a)
        except Exception as e:  # noqa: BLE001
            o.r(f"  !! {fn.__name__} failed: {type(e).__name__}: {e}")
    open(os.path.join(out_dir, "todo_report.txt"), "w").write("\n".join(o.rep) + "\n")
    open(os.path.join(out_dir, "latex_snippets.tex"), "w").write("\n".join(o.tex) + "\n")
    print("\n".join(o.rep))
    print(f"\nwrote {out_dir}/todo_report.txt and {out_dir}/latex_snippets.tex")


if __name__ == "__main__":
    main()
