#!/usr/bin/env python
"""
Regex extraction of AUTHOR-MENTIONED LIMITATIONS from PubMed OA full papers (*.csv.gz).

Your rule, plus a few safeguards
--------------------------------
1. The end matter is cut off first: everything from the first stop heading in the second half of the paper
   (Acknowledgements, Appendix, Ethics, Ethical Considerations, + References, Funding, Conflict of interest,
   Competing interests, Author contributions, Data availability, Supplementary, Declarations, Abbreviations)
   is not searched.
2. HEADER method (preferred): a heading such as 'Limitations', 'Study limitations', 'Strengths and
   limitations', '4.3 Limitations of the study', 'LIMITATIONS:' -> take the text after it up to the next
   heading / stop word.
3. SENTENCE method (your backward rule): search BACKWARD for the last sentence containing 'limitation(s)'
   that reads like a study limitation (strong cues such as 'our study has several limitations', 'a major
   limitation', 'limitations of this study'; field terms like 'nutrient limitation', 'growth limitation',
   'diffusion limitation', 'functional limitation' are ignored). Earlier limitation sentences close to it
   (within --gap sentences) are joined, so the extraction starts at the FIRST sentence of that block.
   From there, sentences are taken up to the end, stopping at a stop word (case-insensitive), at a
   'Conclusion(s)' / 'Future directions' heading, at a paragraph that does not continue the limitations
   (no cue and no 'Second,/Finally,/Another ...' opener), or after --max_sentences sentences.
4. Each paper gets quality signals so you can keep only papers with a good number of limitations later:
   lim_n_mentions, lim_n_cue_sentences, lim_n_points_est (enumerators: first/second/finally/another
   limitation ...), lim_n_words, lim_quality (high/medium/low).

New columns: lim_regex_text, limitations_section (one sentence per line = the units the v2 pipeline
splits on '\\n'), full_text_excl_lim (full_text with the extracted span removed, the input of the
generators), lim_method, lim_quality, lim_n_mentions, lim_n_cue_sentences, lim_n_points_est,
lim_n_sentences, lim_n_words, lim_start_char, lim_end_char, lim_rel_position, lim_stop_reason, source_file

Modes
  check  a few files -> regex_check.txt with N examples per method (inspect the ground truth)
  run    random files until --target papers WITH an extracted limitation are collected
         -> <out_dir>/pubmed_oa_regex_lim_<target>.csv  + regex_report.txt + regex_examples.txt

  python extract_limitations_regex.py --mode check --n_files 2 --n_show 40
  python extract_limitations_regex.py --mode run --target 10000 --max_per_file 400 --workers 16
"""
import argparse
import glob
import gzip
import json
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

csv_field_limit = sys.maxsize
while True:                       # full texts can exceed the default csv field size
    try:
        import csv
        csv.field_size_limit(csv_field_limit)
        break
    except OverflowError:
        csv_field_limit //= 10

IN_DIR = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/sci_lit_synthesis/data_bio_phy_chem/pubmed_oa_full_paper/csv"
OUT_DIR = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/medlimagents/med_limagents_v2/data"

# ----------------------------------------------------------------------------- patterns
USER_STOPS = r"acknowledge?ments?|appendix|appendices|ethics|ethical\s+considerations?"
EXTRA_STOPS = (r"references|bibliography|literature\s+cited|funding(?:\s+(?:information|sources?|statement))?|"
               r"financial\s+(?:support|disclosure)|conflicts?\s+of\s+interests?|competing\s+interests?|"
               r"declaration\s+of\s+(?:competing\s+)?interests?|disclosures?|author\s*(?:s'|'s|s)?\s+contributions?|"
               r"contributions?\s+of\s+authors|data\s+(?:and\s+materials?\s+)?availability(?:\s+statement)?|"
               r"availability\s+of\s+data(?:\s+and\s+materials?)?|supplementary\s+(?:materials?|information|data)|"
               r"supporting\s+information|declarations|abbreviations|consent\s+for\s+publication|"
               r"institutional\s+review\s+board\s+statement|informed\s+consent\s+statement|footnotes|notes")
SECTION_STOPS = r"conclusions?|concluding\s+remarks|summary|future\s+(?:directions?|work|research|perspectives?)|implications"
NUM = r"(?:[IVX]+\.|\d+(?:\.\d+)*\.?)?\s*"

LIM_HEAD = (r"(?:(?:study|potential|main|key|methodological|important)\s+)?(?:strengths?\s*(?:and|&)\s*)?"
            r"(?:weakness(?:es)?\s*(?:and|&)\s*)?limitations?"
            r"(?:\s*(?:and|&)\s*(?:strengths?|future\s+(?:directions?|work|research|perspectives?)|weakness(?:es)?))?"
            r"(?:\s+of\s+(?:the|this|our|present|current)\s+(?:present\s+|current\s+)?(?:study|work|research|analysis|review|trial|investigation|approach|findings))?")

STRONG_CUES = re.compile(
    r"(?:(?:this|our|the\s+present|the\s+current|the)\s+(?:present\s+|current\s+)?(?:study|work|research|analysis|"
    r"review|trial|investigation|survey|findings|results|data|approach|model|method)\s+(?:also\s+)?(?:has|had|have|is|was|"
    r"are|were|has\s+been|is\s+subject\s+to|suffers?\s+from|presents?|faces?)\s+(?:\w+\s+){0,3}limitations?)"
    r"|(?:limitations?\s+(?:of|to|in)\s+(?:the|this|our)\s+(?:present\s+|current\s+)?(?:study|work|research|analysis|"
    r"review|trial|investigation|survey|findings|results|approach))"
    r"|(?:\b(?:a|an|one|the|another|other|further|additional|major|main|key|potential|important|second|third|"
    r"first|final|last|notable|primary|significant|inherent|several|some|few|certain|possible)\s+(?:\w+\s+)?limitations?\s+"
    r"(?:is|was|are|were|of|to|in|that|concerns?|relates?|includes?|lies|stems|should|must|may|might|could|would|"
    r"needs?|deserves?|here|worth))"
    r"|(?:limitations?\s+(?:include|included|are|were|exist|should\s+be|must\s+be|need\s+to\s+be|of\s+note))"
    r"|(?:strengths?\s+and\s+(?:the\s+)?limitations?)|(?:we\s+(?:acknowledge|recognize|note)\s+(?:\w+\s+){0,4}limitations?)"
    r"|(?:(?:is|was|are|were)\s+(?:also\s+)?limited\s+by)|(?:should\s+be\s+interpreted\s+(?:with|in\s+light\s+of)\s+(?:caution|"
    r"(?:the|these|some|several)\s+(?:following\s+)?limitations?))", re.I)
WEAK_CUE = re.compile(r"\blimitations?\b", re.I)
# 'limitation' used as a technical term, not as a study limitation
NEG_CONTEXT = re.compile(
    r"\b(?:nutrient|nitrogen|phosphorus|phosphate|iron|carbon|oxygen|light|substrate|resource|growth|food|water|"
    r"diffusion|mass[-\s]transfer|transport|kinetic|diffraction|resolution|dose|rate|activity|functional|physical|"
    r"mobility|motion|movement|range[-\s]of[-\s]motion|joint|participation|selection|sequence|detection|solubility|"
    r"thermodynamic|current|charge|space[-\s]charge|bandwidth|power|memory)\s+limitations?\b"
    r"|\blimitations?\s+(?:of|in)\s+(?:daily|physical|activity|activities|movement|motion|mobility|function|"
    r"range|growth)", re.I)
ENUM_OPENER = re.compile(
    r"^\s*(?:\(?[ivx\d]+[.)]\s*|first(?:ly)?|second(?:ly)?|third(?:ly)?|fourth(?:ly)?|fifth(?:ly)?|finally|lastly|"
    r"last(?:ly)?,|another|additionally|in\s+addition|moreover|furthermore|also|further(?:more)?|besides|"
    r"other\s+limitations?|a\s+(?:further|final|second|third|last)\s|one\s+(?:other|more))", re.I)
ENUM_WORD = re.compile(r"\b(?:first(?:ly)?|second(?:ly)?|third(?:ly)?|fourth(?:ly)?|fifth(?:ly)?|sixth|finally|"
                       r"lastly|another\s+(?:\w+\s+)?limitation|additional\s+limitation|further\s+limitation|"
                       r"other\s+limitations?)\b", re.I)
DISCUSSION = re.compile(rf"(?:^|\n)\s*{NUM}(?:discussion|general\s+discussion|discussion\s+and\s+conclusions?)\s*[:.]?\s*(?:\n|$)", re.I)

# ---- medical / public-health content filter (title + abstract)
HUMAN_TERMS = re.compile(
    r"\b(?:patients?|participants?|subjects?\s+(?:were|with)|adults?|children|adolescents?|infants?|neonates?|"
    r"women|men|pregnan\w*|elderly|older\s+adults|veterans|residents|individuals|people|population|volunteers|"
    r"cohort|case[-\s]control|cross[-\s]sectional|randomi[sz]ed|clinical\s+trial|trial|survey|questionnaire|"
    r"hospital\w*|clinic\w*|primary\s+care|emergency\s+department|inpatients?|outpatients?|medicare|medicaid|"
    r"nhanes|brfss|electronic\s+health\s+records?|ehr|registry)\b", re.I)
HEALTH_TERMS = re.compile(
    r"\b(?:disease|disorder|syndrome|cancer|tumou?r|carcinoma|diabet\w*|obes\w*|hypertension|stroke|"
    r"cardiovascular|coronary|heart\s+failure|asthma|copd|kidney|renal|hiv|hepatitis|covid[-\s]?19|sars[-\s]cov[-\s]2|"
    r"infection|depress\w*|anxiety|dementia|alzheimer\w*|mental\s+health|smok\w*|alcohol|opioid|substance|"
    r"mortality|morbidity|incidence|prevalence|risk\s+factors?|outcomes?|treatment|therapy|surgery|screening|"
    r"diagnos\w*|prognos\w*|vaccin\w*|health\s+care|healthcare|public\s+health|disparit\w*|equity)\b", re.I)
NONHUMAN = re.compile(r"\b(?:mice|mouse|murine|rats?|zebrafish|drosophila|c\.\s*elegans|in\s+vitro|cell\s+lines?|"
                      r"yeast|arabidopsis|bacteri\w+\s+strains?|plants?|soil|crystal|catalys\w*|polymer|alloy|"
                      r"semiconductor|quantum|galax\w*|spectroscop\w*)\b", re.I)
NO_LIM_TYPES = r"erratum|errata|correction|corrigendum|retraction|editorial|letter|comment|reply|obituary|news|book\s+review"


def medical_score(title, abstract):
    t = f"{title or ''} {abstract or ''}"
    h, d, n = len(HUMAN_TERMS.findall(t)), len(HEALTH_TERMS.findall(t)), len(NONHUMAN.findall(t))
    return h, d, n


ABBREV = re.compile(r"\b(?:et al|e\.g|i\.e|cf|vs|Fig|Figs|Tab|Eq|Eqs|Ref|Refs|No|approx|ca|Dr|Mr|Ms|Mrs|Prof|St|"
                    r"Inc|Ltd|Co|Corp|Jr|Sr|vol|pp|resp|al|min|max|sec|hr|wk|mo|yr|spp|sp|var|subsp)\.", re.I)


def norm(text):
    if not isinstance(text, str):
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n").replace("\xa0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n{3,}", "\n\n", t)


STOP_RE = re.compile(rf"(?:(?:^|\n)[ \t]*{NUM}(?:{USER_STOPS}|{EXTRA_STOPS})\b[^\n]{{0,60}}(?=\n|$)"
                     rf"|(?:^|\n|(?<=[.!?\]\)])[ \t])[ \t]*{NUM}(?:{USER_STOPS}|{EXTRA_STOPS})\b[ \t]*[:.]?[ \t]*(?=[A-Z0-9]))", re.I)
USER_STOP_ANY = re.compile(rf"\b(?:{USER_STOPS})\b", re.I)          # your rule: these words stop extraction anywhere
SECTION_RE = re.compile(rf"(?:(?:^|\n)[ \t]*{NUM}(?:{SECTION_STOPS})\b[^\n]{{0,40}}(?=\n|$)"
                        rf"|(?:^|\n|(?<=[.!?])[ \t])[ \t]*{NUM}(?:{SECTION_STOPS})[ \t]*:[ \t]*)", re.I)
LIM_HEAD_RE = re.compile(rf"(?:(?:^|\n)[ \t]*{NUM}(?:{LIM_HEAD})[ \t]*[:.]?[ \t]*(?=\n|$)"
                         rf"|(?:^|\n|(?<=[.!?])[ \t])[ \t]*{NUM}(?:{LIM_HEAD})[ \t]*:[ \t]*)", re.I)
GENERIC_HEADING = re.compile(r"\n[ \t]*(?:[IVX]+\.|\d+(?:\.\d+)*\.?)?[ \t]*[A-Z][^\n.!?]{2,70}[ \t]*\n")


def sentences(text, offset=0):
    """[(start, end, sentence)] with abbreviation protection; paragraph breaks are sentence breaks."""
    prot = ABBREV.sub(lambda m: m.group(0).replace(".", "\x00"), text)
    prot = re.sub(r"(\d)\.(\d)", "\\1\x00\\2", prot)
    out = []
    for m in re.finditer(r"[^.!?\n]+(?:[.!?]+[\"'\)\]]*(?:\s*\[\d[\d,\s\-\u2013]*\])?|(?=\n)|$)", prot):
        s = m.group(0)
        if s.strip():
            st = m.start() + (len(s) - len(s.lstrip()))
            out.append((offset + st, offset + m.end(), text[st:m.end()].strip()))
    return out


def cue_level(sent):
    if NEG_CONTEXT.search(sent) and not STRONG_CUES.search(NEG_CONTEXT.sub(" ", sent)):
        return 0
    if STRONG_CUES.search(sent):
        return 2
    return 1 if WEAK_CUE.search(sent) else 0


def end_matter_start(t):
    """first stop heading in the second half of the paper (start of the end matter)."""
    half = len(t) // 2
    for m in STOP_RE.finditer(t):
        if m.start() >= half:
            return m.start()
    return len(t)


def first_stop_after(t, pos, end, stops_anywhere):
    """earliest of: stop heading, conclusion/future-work heading, (your rule) a stop word anywhere, body end."""
    cands = [(end, "end_matter")]
    m = STOP_RE.search(t, pos, end)
    if m:
        cands.append((m.start(), "stop_heading"))
    m = SECTION_RE.search(t, pos, end)
    if m:
        cands.append((m.start(), "section_heading"))
    if stops_anywhere:
        m = USER_STOP_ANY.search(t, pos, end)
        if m:
            cands.append((m.start(), "stop_word"))
    return min(cands)


def extract(full_text, gap=4, max_sentences=25, max_words=1500, min_words=12, stops_anywhere=True):
    t = norm(full_text)
    res = {"lim_regex_text": "", "limitations_section": "", "lim_method": "none", "lim_start_char": -1,
           "lim_end_char": -1, "lim_stop_reason": "", "_t": t}
    if len(t) < 500:
        return res
    body_end = end_matter_start(t)

    # ---------------- 1. header method (last limitation heading in the body)
    heads = [m for m in LIM_HEAD_RE.finditer(t, 0, body_end) if m.start() > len(t) * 0.2]
    if heads:
        h = heads[-1]
        start = h.end()
        stop, reason = first_stop_after(t, start, body_end, stops_anywhere)
        g = GENERIC_HEADING.search(t, start + 40, stop)        # next generic heading line
        if g and g.start() < stop and not LIM_HEAD_RE.match(t, g.start()):
            stop, reason = g.start(), "next_heading"
        sents = [s for s in sentences(t[start:stop], start)]
        if len(sents) > max_sentences:
            sents, reason = sents[:max_sentences], "max_sentences"
        txt = " ".join(s[2] for s in sents)
        if len(txt.split()) >= min_words:
            res.update(lim_method="header", lim_start_char=h.start(), lim_end_char=sents[-1][1] if sents else stop,
                       lim_stop_reason=reason, sents=sents)
            return finish(res, max_words)

    # ---------------- 2. sentence method (backward search)
    disc = [m.end() for m in DISCUSSION.finditer(t, 0, body_end)]
    search_from = disc[-1] if disc else int(body_end * 0.3)
    S = sentences(t[search_from:body_end], search_from)
    if not S:
        return res
    lv = [cue_level(s[2]) for s in S]
    strong = [k for k, v in enumerate(lv) if v == 2]
    weak = [k for k, v in enumerate(lv) if v >= 1]
    if strong:
        anchor, method = strong[-1], "sentence_strong"
    elif weak:
        anchor, method = weak[-1], "sentence_weak"
    else:
        return res
    first = anchor
    k = anchor - 1
    while k >= 0 and first - k <= gap:                 # extend backward over the same limitation block
        if lv[k] >= 1:
            first = k
        k -= 1
    stop, reason = first_stop_after(t, S[first][0], body_end, stops_anywhere)
    take = []
    for j in range(first, len(S)):
        st, en, s = S[j]
        if st >= stop:
            break
        if take:
            between = t[S[j - 1][1]:st]
            if "\n\n" in between or (between.count("\n") >= 1 and "\n\n" not in t):
                # new paragraph: continue only if it still talks about limitations
                if lv[j] == 0 and not ENUM_OPENER.match(s):
                    reason = "paragraph_end"
                    break
        take.append(S[j])
        if len(take) >= max_sentences:
            reason = "max_sentences"
            break
    if not take or len(" ".join(x[2] for x in take).split()) < min_words:
        return res
    res.update(lim_method=method, lim_start_char=take[0][0], lim_end_char=take[-1][1], lim_stop_reason=reason,
               sents=take)
    return finish(res, max_words)


def finish(res, max_words):
    sents = res.pop("sents")
    words = 0
    keep = []
    for s in sents:
        words += len(s[2].split())
        keep.append(s)
        if words > max_words:
            res["lim_stop_reason"] = "max_words"
            break
    res["lim_end_char"] = keep[-1][1]
    res["lim_regex_text"] = " ".join(s[2] for s in keep)
    res["limitations_section"] = "\n".join(s[2] for s in keep)
    return res


def quality(r, n_text):
    txt = r["lim_regex_text"]
    sents = r["limitations_section"].split("\n") if txt else []
    lv = [cue_level(s) for s in sents]
    r["lim_n_mentions"] = len(WEAK_CUE.findall(txt))
    r["lim_n_cue_sentences"] = sum(1 for v in lv if v >= 1)
    enum = [m.group(0).lower() for m in ENUM_WORD.finditer(txt)]
    # 'First, ... Second, ... Finally' = 3 points; 'X. Another limitation ...' = 2 (the first is unnumbered)
    r["lim_n_points_est"] = (0 if not txt else 1 if not enum else len(enum) if enum[0].startswith("first")
                             else len(enum) + 1)
    r["lim_n_sentences"] = len(sents)
    r["lim_n_words"] = len(txt.split())
    r["lim_rel_position"] = round(r["lim_start_char"] / max(1, n_text), 3) if txt else -1
    if not txt:
        q = "none"
    elif (r["lim_method"] in ("header", "sentence_strong") and r["lim_n_sentences"] >= 2
          and 30 <= r["lim_n_words"] <= 1200 and r["lim_rel_position"] >= 0.4):
        q = "high"
    elif r["lim_method"] != "sentence_weak" or r["lim_n_cue_sentences"] >= 2:
        q = "medium"
    else:
        q = "low"
    r["lim_quality"] = q
    return r


def _process_chunk(df, args, st):
    """filters + extraction for one chunk of rows; returns the kept rows (or None)."""
    st["rows"] += len(df)
    df = df[df["full_text"].astype(str).str.len() > 2000]
    subj = df["subjects"].astype(str) if "subjects" in df.columns else pd.Series([""] * len(df), index=df.index)
    excl = NO_LIM_TYPES + (f"|{args.exclude_subjects}" if args.exclude_subjects else "")
    bad_type = subj.str.contains(excl, case=False, regex=True, na=False)
    st["dropped_article_type"] += int(bad_type.sum())
    df, subj = df[~bad_type], subj[~bad_type]
    if df.empty:
        return None
    sc = [medical_score(a, b) for a, b in zip(df.get("title", ""), df.get("abstract", ""))]
    df = df.assign(med_human_hits=[x[0] for x in sc], med_health_hits=[x[1] for x in sc],
                   med_nonhuman_hits=[x[2] for x in sc])
    content_ok = ((df.med_human_hits >= args.min_human_hits) & (df.med_health_hits >= args.min_health_hits)
                  & (df.med_nonhuman_hits <= df.med_human_hits))
    subj_ok = (subj.str.contains(args.subjects_regex, case=False, regex=True, na=False)
               if args.subjects_regex else pd.Series(False, index=df.index))
    df = df.assign(med_subject_match=subj_ok.values, med_content_match=content_ok.values)
    rule = {"none": pd.Series(True, index=df.index), "subjects": subj_ok, "content": content_ok,
            "either": subj_ok | content_ok, "both": subj_ok & content_ok}[args.medical_filter]
    df = df[rule.values]
    st["kept_for_extraction"] += len(df)
    if df.empty:
        return None
    recs = []
    for ft in df["full_text"]:
        r = extract(ft, args.gap, args.max_sentences, args.max_words, args.min_words, not args.no_stop_anywhere)
        t = r.pop("_t")
        r = quality(r, len(t))
        r["full_text_excl_lim"] = ((t[:r["lim_start_char"]] + "\n" + t[r["lim_end_char"]:]).strip()
                                   if r["lim_regex_text"] else t)
        recs.append(r)
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(recs).reset_index(drop=True)], axis=1)
    st["scanned"] += len(out)
    for k, v in Counter(out["lim_method"]).items():
        st["methods"][k] = st["methods"].get(k, 0) + v
    st["with_limitation"] += int((out["lim_method"] != "none").sum())
    if not args.keep_without_lim and args.mode == "run":
        out = out[out["lim_method"] != "none"]          # keep only what will be saved
    return out if len(out) else None


def process_file(path, args):
    """Reads the .csv.gz in chunks of --chunk_rows rows, so memory stays small even for the large
    baseline files (some have 50,000-80,000 papers)."""
    st = {"rows": 0, "dropped_article_type": 0, "kept_for_extraction": 0, "with_limitation": 0, "scanned": 0,
          "methods": {}}
    drop = [c.strip() for c in (args.drop_cols or "").split(",") if c.strip()]
    try:
        cols = pd.read_csv(path, compression="gzip", nrows=0).columns.tolist()
        if "full_text" not in cols:
            return path, None, {"error": "no full_text column"}
        use = [c for c in cols if c not in drop]
        parts = []
        reader = pd.read_csv(path, compression="gzip", usecols=use, chunksize=args.chunk_rows,
                             engine="python" if args.python_engine else "c", on_bad_lines="skip")
        for chunk in reader:
            out = _process_chunk(chunk, args, st)
            if out is not None:
                parts.append(out)
    except Exception as e:  # noqa: BLE001
        return path, None, {"error": f"{type(e).__name__}: {e}", **st}
    if not parts:
        return path, None, st
    out = pd.concat(parts, ignore_index=True)
    out["source_file"] = os.path.basename(path)
    if args.mode == "run" and args.max_per_file and len(out) > args.max_per_file:
        # random cap inside the worker, so at most max_per_file rows travel back to the main process
        out = out.sample(args.max_per_file, random_state=args.seed).sort_index()
    return path, out, st


# ----------------------------------------------------------------------------- reporting
def example_block(r, i, show_chars=1800):
    return (f"\n{'#' * 110}\n# [{i}] pmid={r.get('pmid')} pmc={r.get('pmc')} file={r.get('source_file')}\n"
            f"# {str(r.get('title'))[:140]}\n# method={r['lim_method']} quality={r['lim_quality']} "
            f"words={r['lim_n_words']} sentences={r['lim_n_sentences']} mentions={r['lim_n_mentions']} "
            f"points_est={r['lim_n_points_est']} pos={r['lim_rel_position']} stop={r['lim_stop_reason']}\n{'#' * 110}\n"
            + "\n".join(f"   L{k + 1}: {s}" for k, s in enumerate(str(r['limitations_section']).split("\n")))[:show_chars])


def report(df_all_stats, df, args, path_txt, scanned_rows):
    L = []
    W = L.append
    W("=" * 110)
    W(f" REGEX LIMITATION EXTRACTION   {time.strftime('%Y-%m-%d %H:%M')}")
    W(f" input dir : {args.in_dir}")
    W(f" medical filter: {args.medical_filter}  subjects_regex={args.subjects_regex}  min_human_hits="
      f"{args.min_human_hits} min_health_hits={args.min_health_hits}  exclude={args.exclude_subjects}")
    W(f" files read: {len(df_all_stats)}; papers scanned: {scanned_rows:,}; params gap={args.gap} "
      f"max_sentences={args.max_sentences} max_words={args.max_words} min_words={args.min_words}")
    W("=" * 110)
    tot_k = sum(s.get("kept_for_extraction", 0) for s in df_all_stats.values())
    tot_l = sum(s.get("with_limitation", 0) for s in df_all_stats.values())
    W(f" papers with an extracted limitation: {tot_l:,} of {tot_k:,} ({tot_l / max(1, tot_k) * 100:.1f}%)")
    W(f" saved papers: {len(df):,}")
    W("\n method       : " + ", ".join(f"{k}: {v}" for k, v in Counter(df['lim_method']).most_common()))
    W(" quality      : " + ", ".join(f"{k}: {v}" for k, v in Counter(df['lim_quality']).most_common()))
    W(" stop reason  : " + ", ".join(f"{k}: {v}" for k, v in Counter(df['lim_stop_reason']).most_common()))
    for c in ("lim_n_words", "lim_n_sentences", "lim_n_mentions", "lim_n_cue_sentences", "lim_n_points_est",
              "lim_rel_position"):
        x = df[c].astype(float)
        W(f" {c:<20}: mean {x.mean():.2f}  median {x.median():.1f}  p10 {x.quantile(.1):.1f}  p90 {x.quantile(.9):.1f}  max {x.max():.0f}")
    W("\n suggested filters for 'a good number of limitations' (counts in the saved set):")
    for name, m in [("quality == high", df.lim_quality == "high"),
                    ("quality in high/medium", df.lim_quality.isin(["high", "medium"])),
                    ("cue sentences >= 2", df.lim_n_cue_sentences >= 2),
                    ("points_est >= 2", df.lim_n_points_est >= 2),
                    ("points_est >= 3", df.lim_n_points_est >= 3),
                    ("high & sentences 3-20 & points >= 2",
                     (df.lim_quality == "high") & df.lim_n_sentences.between(3, 20) & (df.lim_n_points_est >= 2))]:
        W(f"   {name:<40} {int(m.sum()):>6,}")
    if "publication_year" in df.columns:
        W(" publication years: " + ", ".join(f"{k}:{v}" for k, v in sorted(Counter(
            pd.to_numeric(df.publication_year, errors='coerce').dropna().astype(int)).items())[-12:]))
    if "subjects" in df.columns:
        sj = df["subjects"].astype(str).str.split(r"\s*;\s*").explode().str.strip().str.strip("'\"")
        W(" top subject labels (saved set): " + "; ".join(f"{k} ({v})" for k, v in sj.value_counts().head(30).items()))
    if "med_content_match" in df.columns:
        W(f" medical content match: {int(df.med_content_match.sum()):,}; subject match: {int(df.med_subject_match.sum()):,}")
    W(f" dropped as non-research article types: {sum(s.get('dropped_article_type', 0) for s in df_all_stats.values()):,}")
    W(" files used: " + ", ".join(f"{os.path.basename(p)}({s.get('with_limitation', 0)})" for p, s in df_all_stats.items()))
    W("=" * 110)
    txt = "\n".join(L) + "\n"
    open(path_txt, "w").write(txt)
    print(txt)


def write_examples(df, path, n, seed):
    L = [f"EXAMPLES ({n} per method, random, seed {seed})"]
    for m in ("header", "sentence_strong", "sentence_weak"):
        sub = df[df.lim_method == m]
        if len(sub):
            L.append(f"\n\n{'=' * 110}\nMETHOD {m}: {len(sub)} papers\n{'=' * 110}")
            for i, (_, r) in enumerate(sub.sample(min(n, len(sub)), random_state=seed).iterrows()):
                L.append(example_block(r, i + 1))
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="check", choices=["check", "run"])
    ap.add_argument("--in_dir", default=IN_DIR)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--target", type=int, default=10000, help="papers WITH an extracted limitation to save")
    ap.add_argument("--keep_without_lim", action="store_true", help="also save papers where nothing was found")
    ap.add_argument("--max_per_file", type=int, default=400, help="cap per file so the sample spreads over many files")
    ap.add_argument("--n_files", type=int, default=2, help="check mode: files to read")
    ap.add_argument("--n_show", type=int, default=30, help="examples per method in the txt")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gap", type=int, default=4)
    ap.add_argument("--max_sentences", type=int, default=25)
    ap.add_argument("--max_words", type=int, default=1500)
    ap.add_argument("--min_words", type=int, default=12)
    ap.add_argument("--no_stop_anywhere", action="store_true",
                    help="stop words only count as headings (default: your rule, the word anywhere stops)")
    ap.add_argument("--subjects_regex", default=None, help="specialty regex on 'subjects', e.g. 'oncolog|endocrin|cardio'")
    ap.add_argument("--medical_filter", default="none", choices=["none", "subjects", "content", "either", "both"],
                    help="none = all papers; subjects = --subjects_regex only; content = title/abstract describe a "
                         "human health study; either / both = combine the two")
    ap.add_argument("--min_human_hits", type=int, default=1, help="content filter: human-study terms needed")
    ap.add_argument("--min_health_hits", type=int, default=2, help="content filter: health terms needed")
    ap.add_argument("--exclude_subjects", default=None, help="extra article types to drop, e.g. 'review|protocol'")
    ap.add_argument("--out_format", default="csv", choices=["csv", "csv.gz", "parquet"])
    ap.add_argument("--chunk_rows", type=int, default=1000, help="rows read at a time from each .csv.gz")
    ap.add_argument("--drop_cols", default="references", help="columns not read (saves memory); default 'references'; '' keeps all")
    ap.add_argument("--python_engine", action="store_true", help="slower csv parser for malformed files")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.in_dir, "*.csv.gz")))
    if not files:
        raise SystemExit(f"no *.csv.gz in {args.in_dir}")
    random.Random(args.seed).shuffle(files)
    print(f"{len(files)} files in {args.in_dir}; mode={args.mode}", flush=True)

    if args.mode == "check":
        files = files[:args.n_files]
    stats, parts, scanned, n_lim = {}, [], 0, 0
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        pending, path_of = {}, {}
        it = iter(files)
        for _ in range(args.workers):
            p = next(it, None)
            if p:
                pending[ex.submit(process_file, p, args)] = p
        while pending:
            fut = next(as_completed(pending))
            path_of[fut] = pending.pop(fut)
            path = path_of[fut]
            try:
                path, out, st = fut.result()
            except Exception as e:  # noqa: BLE001  - one bad file must not stop the job
                out, st = None, {"error": f"{type(e).__name__}: {e}"}
            stats[path] = st
            if out is not None:
                scanned += st.get("scanned", len(out))
                got = out if args.keep_without_lim else out[out.lim_method != "none"]
                parts.append(got)
                n_lim += int((got.lim_method != "none").sum())
            print(f"[{time.time() - t0:6.0f}s] {os.path.basename(path)}: {st} -> collected {n_lim:,}", flush=True)
            if args.mode == "run" and n_lim >= args.target:
                for f in pending:
                    f.cancel()
                break
            p = next(it, None)
            if p:
                pending[ex.submit(process_file, p, args)] = p
    if not parts:
        raise SystemExit("nothing extracted")
    df = pd.concat(parts, ignore_index=True)
    if "pmid" in df.columns:
        df = df.drop_duplicates(subset=["pmid"])
    if args.mode == "run":
        lim = df[df.lim_method != "none"]
        if len(lim) > args.target:
            lim = lim.sample(args.target, random_state=args.seed)
        df = pd.concat([lim, df[df.lim_method == "none"]]) if args.keep_without_lim else lim
        df = df.reset_index(drop=True)
        base = os.path.join(args.out_dir, f"pubmed_oa_regex_lim_{args.target}")
        path = base + "." + args.out_format
        if args.out_format == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False, compression="gzip" if args.out_format == "csv.gz" else None)
        print(f"saved {len(df):,} papers -> {path}")
        report(stats, df[df.lim_method != "none"], args, os.path.join(args.out_dir, "regex_report.txt"), scanned)
        write_examples(df[df.lim_method != "none"], os.path.join(args.out_dir, "regex_examples.txt"), args.n_show, args.seed)
        print(f"examples -> {os.path.join(args.out_dir, 'regex_examples.txt')}")
    else:
        lim = df[df.lim_method != "none"]
        report(stats, lim, args, os.path.join(args.out_dir, "regex_check_report.txt"), scanned)
        write_examples(lim, os.path.join(args.out_dir, "regex_check.txt"), args.n_show, args.seed)
        none = df[df.lim_method == "none"].head(10)
        with open(os.path.join(args.out_dir, "regex_check.txt"), "a") as f:
            f.write(f"\n\n{'=' * 110}\nPAPERS WITH NOTHING EXTRACTED (first 10 of {int((df.lim_method == 'none').sum())}): "
                    "'limitation' occurrences in their full text, to spot misses\n")
            for _, r in none.iterrows():
                hits = [m.start() for m in WEAK_CUE.finditer(str(r.full_text))]
                f.write(f"\n# pmid={r.get('pmid')} {str(r.get('title'))[:120]} - {len(hits)} occurrence(s)\n")
                for h in hits[:4]:
                    f.write("   ..." + str(r.full_text)[max(0, h - 150):h + 150].replace("\n", " ") + "...\n")
        print(f"inspect: {os.path.join(args.out_dir, 'regex_check.txt')}")


if __name__ == "__main__":
    main()
