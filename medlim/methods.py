"""
Generation methods: zero-shot (+RAG, +public-health data), pseudo-reference
generation, and the 11-agent PolicyGuard pipeline (workers -> evidence gate ->
leader dedup -> merger with an output budget of k items).
"""
from __future__ import annotations

import asyncio
import json
import re
import time

from . import prompts as P
from .units import parse_limitations, extract_json, clean_unit

BOILERPLATE = ["future work", "more research is needed", "further studies should", "further research is needed"]
SEV = {"low": 0, "medium": 1, "high": 2, "critical": 3}


async def gen_list(llm, system, user, max_tokens=1500):
    """One call; if nothing parseable comes back, one retry asking for 'Limitation N:' lines."""
    t0 = time.time()
    raw = await llm.chat(system, user, max_tokens=max_tokens)
    items, mode = parse_limitations(raw)
    retried = False
    if not items and mode == "empty":
        retried = True
        raw2 = await llm.chat(system, user + "\n\n" + P.FALLBACK_RULE, max_tokens=max_tokens, json_mode=False)
        items, mode = parse_limitations(raw2)
        raw = raw + "\n\n=== RETRY ===\n" + raw2
    return items, {"parse_mode": mode, "retried": retried, "seconds": round(time.time() - t0, 1),
                   "raw_chars": len(raw)}, raw


async def run_zero_shot(llm, title, text, k, passages=None, ph_text="", max_tokens=1500):
    items, meta, raw = await gen_list(llm, P.ZS_SYSTEM, P.zs_user(title, text, k, passages, ph_text), max_tokens)
    items = _tag(items[:k], ph_text)
    meta["n"] = len(items)
    return items, meta, raw


async def run_pseudo(llm, title, text, author_lims, k, ph_text="", passages=None, max_tokens=1500):
    items, meta, raw = await gen_list(llm, P.PSEUDO_SYSTEM, P.pseudo_user(title, text, author_lims, k, ph_text,
                                                                          passages), max_tokens)
    items = _tag(items[:k], ph_text)
    meta["n"] = len(items)
    return items, meta, raw


def _tag(items, ph_text):
    """record which [PH-n] lines a zero-shot item cites (only tags that were actually shown)."""
    shown = set(re.findall(r"PH-\d+", ph_text or ""))
    for it in items:
        it["ph_tags"] = sorted(set(re.findall(r"PH-\d+", json.dumps(it))) & shown)
    return items


# ----------------------------------------------------------------------------- agents
def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()


def _shingles(s, n=5):
    w = _norm(s).split()
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))} if w else set()


def anchored(span, source_text, thr=0.5):
    """Share of the span's 5-word shingles that occur in the source (tolerates small edits)."""
    sh = _shingles(span)
    if not sh or not source_text:
        return False
    src = " " + _norm(source_text) + " "
    return sum(1 for x in sh if f" {x} " in src) / len(sh) >= thr


def evidence_gate(cands, paper_text, passages, ph_text, ph_tags):
    keep, dropped = [], []
    corpus = "\n".join(passages or [])
    for c in cands:
        claim, span = c.get("claim", ""), c.get("evidence_span", "") or ""
        src = str(c.get("evidence_source", "manuscript")).lower()
        tags = [t for t in (c.get("ph_tags") or []) if isinstance(t, str)]
        tags += re.findall(r"PH-\d+", span)
        if any(b in claim.lower() for b in BOILERPLATE) or len(claim.split()) < 4:
            dropped.append((c, "boilerplate/short"))
            continue
        ok = False
        if "public" in src or "data" in src:
            ok = bool(ph_text) and (any(t in ph_tags for t in tags) or anchored(span, ph_text))
        elif "corpus" in src or "reference" in src or "literature" in src:
            ok = anchored(span, corpus) or anchored(span, paper_text)
        else:
            ok = anchored(span, paper_text) or (bool(tags) and any(t in ph_tags for t in tags))
        c["anchored"] = ok
        c["ph_tags"] = sorted(set(t for t in tags if t in ph_tags))
        (keep if ok else dropped).append(c if ok else (c, "not anchored"))
    return keep, dropped


def leader_dedup(cands, thr=0.6):
    out = []
    for c in cands:
        a = set(_norm(c["claim"]).split())
        dup = None
        for o in out:
            b = set(_norm(o["claim"]).split())
            if a and b and len(a & b) / len(a | b) >= thr:
                dup = o
                break
        if dup is None:
            c.setdefault("support", 1)
            out.append(c)
        else:
            dup["support"] = dup.get("support", 1) + 1
            dup["severity"] = max(dup.get("severity", "medium"), c.get("severity", "medium"), key=lambda s: SEV.get(s, 1))
    return out


def _rank(c):
    return (c.get("anchored", False), c.get("support", 1), SEV.get(str(c.get("severity", "medium")).lower(), 1),
            float(c.get("confidence", 0.5) or 0.5))


async def _one_agent(llm, name, dim, system, user, max_tokens):
    raw = await llm.chat(system, user, max_tokens=max_tokens)
    items, mode = parse_limitations(raw)
    out = []
    for it in items[:3]:
        out.append({"agent": name, "dimension": it.get("dimension") or dim, "claim": it["limitation"],
                    "evidence_span": str(it.get("evidence_span") or it.get("evidence") or ""),
                    "evidence_source": str(it.get("evidence_source") or "manuscript"),
                    "ph_tags": it.get("ph_tags") if isinstance(it.get("ph_tags"), list) else [],
                    "affected_subgroup": it.get("affected_subgroup", "any"),
                    "severity": str(it.get("severity", "medium")).lower(),
                    "confidence": it.get("confidence", 0.6)})
    return out, mode


async def run_agents(llm, title, text, k, passages=None, ph_text="", ph_prov=None, merger=None,
                     max_tokens=1200, gate="strict"):
    t0 = time.time()
    merger = merger or llm
    ph_tags = {p["tag"] for p in (ph_prov or [])}
    jobs = []
    for name, dim, uses_ph, system in P.AGENTS:
        user = P.agent_user(title, text, system, passages, ph_text if uses_ph else "")
        jobs.append(_one_agent(llm, name, dim, P.AGENT_COMMON_SYSTEM, user, max_tokens))
    res = await asyncio.gather(*jobs, return_exceptions=True)
    cands, modes, errors = [], {}, 0
    for (name, *_), r in zip(P.AGENTS, res):
        if isinstance(r, Exception):
            errors += 1
            continue
        cands.extend(r[0])
        modes[name] = r[1]
    n_emitted = len(cands)
    if gate == "off":
        kept, dropped = [dict(c, anchored=None) for c in cands], []
    else:
        kept, dropped = evidence_gate(cands, text, passages, ph_text, ph_tags)
    gate_fallback = False
    if not kept and cands and gate != "strict-empty":
        # the model emitted critiques but none could be anchored: keep them (flagged) rather than an empty output
        kept, gate_fallback = [c if isinstance(c, dict) else c for c in cands], True
        for c in kept:
            c.setdefault("anchored", False)
    led = leader_dedup(sorted(kept, key=_rank, reverse=True))
    final, merged_ok = [], False
    if len(led) > 1:
        raw = await merger.chat(P.MERGE_SYSTEM, P.merge_user(title, led, k), max_tokens=1200)
        obj = extract_json(raw)
        lst = obj.get("limitations") if isinstance(obj, dict) else obj if isinstance(obj, list) else None
        if isinstance(lst, list):
            for it in lst[:k]:
                claim = clean_unit(it.get("claim", "") if isinstance(it, dict) else str(it))
                if len(claim.split()) < 4:
                    continue
                ids = [int(x[1:]) - 1 for x in (it.get("source_ids", []) if isinstance(it, dict) else [])
                       if isinstance(x, str) and re.fullmatch(r"C\d+", x) and 0 < int(x[1:]) <= len(led)]
                src = [led[i] for i in ids]
                best = max(src, key=_rank) if src else None
                final.append({"limitation": claim, "source_ids": [f"C{i + 1}" for i in ids],
                              "dimension": best["dimension"] if best else None,
                              "agents": sorted({s["agent"] for s in src}),
                              "evidence_span": best["evidence_span"] if best else "",
                              "evidence_source": best["evidence_source"] if best else "",
                              "ph_tags": sorted({t for s in src for t in s.get("ph_tags", [])}),
                              "severity": best["severity"] if best else None,
                              "anchored": bool(best and best.get("anchored"))})
            merged_ok = bool(final)
    if not final:
        for c in led[:k]:
            final.append({"limitation": c["claim"], "dimension": c["dimension"], "agents": [c["agent"]],
                          "evidence_span": c["evidence_span"], "evidence_source": c["evidence_source"],
                          "ph_tags": c.get("ph_tags", []), "severity": c["severity"],
                          "anchored": c.get("anchored")})
    meta = {"n": len(final), "n_emitted": n_emitted, "n_after_gate": len(kept) if not gate_fallback else 0,
            "n_after_leader": len(led), "merged_by_llm": merged_ok, "gate_fallback": gate_fallback,
            "agent_errors": errors, "agent_parse_modes": modes, "seconds": round(time.time() - t0, 1),
            "parse_mode": "agents"}
    trace = {"candidates": cands, "dropped_by_gate": [{"claim": d[0].get("claim"), "why": d[1]} for d in dropped]}
    return final, meta, json.dumps(trace, ensure_ascii=False, default=str)
