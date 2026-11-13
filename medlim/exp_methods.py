"""
Experiment methods (open LLMs):
  run_extract        author-mentioned limitations as atomic units, each with a verbatim quote checked against the text
  run_pseudo         additional limitations with societal impact (pseudo-label candidates)
  run_zero_shot      single-model generation (+RAG / +public-health data)
  run_agents_seq     11 specialist agents run SEQUENTIALLY (each sees what teammates already found)
                     -> evidence gate -> leader/master agent writes the final <= k limitations with societal impact
"""
from __future__ import annotations

import json
import re
import time

from . import prompts_exp as P
from .methods import anchored, gen_list, leader_dedup, _rank, BOILERPLATE
from .units import parse_limitations, extract_json, clean_unit


def _ph_tags(items, ph_text):
    shown = set(re.findall(r"PH-\d+", ph_text or ""))
    for it in items:
        it["ph_tags"] = sorted(set(re.findall(r"PH-\d+", json.dumps(it))) & shown)
    return items


async def run_extract(llm, title, view, k=15, max_tokens=1800, keep_unanchored=False):
    items, meta, raw = await gen_list(llm, P.EXTRACT_SYSTEM, P.extract_user(title, view, k), max_tokens)
    kept, dropped = [], []
    for it in items[:k]:
        q = str(it.get("quote") or "")
        ok = bool(q) and (anchored(q, view, 0.5) or anchored(it["limitation"], view, 0.6))
        it["quote_anchored"] = ok
        (kept if ok or keep_unanchored else dropped).append(it)
    meta.update(n=len(kept), n_raw=len(items), n_unanchored_dropped=len(dropped) if not keep_unanchored else 0)
    return kept, meta, json.dumps({"dropped_unanchored": dropped}, ensure_ascii=False)


async def run_pseudo(llm, title, text, author_lims, k, ph_text="", max_tokens=1800):
    items, meta, raw = await gen_list(llm, P.PSEUDO_SYSTEM, P.pseudo_user(title, text, author_lims, k, ph_text),
                                      max_tokens)
    items = _ph_tags(items[:k], ph_text)
    meta["n"] = len(items)
    return items, meta, raw


async def run_zero_shot(llm, title, text, k, passages=None, ph_text="", max_tokens=1800):
    items, meta, raw = await gen_list(llm, P.ZS_SYSTEM, P.zs_user(title, text, k, passages, ph_text), max_tokens)
    items = _ph_tags(items[:k], ph_text)
    meta["n"] = len(items)
    return items, meta, raw


def _gate(c, paper_text, passages, ph_text, ph_tags):
    span = c.get("evidence_span", "") or ""
    src = str(c.get("evidence_source", "manuscript")).lower()
    tags = [t for t in (c.get("ph_tags") or []) if isinstance(t, str)] + re.findall(r"PH-\d+", span)
    c["ph_tags"] = sorted(set(t for t in tags if t in ph_tags))
    if "public" in src or "data" in src:
        return bool(ph_text) and (bool(c["ph_tags"]) or anchored(span, ph_text))
    if "corpus" in src or "literature" in src:
        return anchored(span, "\n".join(passages or [])) or anchored(span, paper_text)
    return anchored(span, paper_text) or bool(c["ph_tags"])


async def run_agents_seq(llm, title, abstract, text, k, passages=None, ph_text="", ph_prov=None, leader=None,
                         max_tokens=1000, gate="strict", parallel=False):
    t0 = time.time()
    leader = leader or llm
    ph_tags = {p["tag"] for p in (ph_prov or [])}
    cands, modes, errors = [], {}, 0

    async def one(name, dim, uses_ph, role, prior):
        user = P.agent_user(title, text, role, prior, passages, ph_text if uses_ph else "")
        raw = await llm.chat(P.AGENT_COMMON_SYSTEM, user, max_tokens=max_tokens)
        items, mode = parse_limitations(raw)
        out = []
        for it in items[:3]:
            out.append({"agent": name, "dimension": dim, "claim": it["limitation"],
                        "societal_impact": str(it.get("societal_impact") or ""),
                        "affected_population": str(it.get("affected_population") or ""),
                        "evidence_span": str(it.get("evidence_span") or it.get("evidence") or ""),
                        "evidence_source": str(it.get("evidence_source") or "manuscript"),
                        "ph_tags": it.get("ph_tags") if isinstance(it.get("ph_tags"), list) else [],
                        "severity": str(it.get("severity", "medium")).lower(),
                        "confidence": it.get("confidence", 0.6)})
        return out, mode

    if parallel:
        import asyncio
        res = await asyncio.gather(*[one(n, d, u, r, []) for n, d, u, r in P.AGENTS], return_exceptions=True)
        for (n, *_), r in zip(P.AGENTS, res):
            if isinstance(r, Exception):
                errors += 1
                continue
            cands += r[0]
            modes[n] = r[1]
    else:
        for n, d, u, r in P.AGENTS:                     # sequential: each agent sees teammates' findings
            try:
                out, mode = await one(n, d, u, r, cands)
                cands += out
                modes[n] = mode
            except Exception:  # noqa: BLE001
                errors += 1
    n_emitted = len(cands)
    kept, dropped = [], []
    for c in cands:
        if any(b in c["claim"].lower() for b in BOILERPLATE) or len(c["claim"].split()) < 4:
            dropped.append((c, "boilerplate/short"))
            continue
        c["anchored"] = True if gate == "off" else _gate(c, text, passages, ph_text, ph_tags)
        (kept if c["anchored"] else dropped).append(c if c["anchored"] else (c, "not anchored"))
    gate_fallback = False
    if not kept and cands:
        kept, gate_fallback = [c for c in cands if len(c["claim"].split()) >= 4], True
    led = leader_dedup(sorted(kept, key=_rank, reverse=True))
    final = []
    if led:
        raw = await leader.chat(P.LEADER_SYSTEM, P.leader_user(title, abstract or "", led, k, ph_text), max_tokens=1800)
        obj = extract_json(raw)
        lst = obj.get("limitations") if isinstance(obj, dict) else obj if isinstance(obj, list) else None
        for it in (lst or [])[:k]:
            if not isinstance(it, dict):
                continue
            claim = clean_unit(str(it.get("limitation") or it.get("claim") or ""))
            if len(claim.split()) < 4:
                continue
            ids = [int(x[1:]) - 1 for x in it.get("source_ids", []) or []
                   if isinstance(x, str) and re.fullmatch(r"C\d+", x) and 0 < int(x[1:]) <= len(led)]
            src = [led[i] for i in ids]
            best = max(src, key=_rank) if src else None
            final.append({"limitation": claim, "societal_impact": str(it.get("societal_impact") or ""),
                          "affected_population": str(it.get("affected_population") or ""),
                          "source_ids": [f"C{i + 1}" for i in ids], "agents": sorted({s["agent"] for s in src}),
                          "evidence_span": best["evidence_span"] if best else "",
                          "evidence_source": best["evidence_source"] if best else "",
                          "ph_tags": sorted({t for s in src for t in s.get("ph_tags", [])}
                                            | set(re.findall(r"PH-\d+", json.dumps(it)))),
                          "anchored": bool(best and best.get("anchored"))})
    leader_ok = bool(final)
    if not final:                                       # leader failed: top-k by rank
        for c in led[:k]:
            final.append({"limitation": c["claim"], "societal_impact": c["societal_impact"],
                          "affected_population": c["affected_population"], "agents": [c["agent"]],
                          "evidence_span": c["evidence_span"], "evidence_source": c["evidence_source"],
                          "ph_tags": c.get("ph_tags", []), "anchored": c.get("anchored")})
    meta = {"n": len(final), "n_emitted": n_emitted, "n_after_gate": 0 if gate_fallback else len(kept),
            "n_after_leader_dedup": len(led), "leader_ok": leader_ok, "gate_fallback": gate_fallback,
            "agent_errors": errors, "agent_parse_modes": modes, "mode": "parallel" if parallel else "sequential",
            "seconds": round(time.time() - t0, 1), "parse_mode": "agents"}
    trace = {"candidates": cands, "dropped_by_gate": [{"claim": d[0].get("claim"), "why": d[1]} for d in dropped]}
    return final, meta, json.dumps(trace, ensure_ascii=False, default=str)
