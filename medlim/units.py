"""
Splitting text into limitation units and parsing LLM output.

* split_units(): author limitations (and any newline text) -> atomic units.
  Same rules as the archived eval_limitations.py: split on newline, strip bullets
  and '**', drop units with < min_words words, header lines and duplicates.
* parse_limitations(): LLM output -> list[dict]. Tries JSON first, then the
  'Limitation N: ...' structure, then plain bullet / numbered lines.
* format_limitations(): list -> 'Limitation 1: ...\nLimitation 2: ...'
"""
from __future__ import annotations

import ast
import json
import re

BULLET = re.compile(r"^\s*(?:[-*•▪●–]+|\(?\d{1,2}[.)]|\(?[a-hA-H][.)])\s*")
LIM_PREFIX = re.compile(r"^\s*(?:\*\*)?\s*limitation\s*#?\s*\d+\s*(?:\*\*)?\s*[:.)\-–]\s*(?:\*\*)?", re.I)
THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def _as_list(text: str):
    t = text.strip()
    if t.startswith("[") and t.endswith("]"):
        for loader in (json.loads, ast.literal_eval):
            try:
                v = loader(t)
                if isinstance(v, (list, tuple)):
                    return [str(x) for x in v]
            except Exception:
                pass
    return None


def is_empty(text) -> bool:
    return not isinstance(text, str) or not text.strip() or text.strip().lower() in ("nan", "none", "[]", "null")


def clean_unit(raw: str) -> str:
    s = LIM_PREFIX.sub("", raw.strip())
    s = BULLET.sub("", s)
    return s.replace("**", "").strip().strip('"').strip("'").strip()


def split_units(text, seps=("\n",), min_words: int = 3):
    """Returns (units, dropped)."""
    if is_empty(text):
        return [], []
    pieces = _as_list(text) or [text]
    pieces = [p.replace("\r\n", "\n").replace("\\n", "\n") for p in pieces]
    for sep in seps:
        pieces = [q for p in pieces for q in p.split(sep)]
    units, dropped, seen = [], [], set()
    for raw in pieces:
        raw = raw.strip()
        if not raw:
            continue
        s = clean_unit(raw)
        nw = len(s.split())
        if nw < min_words or (s.endswith(":") and nw <= 8):
            dropped.append(raw)
            continue
        k = re.sub(r"\s+", " ", s.lower())
        if k in seen:
            dropped.append(raw + "   (duplicate)")
            continue
        seen.add(k)
        units.append(s)
    return units, dropped


def author_units(text, min_words: int = 3) -> list[str]:
    """Author-mentioned limitations: one unit per '\n'-separated line."""
    return split_units(text, ("\n",), min_words)[0]


# ----------------------------------------------------------------------------- LLM output parsing
def strip_think(text: str) -> str:
    text = THINK.sub("", text or "")
    if "</think>" in text:                 # unterminated opening tag cut off by the template
        text = text.split("</think>", 1)[1]
    return text.strip()


def extract_json(text: str):
    """First JSON object/array in a string (handles ``` fences and prose around it)."""
    t = strip_think(text)
    t = re.sub(r"```(?:json|JSON)?", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    for o, c in (("{", "}"), ("[", "]")):
        s, e = t.find(o), t.rfind(c)
        if s != -1 and e > s:
            chunk = t[s:e + 1]
            for loader in (json.loads, ast.literal_eval):
                try:
                    return loader(chunk)
                except Exception:
                    continue
    return None


TEXT_KEYS = ("limitation", "claim", "text", "description", "issue", "canonical", "statement")


def _item_text(it) -> str:
    if isinstance(it, str):
        return it
    if isinstance(it, dict):
        for k in TEXT_KEYS:
            if isinstance(it.get(k), str) and it[k].strip():
                return it[k]
    return ""


def _json_items(obj):
    if isinstance(obj, dict):
        for k in ("limitations", "additional_limitations", "items", "results", "critiques"):
            if isinstance(obj.get(k), list):
                return obj[k]
        if any(isinstance(obj.get(k), str) for k in TEXT_KEYS):
            return [obj]
        return None
    if isinstance(obj, list):
        return obj
    return None


def parse_limitations(text: str, min_words: int = 3):
    """LLM output -> (items, mode). items: list[dict] each with at least 'limitation'.
    mode: 'json' | 'structured' | 'lines' | 'empty'."""
    if is_empty(text):
        return [], "empty"
    obj = extract_json(text)
    items = _json_items(obj) if obj is not None else None
    if items == [] and isinstance(obj, (dict, list)):
        return [], "json_empty"                 # a valid, deliberately empty answer
    if items:
        out, seen = [], set()
        for it in items:
            s = clean_unit(_item_text(it))
            if len(s.split()) < min_words or s.lower() in seen:
                continue
            seen.add(s.lower())
            d = dict(it) if isinstance(it, dict) else {}
            d["limitation"] = s
            out.append(d)
        if out:
            return out, "json"
    body = strip_think(text)
    # 'Limitation N: ...' blocks (may span several lines)
    blocks = re.split(r"(?im)^\s*(?:\*\*)?\s*limitation\s*#?\s*\d+\s*(?:\*\*)?\s*[:.)\-–]", body)
    if len(blocks) > 1:
        units = [clean_unit(" ".join(b.split())) for b in blocks[1:]]
        units = [u for u in units if len(u.split()) >= min_words]
        if units:
            return [{"limitation": u} for u in dict.fromkeys(units)], "structured"
    units, _ = split_units(body, ("\n",), min_words)
    units = [u for u in units if not re.match(r"(?i)^(here (are|is)|the following|sure|below)", u)]
    return [{"limitation": u} for u in units], ("lines" if units else "empty")


def format_limitations(items) -> str:
    texts = [_item_text(x) if not isinstance(x, str) else x for x in items]
    return "\n".join(f"Limitation {i + 1}: {t.strip()}" for i, t in enumerate(texts) if t and t.strip())


def parse_formatted(text) -> list[str]:
    """Inverse of format_limitations (also accepts any newline list)."""
    return split_units(text, ("\n",), 1)[0]
