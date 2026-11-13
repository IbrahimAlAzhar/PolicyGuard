"""
Paper-level public-health context from the locally downloaded CDC / Census / CMS / HRSA data.

For each paper we infer (a) the target condition (-> CDC PLACES MeasureId) and
(b) the US states the paper mentions, and build a short context block. Every
number in the block carries a tag like [PH-3]; ph_provenance_json maps each tag
to the exact file (path, size, mtime, sha256), data release fields (Year /
DataSource when present), the filter that was applied and the row indices used,
so any critique that cites [PH-3] can be audited back to the source rows.

Everything is defensive: a missing file or an unknown schema drops that block
with a note instead of failing the run.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config

STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}
NON_US = ["China", "Chinese", "Japan", "Korea", "Taiwan", "India", "Iran", "Brazil", "Germany", "France",
          "Italy", "Spain", "United Kingdom", "UK Biobank", "England", "Canada", "Australia", "Netherlands",
          "Sweden", "Denmark", "Norway", "Finland", "Switzerland", "Turkey", "Israel", "Saudi", "Egypt",
          "Nigeria", "South Africa", "Mexico", "Singapore", "Thailand", "Pakistan", "Bangladesh", "Ethiopia"]
US_TERMS = re.compile(r"\b(United States|U\.S\.|USA|US adults|Medicare|Medicaid|NHANES|BRFSS|Veterans Affairs|VA health)\b")

# keyword -> CDC PLACES MeasureId (first match wins; order matters)
DISEASE_MAP = [
    ("diabet", "DIABETES"), ("glyc", "DIABETES"), ("coronary", "CHD"), ("myocardial", "CHD"),
    ("heart failure", "CHD"), ("heart disease", "CHD"), ("cardiovascular", "CHD"), ("copd", "COPD"),
    ("chronic obstructive", "COPD"), ("asthma", "CASTHMA"), ("stroke", "STROKE"), ("hypertens", "BPHIGH"),
    ("blood pressure", "BPHIGH"), ("kidney", "KIDNEY"), ("renal", "KIDNEY"), ("obes", "OBESITY"),
    ("body mass", "OBESITY"), ("cholesterol", "HIGHCHOL"), ("lipid", "HIGHCHOL"), ("depress", "DEPRESSION"),
    ("mental health", "MHLTH"), ("arthritis", "ARTHRITIS"), ("smok", "CSMOKING"), ("tobacco", "CSMOKING"),
    ("cancer", "CANCER"), ("tumor", "CANCER"), ("oncolog", "CANCER"), ("carcinoma", "CANCER"),
    ("dental", "DENTAL"), ("oral health", "DENTAL"), ("sleep", "SLEEP"), ("alcohol", "BINGE"),
    ("physical activity", "LPA"), ("sedentary", "LPA"),
]
ACCESS_MEASURE = "ACCESS2"   # PLACES: lack of health insurance among adults 18-64 (always added)


# ----------------------------------------------------------------------------- inference from text
def infer_disease_code(text: str) -> str | None:
    t = (text or "").lower()
    best, pos = None, 10 ** 9
    for kw, code in DISEASE_MAP:                  # earliest mention in title+abstract wins
        i = t.find(kw)
        if 0 <= i < pos:
            best, pos = code, i
    return best


def infer_states(text: str) -> list[str]:
    found = []
    for name in sorted(STATES, key=len, reverse=True):
        if re.search(rf"\b{name}\b", text or ""):
            if name == "Washington" and re.search(r"Washington (University|DC|D\.C\.)", text):
                continue
            if name == "Georgia" and not re.search(r"Georgia\W+(USA|US|United States|Atlanta)|Atlanta", text):
                continue
            found.append(name)
    # 'West Virginia' contains 'Virginia': drop the shorter one when only the longer matches
    if "West Virginia" in found and "Virginia" in found and not re.search(r"(?<!West )\bVirginia\b", text):
        found.remove("Virginia")
    return found[:4]


def infer_setting(text: str) -> dict:
    t = text or ""
    non_us = sorted({c for c in NON_US if re.search(rf"\b{re.escape(c)}\b", t)})
    return {"non_us_mentions": non_us[:5], "us_mentioned": bool(US_TERMS.search(t)),
            "rural_mentioned": bool(re.search(r"\brural\b", t, re.I))}


# ----------------------------------------------------------------------------- file helpers
def _fingerprint_cache_path():
    return os.path.join(config.CACHE_DIR, "file_fingerprints.json")


@lru_cache(maxsize=64)
def fingerprint(path: str) -> dict:
    """size / mtime / sha256 of a data file (sha cached on disk by size+mtime)."""
    if not path or not os.path.exists(path):
        return {"file": path, "exists": False}
    st = os.stat(path)
    key = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    cp = _fingerprint_cache_path()
    cache = {}
    try:
        cache = json.load(open(cp)) if os.path.exists(cp) else {}
    except Exception:
        cache = {}
    if key not in cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        cache[key] = h.hexdigest()
        try:
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            json.dump(cache, open(cp, "w"), indent=0)
        except Exception:
            pass
    return {"file": path, "exists": True, "size_bytes": st.st_size,
            "mtime": pd.Timestamp(st.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S"), "sha256": cache[key]}


@lru_cache(maxsize=32)
def _load(path: str):
    if not path or not os.path.exists(path):
        alt = os.path.splitext(path)[0] + ".csv" if path else None
        if alt and os.path.exists(alt):
            path = alt
        else:
            return None
    try:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path, low_memory=False, on_bad_lines="skip")
    except Exception as e:  # noqa: BLE001
        print(f"[publichealth] could not read {path}: {e}")
        return None


def _resolved(path: str) -> str:
    if path and not os.path.exists(path):
        alt = os.path.splitext(path)[0] + ".csv"
        if os.path.exists(alt):
            return alt
    return path


def _pick_csv(directory: str, keywords=()) -> str | None:
    if not directory or not os.path.isdir(directory):
        return None
    files = sorted(glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True))
    for kw in keywords:
        hit = [f for f in files if kw.lower() in os.path.basename(f).lower()]
        if hit:
            return hit[0]
    return files[0] if files else None


def _col(df, *cands, contains=None):
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    if contains:
        for c in df.columns:
            if all(x in c.lower() for x in contains):
                return c
    return None


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _release(df, cols=("Year", "DataSource", "data_source", "year", "release", "Release")):
    out = {}
    for c in cols:
        if c in df.columns:
            v = df[c].dropna().astype(str).unique().tolist()
            out[c] = v[:5]
    return out


def _state_mask(df, states):
    """Rows of df in any of the states (full name or 2-letter abbreviation)."""
    if not states:
        return None
    scol = _col(df, "StateAbbr", "State", "STATE", "state_abbr", "StateDesc", "ST_ABBR", "state")
    if scol is None:
        scol = _col(df, contains=("state",))
    if scol is None:
        return None
    vals = df[scol].astype(str).str.strip().str.lower()
    keys = {s.lower() for s in states} | {STATES[s].lower() for s in states}
    return vals.isin(keys)


# ----------------------------------------------------------------------------- context builder
class Block:
    def __init__(self, ctx):
        self.ctx = ctx

    def add(self, text, source, path, df=None, filt="", rows=None, statistic="", values=None):
        tag = f"PH-{len(self.ctx.prov) + 1}"
        self.ctx.lines.append(f"[{tag}] {text}")
        rec = {"tag": tag, "source": source, "text": text, **fingerprint(path), "filter": filt,
               "statistic": statistic}
        if df is not None:
            rec["release"] = _release(df)
        if rows is not None:
            rows = [int(r) for r in list(rows)[:25]]
            rec["row_indices"] = rows
        if values is not None:
            rec["row_values"] = values[:10]
        self.ctx.prov.append(rec)


class _Ctx:
    def __init__(self):
        self.lines, self.prov, self.notes = [], [], []


def _places(ctx, code, states, label):
    path = _resolved(config.PH_PATHS["cdc_places_county"])
    df = _load(path)
    if df is None or not code:
        return
    mcol, vcol = _col(df, "MeasureId", "measureid"), _col(df, "Data_Value", "data_value")
    if mcol is None or vcol is None:
        ctx.notes.append("CDC PLACES schema not recognised (MeasureId / Data_Value)")
        return
    tcol = _col(df, "Data_Value_Type", "DataValueTypeID")
    sub = df[df[mcol].astype(str).str.upper() == code]
    filt = f"{mcol}=='{code}'"
    if tcol is not None and len(sub):
        crude = sub[tcol].astype(str).str.contains("crude", case=False, na=False)
        if crude.any():
            sub, filt = sub[crude], filt + f" & {tcol} contains 'Crude'"
    if sub.empty:
        return
    v = _num(sub[vcol])
    ok = v.notna()
    sub, v = sub[ok], v[ok]
    meas_col = _col(df, "Measure", "measure")
    name = str(sub[meas_col].iloc[0]) if meas_col else code
    loc = _col(df, "LocationName", "CountyName")
    st = _col(df, "StateAbbr", "StateDesc")
    lo, md, hi = np.percentile(v, [10, 50, 90])
    yr = _release(sub).get(_col(df, "Year") or "Year", [])
    B = Block(ctx)
    B.add(f"CDC PLACES{(' ' + '/'.join(yr)) if yr else ''} - {name} ({label}): US county median {md:.1f}% "
          f"(10th-90th pct {lo:.1f}-{hi:.1f}%, {len(v):,} counties).", "cdc_places", path, sub, filt,
          sub.index[:25], f"percentiles over {len(v)} county rows")
    top = sub.loc[v.sort_values(ascending=False).index[:3]]
    vals = [{"county": str(r.get(loc, "")), "state": str(r.get(st, "")), "value": float(r[vcol])}
            for _, r in top.iterrows()]
    B.add("Highest-burden counties: " + "; ".join(f"{x['county']}, {x['state']} {x['value']:.1f}%" for x in vals)
          + ".", "cdc_places", path, sub, filt + " ; top-3 by value", top.index, "top-3 rows", vals)
    m = _state_mask(sub, states)
    if m is not None and m.any():
        ss = sub[m]
        sv = _num(ss[vcol])
        B.add(f"In {', '.join(states)}: county median {sv.median():.1f}% (range {sv.min():.1f}-{sv.max():.1f}%, "
              f"{len(ss)} counties).", "cdc_places", path, ss, filt + f" ; state in {states}", ss.index,
              "median/min/max of in-state county rows")


def _hospitals(ctx, code, states):
    path = _resolved(config.PH_PATHS["hospital_risk_matrix"])
    df = _load(path)
    if df is None:
        return False
    B = Block(ctx)
    m = _state_mask(df, states)
    scope = df[m] if m is not None and m.any() else df
    where = ", ".join(states) if scope is not df else "US"
    n = len(scope)
    parts = [f"{n:,} hospitals ({where})"]
    hp = _col(df, "hpsa_primary_care_flag", contains=("hpsa",))
    if hp is not None:
        share = _num(scope[hp]).fillna(0).astype(bool).mean() * 100
        parts.append(f"{share:.0f}% in primary-care HPSAs")
    ht = _col(df, "Hospital Type", "hospital_type")
    if ht is not None:
        cah = scope[ht].astype(str).str.contains("critical access", case=False, na=False).mean() * 100
        parts.append(f"{cah:.0f}% critical-access hospitals")
    B.add("CMS hospitals x HRSA: " + ", ".join(parts) + ".", "hospital_risk_matrix", path, scope,
          f"state in {states}" if scope is not df else "all rows", scope.index[:25], "share of rows")
    if code:
        col = f"{code.lower()}_prev"
        if col in df.columns:
            top = scope.assign(_v=_num(scope[col])).nlargest(3, "_v")
            name = _col(df, "Facility Name", "facility_name")
            vals = [{"hospital": str(r.get(name, "")), "state": str(r.get(_col(df, "State") or "State", "")),
                     "prev": float(r["_v"]), "hpsa": str(r.get(hp, "")) if hp else ""} for _, r in top.iterrows()]
            B.add("Hospitals in the highest local " + code + " prevalence areas: "
                  + "; ".join(f"{x['hospital']} ({x['state']}, {x['prev']:.1f}%)" for x in vals) + ".",
                  "hospital_risk_matrix", path, scope, f"top-3 by {col}", top.index, "top-3 rows", vals)
    return True


def _cms_raw(ctx, states):
    path = _pick_csv(config.PH_PATHS["cms_hospital_dir"], ("general", "hospital"))
    df = _load(path) if path else None
    if df is None:
        return
    ht = _col(df, "Hospital Type", "hospital_type")
    if ht is None:
        return
    m = _state_mask(df, states)
    scope = df[m] if m is not None and m.any() else df
    vc = scope[ht].astype(str).value_counts().head(3)
    Block(ctx).add(f"CMS Hospital General Information ({', '.join(states) if scope is not df else 'US'}): "
                   + "; ".join(f"{k}: {v:,}" for k, v in vc.items()) + ".", "cms", path, scope,
                   f"value_counts({ht})", scope.index[:25], "counts")


def _tracts(ctx, states):
    path = _resolved(config.PH_PATHS["tract_equity"])
    df = _load(path)
    if df is None:
        return
    svi = _col(df, "svi_overall", "RPL_THEMES")
    if svi is None:
        return
    m = _state_mask(df, states)
    scope = df[m] if m is not None and m.any() else df
    s = _num(scope[svi])
    hv = scope[s > 0.75]
    parts = [f"{len(hv):,} of {len(scope):,} tracts ({len(hv) / max(1, len(scope)) * 100:.0f}%) have SVI > 0.75"]
    for c, lab in (("pct_black", "Black"), ("pct_hispanic", "Hispanic")):
        if c in df.columns:
            a, b = _num(hv[c]).mean(), _num(scope[c]).mean()
            if pd.notna(a) and pd.notna(b):
                parts.append(f"mean % {lab} {a:.1f} in high-SVI tracts vs {b:.1f} overall")
    Block(ctx).add("CDC SVI x Census ACS (" + (", ".join(states) if scope is not df else "US") + "): "
                   + "; ".join(parts) + ".", "tract_equity", path, scope, f"{svi} > 0.75", hv.index[:25],
                   "counts / means over tract rows")


def _hrsa(ctx, states):
    path = _pick_csv(config.PH_PATHS["hrsa_dir"], ("primary", "hpsa", "shortage"))
    df = _load(path) if path else None
    if df is None:
        return
    tcol = (_col(df, "HPSA Discipline Class", "Designation Type", "HPSA Type", "hpsa_type")
            or _col(df, contains=("discipline",)) or _col(df, contains=("type",)))
    m = _state_mask(df, states)
    scope = df[m] if m is not None and m.any() else df
    stat = _col(df, "HPSA Status", "hpsa_status")
    if stat is not None:
        des = scope[stat].astype(str).str.contains("designated", case=False, na=False)
        if des.any():
            scope = scope[des]
    txt = f"HRSA shortage areas ({', '.join(states) if m is not None and m.any() else 'US'}): {len(scope):,} designations"
    if tcol is not None:
        vc = scope[tcol].astype(str).value_counts().head(3)
        txt += " (" + "; ".join(f"{k}: {v:,}" for k, v in vc.items()) + ")"
    Block(ctx).add(txt + ".", "hrsa", path, scope, f"state in {states}" if states else "all designated rows",
                   scope.index[:25], "row counts")


@lru_cache(maxsize=512)
def _build_cached(code, states: tuple):
    ctx = _Ctx()
    st = list(states)
    try:
        _places(ctx, code, st, "target condition")
    except Exception as e:  # noqa: BLE001
        ctx.notes.append(f"cdc_places failed: {e!r}")
    try:
        _places(ctx, ACCESS_MEASURE if code != ACCESS_MEASURE else None, st, "uninsured adults 18-64")
    except Exception as e:  # noqa: BLE001
        ctx.notes.append(f"cdc_places access failed: {e!r}")
    for fn in (lambda: _hospitals(ctx, code, st) or _cms_raw(ctx, st), lambda: _tracts(ctx, st),
               lambda: _hrsa(ctx, st)):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            ctx.notes.append(f"block failed: {e!r}")
    return list(ctx.lines), list(ctx.prov), list(ctx.notes)


def build_context(title: str, abstract: str, text: str, max_chars: int = 2600) -> dict:
    head = f"{title or ''}\n{abstract or ''}"
    code = infer_disease_code(head) or infer_disease_code((text or "")[:6000])
    states = infer_states(f"{head}\n{(text or '')[:20000]}")
    setting = infer_setting(f"{head}\n{(text or '')[:20000]}")
    lines, prov, notes = _build_cached(code, tuple(states))
    hdr = [f"Target condition (inferred): {code or 'none detected'}; US states mentioned: "
           f"{', '.join(states) if states else 'none'}."]
    if setting["non_us_mentions"] and not setting["us_mentioned"]:
        hdr.append("The study appears to be conducted outside the US (mentions: "
                   + ", ".join(setting["non_us_mentions"]) + "); the US figures below describe a potential US "
                   "deployment population.")
    body = "\n".join(hdr + lines)
    if len(body) > max_chars:
        body = body[:max_chars].rsplit("\n", 1)[0]
    return {"text": body if lines else "", "provenance": prov, "notes": notes, "disease_code": code,
            "states": states, "setting": setting}


def data_inventory() -> dict:
    """Which public-health files exist (printed at the start of a run)."""
    inv = {}
    for k, p in config.PH_PATHS.items():
        if k.endswith("_dir"):
            inv[k] = f"{len(glob.glob(os.path.join(p, '**', '*.csv'), recursive=True))} csv" if os.path.isdir(p) else "missing"
        else:
            rp = _resolved(p)
            inv[k] = "ok" if os.path.exists(rp) else "missing"
    return inv
