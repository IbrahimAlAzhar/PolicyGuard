"""Logging tee, atomic CSV checkpoints, row selection and RAG passage loading."""
from __future__ import annotations

import json
import os
import sys
import threading
import time

import pandas as pd


class Tee:
    def __init__(self, stream, path):
        self.stream, self.f = stream, open(path, "a", buffering=1)
        self.lock = threading.Lock()

    def write(self, s):
        with self.lock:
            self.stream.write(s)
            self.f.write(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()

    def isatty(self):
        return False


def setup_logging(log_dir, name):
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Tee(sys.__stdout__, os.path.join(log_dir, f"{name}_output.txt"))
    sys.stderr = Tee(sys.__stderr__, os.path.join(log_dir, f"{name}_error.txt"))
    print("\n" + "=" * 100)
    log(f"{name} started: {' '.join(sys.argv)}")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def select_rows(df: pd.DataFrame, start_row: int, n_rows: int) -> list:
    end = None if n_rows is None or n_rows < 0 else start_row + n_rows
    return list(df.index[start_row:end])


def rag_passages(row, max_chars=2500, k=3) -> list[str]:
    v = row.get("rag_top3_json")
    if not isinstance(v, str) or not v.strip().startswith("["):
        return []
    try:
        lst = json.loads(v)
    except Exception:
        return []
    return [str(p)[:max_chars] for p in lst[:k] if isinstance(p, str) and p.strip()]


def ensure_cols(df, cols, dtype=object):
    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series([None] * len(df), index=df.index, dtype=dtype)
        elif df[c].dtype != object:
            df[c] = df[c].astype(object)


def done(v) -> bool:
    return isinstance(v, str) and v.strip() != "" and v.strip().lower() != "nan"
