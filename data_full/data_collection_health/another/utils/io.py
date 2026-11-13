import json
import os
import re
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import requests
import yaml


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> int:
    ensure_parent(path)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def slugify(text: str, max_len: int = 120) -> str:
    text = text or "untitled"
    text = text.lower()
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-._")
    if len(text) > max_len:
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        text = text[: max_len - 9].rstrip("-") + "-" + digest
    return text or "record"


def get_session(user_agent: Optional[str] = None) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent or "health-peer-review-crawler/0.1",
        "Accept": "text/html,application/json,application/pdf,*/*;q=0.8",
    })
    return session


def safe_get(session: requests.Session, url: str, sleep: float = 1.0, timeout: int = 30, **kwargs) -> Optional[requests.Response]:
    try:
        time.sleep(sleep)
        r = session.get(url, timeout=timeout, allow_redirects=True, **kwargs)
        if r.status_code == 200:
            return r
        return None
    except requests.RequestException:
        return None


def download_file(session: requests.Session, url: str, out_path: str, sleep: float = 1.0) -> bool:
    ensure_parent(out_path)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True
    r = safe_get(session, url, sleep=sleep, timeout=60, stream=True)
    if not r:
        return False
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 64):
            if chunk:
                f.write(chunk)
    return os.path.getsize(out_path) > 0


def first_nonempty(*values):
    for v in values:
        if v:
            return v
    return None
