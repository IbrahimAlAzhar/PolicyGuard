"""
rag_hybrid.py (copied from v1 gpt+rag/rag.py; used only with --live_rag) — Hybrid FAISS + BM25 retrieval over the RAG corpus.

Pipeline:
  1. Read rag_corpus.csv.
  2. Build RAG_text column with the format:
        Title: {title}
        Abstract: {abstract}
        Other Sections: {first 500 words of full_text}
        Limitations: {limitations_section}
  3. Encode each RAG_text with BAAI/bge-large-en-v1.5 (top-MTEB dense model).
  4. Build a FAISS IndexFlatIP over normalized embeddings (cosine).
  5. Build a BM25Okapi index over whitespace/alpha-num tokens.
  6. At query time: hybrid score = 0.7 * FAISS_norm + 0.3 * BM25_norm.
     Returns the top-k RAG_text strings.

All artifacts (formatted corpus, embeddings, FAISS index, BM25) are cached
to disk so subsequent runs skip the expensive encode step.

Requires:
    pip install sentence-transformers faiss-cpu rank_bm25 tqdm pyarrow

Usage:
    from rag import HybridRAG
    rag = HybridRAG()                       # builds or loads cache
    hits = rag.retrieve(query_text, k=3)    # list[str] of top RAG_text
"""

from __future__ import annotations
import re
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
from . import config as _cfg

RAG_CORPUS_CSV = Path(_cfg.RAG_CORPUS_CSV)
RAG_CACHE_DIR = Path(_cfg.RAG_CACHE_DIR)

# Strong general-purpose retrieval model (MTEB top-tier, 1024-dim).
EMBED_MODEL_NAME = _cfg.RAG_EMBED_MODEL
FAISS_WEIGHT = 0.7
BM25_WEIGHT = 0.3
DEFAULT_TOP_K = 3

# Truncate query to keep encoder happy and retrieval fast.
QUERY_MAX_CHARS = 3000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _first_n_words(text: str, n: int = 500) -> str:
    if not isinstance(text, str):
        return ""
    words = text.split()
    return " ".join(words[:n])


def build_rag_text(row) -> str:
    """Format one row of the RAG corpus into the canonical RAG_text string."""
    title = str(row.get("title", "") or "").strip()
    abstract = str(row.get("abstract", "") or "").strip()
    other = _first_n_words(str(row.get("full_text", "") or ""), n=500)
    limitations = str(row.get("limitations_section", "") or "").strip()
    return (
        f"Title: {title}\n"
        f"Abstract: {abstract}\n"
        f"Other Sections: {other}\n"
        f"Limitations: {limitations}"
    )


def _tokenize_for_bm25(text: str) -> list[str]:
    """Lowercase alpha-num tokens for BM25."""
    if not isinstance(text, str):
        return []
    return re.findall(r"\w+", text.lower())


def _min_max(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


# ---------------------------------------------------------------------------
# HybridRAG
# ---------------------------------------------------------------------------
class HybridRAG:
    """FAISS (dense, 70%) + BM25 (sparse, 30%) hybrid retriever."""

    def __init__(
        self,
        corpus_csv: Path = RAG_CORPUS_CSV,
        cache_dir: Path = RAG_CACHE_DIR,
        embed_model_name: str = EMBED_MODEL_NAME,
        faiss_weight: float = FAISS_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
        device: Optional[str] = None,
    ):
        self.corpus_csv = Path(corpus_csv)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embed_model_name = embed_model_name
        self.faiss_weight = faiss_weight
        self.bm25_weight = bm25_weight
        self.device = device

        self.df: Optional[pd.DataFrame] = None
        self.rag_texts: list[str] = []
        self.embed_model = None
        self.faiss_index = None
        self.bm25 = None

        self._build_or_load()

    # ----- cache paths -----
    @property
    def _texts_path(self) -> Path:
        return self.cache_dir / "rag_corpus_formatted.parquet"

    @property
    def _emb_path(self) -> Path:
        return self.cache_dir / "rag_embeddings.npy"

    @property
    def _faiss_path(self) -> Path:
        return self.cache_dir / "rag_faiss.index"

    @property
    def _bm25_path(self) -> Path:
        return self.cache_dir / "rag_bm25.pkl"

    # ----- lazy model load -----
    def _load_model(self):
        if self.embed_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[rag] loading embedding model: {self.embed_model_name}")
            self.embed_model = SentenceTransformer(
                self.embed_model_name, device=self.device
            )

    # ----- one-time build / load -----
    def _build_or_load(self):
        import faiss

        # 1. Corpus + RAG_text
        if self._texts_path.exists():
            print(f"[rag] loading cached formatted corpus: {self._texts_path}")
            self.df = pd.read_parquet(self._texts_path)
        else:
            print(f"[rag] reading raw corpus: {self.corpus_csv}")
            self.df = pd.read_csv(self.corpus_csv, low_memory=False)
            print(f"[rag] formatting RAG_text for {len(self.df):,} papers")
            self.df["RAG_text"] = self.df.apply(build_rag_text, axis=1)
            # keep only what we need downstream
            keep_cols = [c for c in ("title", "pmid", "pmc", "RAG_text") if c in self.df.columns]
            self.df = self.df[keep_cols].reset_index(drop=True)
            self.df.to_parquet(self._texts_path, index=False)
        self.rag_texts = self.df["RAG_text"].astype(str).tolist()

        # 2. FAISS (dense)
        if self._faiss_path.exists():
            print(f"[rag] loading cached FAISS index: {self._faiss_path}")
            self.faiss_index = faiss.read_index(str(self._faiss_path))
        else:
            self._load_model()
            print(f"[rag] encoding {len(self.rag_texts):,} corpus passages")
            embeddings = self.embed_model.encode(
                self.rag_texts,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")
            np.save(self._emb_path, embeddings)
            self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
            self.faiss_index.add(embeddings)
            faiss.write_index(self.faiss_index, str(self._faiss_path))
            print(f"[rag] FAISS index built: {self.faiss_index.ntotal} vectors")

        # 3. BM25 (sparse)
        if self._bm25_path.exists():
            print(f"[rag] loading cached BM25 index: {self._bm25_path}")
            with open(self._bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
        else:
            from rank_bm25 import BM25Okapi
            print(f"[rag] tokenizing {len(self.rag_texts):,} passages for BM25")
            tokenized = [_tokenize_for_bm25(t) for t in tqdm(self.rag_texts)]
            self.bm25 = BM25Okapi(tokenized)
            with open(self._bm25_path, "wb") as f:
                pickle.dump(self.bm25, f)
            print(f"[rag] BM25 index built")

    # ----- query API -----
    def _encode_query(self, query: str) -> np.ndarray:
        self._load_model()
        v = self.embed_model.encode(
            [query[:QUERY_MAX_CHARS]],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        return v

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> list[str]:
        """
        Hybrid retrieval over the corpus.

        score = 0.7 * minmax(FAISS_cosine) + 0.3 * minmax(BM25)

        Returns the RAG_text strings of the top-k papers.
        """
        if not query or not isinstance(query, str):
            return []
        n = len(self.rag_texts)
        if n == 0:
            return []

        # Pull a broader pool from each index so the re-rank step has something
        # meaningful to combine.
        top_pool = min(max(k * 30, 100), n)

        # ---- FAISS ----
        qv = self._encode_query(query)
        f_scores, f_ids = self.faiss_index.search(qv, top_pool)
        f_scores = f_scores[0]
        f_ids = f_ids[0]

        # ---- BM25 ----
        tokens = _tokenize_for_bm25(query)
        bm25_all = (
            np.asarray(self.bm25.get_scores(tokens), dtype=np.float32)
            if tokens else np.zeros(n, dtype=np.float32)
        )

        # Union of candidate pool
        candidates: set[int] = {int(i) for i in f_ids if i >= 0}
        if tokens:
            bm25_top_idx = np.argpartition(-bm25_all, min(top_pool, n - 1))[:top_pool]
            candidates.update(int(i) for i in bm25_top_idx)
        cand = sorted(candidates)
        cand_arr = np.array(cand, dtype=np.int64)

        # Map FAISS hits back to a full-length score vector
        f_full = np.full(n, f_scores.min() if f_scores.size else 0.0, dtype=np.float32)
        for sid, sc in zip(f_ids, f_scores):
            if sid >= 0:
                f_full[sid] = sc

        f_norm = _min_max(f_full[cand_arr])
        b_norm = _min_max(bm25_all[cand_arr])
        combined = self.faiss_weight * f_norm + self.bm25_weight * b_norm

        order = np.argsort(-combined)[:k]
        top_ids = cand_arr[order].tolist()
        return [self.rag_texts[i] for i in top_ids]

    # ----- public encode helper (for offline pre-computation) -----
    def encode_text(self, text: str) -> np.ndarray:
        return self._encode_query(text)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rag = HybridRAG()
    sample_q = (
        "Title: Federated learning for diabetic retinopathy screening\n"
        "Abstract: We evaluated a federated model trained on EHR data from "
        "three academic medical centers; external validation was limited "
        "to one rural site."
    )
    hits = rag.retrieve(sample_q, k=3)
    for i, h in enumerate(hits):
        print(f"\n=== HIT {i+1} ===\n{h[:500]}...\n")