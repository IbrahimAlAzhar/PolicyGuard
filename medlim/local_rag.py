"""
Local hybrid retrieval for the experiment (no internet): BM25 + dense (local all-MiniLM-L6-v2 through
transformers, mean pooling), fused with reciprocal-rank fusion. Corpus = the OTHER papers of the 10k regex
dataset (the sampled papers are excluded, so a paper never retrieves itself). A passage = title + abstract +
the author-stated limitations of the similar paper (as in v1).
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

TOK = re.compile(r"[a-z0-9]{2,}")
STOP = set("the of and in to a for with on by is are was were from at as be that this an or we our study patients "
           "results methods using used between among than which these those not no".split())


def toks(s):
    return [w for w in TOK.findall(str(s).lower()) if w not in STOP]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [Counter(toks(d)) for d in docs]
        self.len = np.array([sum(c.values()) for c in self.docs], float)
        self.avg = self.len.mean() if len(self.len) else 1.0
        df = Counter(w for c in self.docs for w in c)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.k1, self.b = k1, b
        self.inv = {}
        for i, c in enumerate(self.docs):
            for w, f in c.items():
                self.inv.setdefault(w, []).append((i, f))

    def scores(self, q):
        s = np.zeros(len(self.docs))
        for w in set(toks(q)):
            if w not in self.inv:
                continue
            idf = self.idf[w]
            for i, f in self.inv[w]:
                s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


class Dense:
    def __init__(self, model_path, device=None, batch=64):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.m = AutoModel.from_pretrained(model_path).to(self.dev).eval()
        self.batch = batch

    def encode(self, texts):
        out = []
        with self.torch.no_grad():
            for s in range(0, len(texts), self.batch):
                b = self.tok([str(t)[:2000] for t in texts[s:s + self.batch]], padding=True, truncation=True,
                             max_length=256, return_tensors="pt").to(self.dev)
                h = self.m(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                out.append(self.torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 384))


def hybrid_topk(queries, corpus_docs, k=3, dense_model=None, rrf_k=60, log=print):
    bm = BM25(corpus_docs)
    dq = dc = None
    if dense_model:
        try:
            d = Dense(dense_model)
            log(f"[rag] dense encoder {dense_model} on {d.dev}: encoding {len(corpus_docs):,} corpus docs")
            dc, dq = d.encode(corpus_docs), d.encode(queries)
        except Exception as e:  # noqa: BLE001
            log(f"[rag] dense encoder unavailable ({e!r}) -> BM25 only")
    res = []
    for qi, q in enumerate(queries):
        s_b = bm.scores(q)
        rank = {}
        for r, i in enumerate(np.argsort(-s_b)[:100]):
            rank[i] = rank.get(i, 0) + 1 / (rrf_k + r + 1)
        if dc is not None:
            s_d = dc @ dq[qi]
            for r, i in enumerate(np.argsort(-s_d)[:100]):
                rank[i] = rank.get(i, 0) + 1 / (rrf_k + r + 1)
        res.append([int(i) for i, _ in sorted(rank.items(), key=lambda x: -x[1])[:k]])
    return res, ("bm25+dense" if dc is not None else "bm25")
