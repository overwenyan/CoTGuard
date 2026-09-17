"""Non-lexical read-outs for M11, with the exact M5 definitions (score_m5.py), cached by text hash."""

from __future__ import annotations

import hashlib
import os
import pathlib
import pickle
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = pathlib.Path(__file__).resolve().parent
_CACHE: dict = {}


def _h(t):
    return hashlib.md5(t.encode("utf-8", "ignore")).hexdigest()


def _cache_fp(kind):
    return pathlib.Path(os.environ.get("M11_CACHE", HERE / "data_m7")) / f"m11_cache_{kind}.pkl"


def load_cache(kind):
    fp = _cache_fp(kind)
    if kind not in _CACHE:
        _CACHE[kind] = pickle.loads(fp.read_bytes()) if fp.exists() else {}
    return _CACHE[kind]


def save_cache(kind):
    if kind in _CACHE:
        _cache_fp(kind).write_bytes(pickle.dumps(_CACHE[kind]))


def pos_tags(texts):
    sys.path.insert(0, str(HERE / ".pylib"))
    import nltk
    nltk.data.path.insert(0, str(HERE / ".pylib" / "nltk_data"))
    c = load_cache("pos")
    todo = [t for t in dict.fromkeys(texts) if _h(t) not in c]
    for t in todo:
        c[_h(t)] = " ".join(tag for _, tag in nltk.pos_tag(t.split()))
    return [c[_h(t)] for t in texts]


_ENC = None


def embed(texts):
    global _ENC
    c = load_cache("emb")
    todo = [t for t in dict.fromkeys(texts) if _h(t) not in c]
    if todo:
        if _ENC is None:
            from sentence_transformers import SentenceTransformer
            _ENC = SentenceTransformer("thenlper/gte-base", device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
        E = _ENC.encode(todo, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
        for t, e in zip(todo, E):
            c[_h(t)] = e.astype(np.float32)
    return np.stack([c[_h(t)] for t in texts])


class EmbVec:
    def fit_transform(self, X):
        return embed(X)

    def transform(self, X):
        return embed(X)


class PosVec:
    def __init__(self):
        self.v = TfidfVectorizer(ngram_range=(2, 4), sublinear_tf=True, min_df=2, max_features=50000,
                                 token_pattern=r"\S+", lowercase=False)

    def fit_transform(self, X):
        return self.v.fit_transform(pos_tags(X))

    def transform(self, X):
        return self.v.transform(pos_tags(X))


def make_vectorizer(kind):
    return {"emb": EmbVec, "pos": PosVec}[kind]()
