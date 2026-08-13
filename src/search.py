"""
Search logic exposed to the agent as a single tool: `search_similar_sarees`.

Two-stage retrieval:
  1. FAISS exact search on the *global* CLIP vector -> a generous candidate
     pool (config.CANDIDATE_POOL). This is cheap and gets us "definitely a
     saree, roughly the right style" fast.
  2. Re-rank that pool with a weighted fusion of global + top-region +
     bottom-region + center-region CLIP similarity + colour-histogram
     similarity (config.FUSION_WEIGHTS). This is what pulls genuinely
     visually-close matches (same border work, same colourway, same weave)
     to the top instead of just "also a saree".

Public API:
    load_resources()              -> caches index/store/metadata in memory
    search(pil_image, top_k=5)    -> List[MatchResult]
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import faiss
from PIL import Image

import config
from src.embeddings import embed_all

_index = None
_store = None
_meta = None


@dataclass
class MatchResult:
    rank: int
    sku: str
    name: str
    score: float
    score_breakdown: dict
    image_url: str
    product_url: str
    retail_price: Optional[float] = None
    discounted_price: Optional[float] = None


def load_resources():
    global _index, _store, _meta
    if _index is not None:
        return
    if not config.FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No index found at {config.FAISS_INDEX_PATH}. "
            "Run `python -m src.download_images` then `python -m src.build_index` "
            "first (see README)."
        )
    _index = faiss.read_index(str(config.FAISS_INDEX_PATH))
    npz = np.load(config.STORE_PATH)
    _store = {
        "global": npz["global_"],
        "top": npz["top"],
        "bottom": npz["bottom"],
        "center": npz["center"],
        "color": npz["color"],
    }
    _meta = pd.read_parquet(config.METADATA_PATH)


def _cosine_rows(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine sim of a single query vector against every row of matrix.
    Vectors are already L2-normalized at embedding time, so this is a plain
    dot product; guarded for any zero-vectors (failed downloads)."""
    denom = (np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) or 1e-8))
    denom[denom == 0] = 1e-8
    return (matrix @ query) / denom


def search(
    pil_image: Image.Image,
    top_k: int = config.DEFAULT_TOP_K,
    weights: dict | None = None,
) -> list[MatchResult]:
    load_resources()
    weights = weights or config.FUSION_WEIGHTS
    top_k = max(1, min(top_k, config.MAX_TOP_K))

    q = embed_all(pil_image)

    # Stage 1: FAISS candidate pool on global vector
    qg = q["global"].reshape(1, -1).astype("float32")
    pool = min(config.CANDIDATE_POOL, _index.ntotal)
    _, ids = _index.search(qg, pool)
    ids = ids[0]
    ids = ids[ids >= 0]

    # Stage 2: fused re-rank over the candidate pool
    sims = {
        "global": _cosine_rows(q["global"], _store["global"][ids]),
        "top": _cosine_rows(q["top"], _store["top"][ids]),
        "bottom": _cosine_rows(q["bottom"], _store["bottom"][ids]),
        "center": _cosine_rows(q["center"], _store["center"][ids]),
        "color": _cosine_rows(q["color"], _store["color"][ids]),
    }
    fused = np.zeros(len(ids), dtype="float32")
    for key, w in weights.items():
        fused += w * sims[key]

    order = np.argsort(-fused)[:top_k]

    results = []
    for rank, idx_in_pool in enumerate(order, start=1):
        row_id = ids[idx_in_pool]
        row = _meta.iloc[row_id]
        results.append(MatchResult(
            rank=rank,
            sku=str(row["SKU"]),
            name=str(row["Name"]),
            score=float(fused[idx_in_pool]),
            score_breakdown={k: float(sims[k][idx_in_pool]) for k in sims},
            image_url=str(row["image_url"]),
            product_url=str(row["Website Link"]),
            retail_price=float(row["Retail Price"]) if pd.notna(row["Retail Price"]) else None,
            discounted_price=float(row["Discounted Price"]) if pd.notna(row["Discounted Price"]) else None,
        ))
    return results


def results_to_json(results: list[MatchResult]) -> list[dict]:
    return [r.__dict__ for r in results]
