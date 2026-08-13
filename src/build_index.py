"""
Builds the search index from data/manifest.csv (produced by download_images.py):

  - indexes/global.index      FAISS IndexFlatIP over L2-normalized global CLIP
                               vectors (exact nearest-neighbour search — the
                               catalogue is small enough that approximate
                               indexing isn't needed for speed).
  - indexes/store.npz         top / bottom / center CLIP vectors + colour
                               histograms for every image, keyed by row id.
                               Used to compute the fused re-ranking score for
                               FAISS's candidate pool at query time.
  - indexes/metadata.parquet  name / sku / price / image_url / product_url
                               per row id, for rendering results.

Checkpoints every 50 images so an interrupted run can resume.
Run: python -m src.build_index
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import faiss

import config
from src.embeddings import embed_all

CHECKPOINT_EVERY = 50


def main():
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.MANIFEST_PATH)
    n = len(df)
    print(f"Building index for {n} images "
          f"(model={config.CLIP_MODEL_NAME}/{config.CLIP_PRETRAINED})")

    ckpt_path = config.INDEX_DIR / "_checkpoint.npz"
    start = 0
    global_vecs = np.zeros((n, config.EMBED_DIM), dtype="float32")
    top_vecs = np.zeros((n, config.EMBED_DIM), dtype="float32")
    bottom_vecs = np.zeros((n, config.EMBED_DIM), dtype="float32")
    center_vecs = np.zeros((n, config.EMBED_DIM), dtype="float32")
    color_vecs = np.zeros((n, config.COLOR_DIM), dtype="float32")

    if ckpt_path.exists():
        ckpt = np.load(ckpt_path)
        start = int(ckpt["done"])
        global_vecs[:start] = ckpt["global"][:start]
        top_vecs[:start] = ckpt["top"][:start]
        bottom_vecs[:start] = ckpt["bottom"][:start]
        center_vecs[:start] = ckpt["center"][:start]
        color_vecs[:start] = ckpt["color"][:start]
        print(f"Resuming from checkpoint at row {start}")

    for i in tqdm(range(start, n), desc="embedding"):
        path = df.iloc[i]["local_path"]
        try:
            img = Image.open(path)
            vecs = embed_all(img)
        except Exception as e:
            print(f"[WARN] row {i} ({path}) failed: {e} — using zero vector")
            vecs = {
                "global": np.zeros(config.EMBED_DIM, dtype="float32"),
                "top": np.zeros(config.EMBED_DIM, dtype="float32"),
                "bottom": np.zeros(config.EMBED_DIM, dtype="float32"),
                "center": np.zeros(config.EMBED_DIM, dtype="float32"),
                "color": np.zeros(config.COLOR_DIM, dtype="float32"),
            }
        global_vecs[i] = vecs["global"]
        top_vecs[i] = vecs["top"]
        bottom_vecs[i] = vecs["bottom"]
        center_vecs[i] = vecs["center"]
        color_vecs[i] = vecs["color"]

        if (i + 1) % CHECKPOINT_EVERY == 0 or i == n - 1:
            np.savez(
                ckpt_path, done=i + 1, global_=global_vecs, top=top_vecs,
                bottom=bottom_vecs, center=center_vecs, color=color_vecs,
                **{"global": global_vecs},
            )

    # --- FAISS index on the global vectors ---
    index = faiss.IndexFlatIP(config.EMBED_DIM)
    index.add(global_vecs)
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    print(f"FAISS index written: {config.FAISS_INDEX_PATH} ({index.ntotal} vectors)")

    # --- sub-vector store for fused re-ranking ---
    np.savez(
        config.STORE_PATH,
        global_=global_vecs, top=top_vecs, bottom=bottom_vecs,
        center=center_vecs, color=color_vecs,
    )
    print(f"Sub-vector store written: {config.STORE_PATH}")

    # --- metadata ---
    meta_cols = ["Name", "SKU", "Retail Price", "Discounted Price",
                 "image_url", "Website Link", "local_path"]
    df[meta_cols].to_parquet(config.METADATA_PATH, index=True)
    print(f"Metadata written: {config.METADATA_PATH}")

    if ckpt_path.exists():
        ckpt_path.unlink()
    print("Done.")


if __name__ == "__main__":
    main()
