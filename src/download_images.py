"""
Downloads every product image referenced in data/products.csv, converts it to
RGB JPEG (webp -> jpg, strips alpha), and writes data/manifest.csv mapping each
row to its local image path.

Resumable: re-running skips rows whose target file already exists.
Run this from a machine/CI runner with normal internet access:

    python -m src.download_images
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import io
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from tqdm import tqdm

import config

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
TIMEOUT = 15
MAX_RETRIES = 3


def download_one(row_id: int, url: str) -> tuple[int, str | None]:
    """Download+convert a single image. Returns (row_id, local_path or None on failure)."""
    target = config.IMAGES_DIR / f"{row_id:05d}.jpg"
    if target.exists():
        return row_id, str(target)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            target.parent.mkdir(parents=True, exist_ok=True)
            img.save(target, "JPEG", quality=92)
            return row_id, str(target)
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"[FAIL] row {row_id} ({url}): {e}")
                return row_id, None
            time.sleep(1.5 * attempt)
    return row_id, None


def main(max_workers: int = 8):
    import pandas as pd

    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.CSV_PATH)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(download_one, row.row_id, row.image_url): row.row_id
            for row in df.itertuples()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="downloading"):
            row_id, path = fut.result()
            results[row_id] = path

    df["local_path"] = df["row_id"].map(results)
    ok = df["local_path"].notna().sum()
    print(f"Downloaded/available: {ok}/{len(df)}")

    df_ok = df[df["local_path"].notna()].copy()
    df_ok.to_csv(config.MANIFEST_PATH, index=False)
    print(f"Manifest written to {config.MANIFEST_PATH}")


if __name__ == "__main__":
    main()
