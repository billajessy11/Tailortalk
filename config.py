"""
TailorTalk — central configuration.
Change these values to tune model choice, paths, and fusion weights
without touching pipeline code.
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent

# ---- Data / index locations -------------------------------------------------
CSV_PATH = ROOT / "data" / "products.csv"
IMAGES_DIR = ROOT / "data" / "images"
MANIFEST_PATH = ROOT / "data" / "manifest.csv"

INDEX_DIR = ROOT / "indexes"
FAISS_INDEX_PATH = INDEX_DIR / "global.index"
STORE_PATH = INDEX_DIR / "store.npz"          # all sub-vectors, keyed by row id
METADATA_PATH = INDEX_DIR / "metadata.parquet"  # name/sku/price/urls, keyed by row id

# ---- Embedding model ---------------------------------------------------------
# ViT-B-32 (openai) is fast enough for CPU-only free hosting and gives solid
# global semantic embeddings. Swap to "ViT-L-14" + "openai" for higher quality
# if you have GPU / more build time budget (see README "Model choice").
CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL_NAME", "ViT-B-32")
CLIP_PRETRAINED = os.environ.get("CLIP_PRETRAINED", "openai")
EMBED_DIM = {"ViT-B-32": 512, "ViT-L-14": 768}.get(CLIP_MODEL_NAME, 512)

# Color histogram size: L*a*b bins per channel -> vector length = bins**3
COLOR_BINS = 6
COLOR_DIM = COLOR_BINS ** 3

# ---- Region-crop heuristic ----------------------------------------------------
# Catalogue saree shots are consistently full-length (worn or flat-laid), so we
# use fixed fractional crops instead of a segmentation model (kept out of scope
# for build-time/CPU budget — see README trade-offs).
TOP_FRACTION = 0.33      # pallu / shoulder drape / upper pattern detail
BOTTOM_FRACTION = 0.33   # hemline / border detail
CENTER_MARGIN = 0.20     # tighter central crop -> fabric field / weave texture

# ---- Fusion weights (must sum to 1.0; tune freely) ---------------------------
FUSION_WEIGHTS = {
    "global": 0.40,
    "top": 0.15,
    "bottom": 0.15,
    "center": 0.10,
    "color": 0.20,
}

# How many nearest neighbours to pull from FAISS on the global vector before
# re-ranking with the fused, region+colour-aware score. Dataset is small
# (~1k), so this pool is close to exhaustive -> re-ranking is effectively exact.
CANDIDATE_POOL = 200

# ---- LLM / agent ---------------------------------------------------------------
# Google Gemini's free tier requires no credit card / billing setup — good fit for
# a project that shouldn't cost anything to run. Get a free key at
# https://aistudio.google.com/app/apikey
GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_TOP_K = 5
MAX_TOP_K = 12
