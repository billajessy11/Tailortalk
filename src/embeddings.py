"""
Multi-signal embedding for saree images.

Whole-catalogue CLIP similarity alone tends to cluster on "it's a saree" —
the wrong granularity for this dataset. To capture the details that actually
differentiate two sarees (fabric/weave texture, border pattern, pallu work,
overall colourway) we compute FIVE vectors per image and fuse them at query
time (see search.py):

  global  - CLIP embedding of the full image        (overall look/style)
  top     - CLIP embedding of the top third          (pallu / shoulder drape)
  bottom  - CLIP embedding of the bottom third        (hemline / border)
  center  - CLIP embedding of a tight central crop    (fabric weave/texture,
                                                        margin-cropped to
                                                        reduce background)
  color   - Lab-space colour histogram, background-suppressed
                                                       (colourway match,
                                                        independent of CLIP's
                                                        texture/shape bias)

All CLIP vectors are L2-normalized so cosine similarity == dot product.
"""
import numpy as np
import cv2
from PIL import Image

import config

_model = None
_preprocess = None
_device = "cpu"


def _load_clip():
    global _model, _preprocess
    if _model is not None:
        return _model, _preprocess
    import torch
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED
    )
    model.eval()
    _model, _preprocess = model, preprocess
    return _model, _preprocess


def _clip_embed(pil_img: Image.Image) -> np.ndarray:
    import torch

    model, preprocess = _load_clip()
    with torch.no_grad():
        tensor = preprocess(pil_img).unsqueeze(0)
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).numpy().astype("float32")


def _crop_top(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.crop((0, 0, w, int(h * config.TOP_FRACTION)))


def _crop_bottom(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.crop((0, int(h * (1 - config.BOTTOM_FRACTION)), w, h))


def _crop_center(img: Image.Image) -> Image.Image:
    w, h = img.size
    mx, my = int(w * config.CENTER_MARGIN), int(h * config.CENTER_MARGIN)
    return img.crop((mx, my, w - mx, h - my))


def _color_histogram(img: Image.Image) -> np.ndarray:
    """
    Background-suppressed Lab histogram. Catalogue shots are typically on
    plain/near-white backdrops, so a raw full-image histogram would be
    dominated by background rather than garment colour. We mask out
    near-white / near-black pixels before histogramming.
    """
    arr = np.array(img.convert("RGB"))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    mask = (L > 12) & (L < 245)  # drop near-black / near-white (background)
    pixels = lab[mask]
    if pixels.shape[0] < 200:  # too little foreground detected -> fall back
        pixels = lab.reshape(-1, 3)

    bins = config.COLOR_BINS
    hist, _ = np.histogramdd(
        pixels.astype("float32"),
        bins=(bins, bins, bins),
        range=((0, 255), (0, 255), (0, 255)),
    )
    hist = hist.flatten().astype("float32")
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist = hist / norm
    return hist


def embed_all(pil_img: Image.Image) -> dict:
    """Return the full set of sub-vectors for one image."""
    img = pil_img.convert("RGB")
    return {
        "global": _clip_embed(img),
        "top": _clip_embed(_crop_top(img)),
        "bottom": _clip_embed(_crop_bottom(img)),
        "center": _clip_embed(_crop_center(img)),
        "color": _color_histogram(img),
    }
