# TailorTalk 🧵

A chat agent that finds visually similar sarees from a 1,074-item catalogue. Attach a photo
(upload or paste a link), and ask the agent to find matches — it decides when to search, calls
a vector-search tool behind the scenes, and returns ranked results with scores.

- **App URL:** _add after deploying — see "Deploy" below_
- **Code:** _this repository_

---

## What's in the box

| Layer | Choice | Why |
|---|---|---|
| Embeddings | OpenAI CLIP `ViT-B-32` via `open_clip` | Strong general visual-semantic embeddings; CPU-inference is fast enough for free-tier hosting. Swappable to `ViT-L-14` for more quality at the cost of build time (see below). |
| Vector store | **FAISS** (`IndexFlatIP`, exact search) | Dataset is ~1k images — exact search is cheap and removes ANN-approximation as a variable; still a real vector DB satisfying the required stack. |
| Agent / tool-calling | Google Gemini API (`google-genai`), native function calling | A single well-typed function (`search_similar_sarees`) the model calls when it detects similarity-search intent. Chosen specifically because Google AI Studio's free tier needs no billing/credit card, keeping this project runnable at zero cost — no LangChain/LlamaIndex dependency overhead either, but the pattern (tool schema, function-calling loop) is the same thing those frameworks wrap. |
| Frontend | Streamlit chat UI | Per spec. |

---

## Why raw CLIP similarity isn't enough here (and what I did about it)

Every image in this dataset is "a saree" — global CLIP embeddings cluster almost entirely on
that fact, and generic search returns loosely-similar results that share silhouette but not
colour, weave, or border work. To fix this, **each image is represented by five vectors, not
one**, fused at query time:

1. **`global`** — CLIP embedding of the whole image (overall style/silhouette)
2. **`top`** — CLIP embedding of the top third (pallu / shoulder-drape detail)
3. **`bottom`** — CLIP embedding of the bottom third (hemline / border detail)
4. **`center`** — CLIP embedding of a tightly-margined central crop (fabric weave/texture,
   with background cropped out)
5. **`color`** — a background-suppressed Lab-space colour histogram (colourway match,
   independent of CLIP's texture/shape bias — CLIP is known to under-weight exact colour)

**Search is two-stage:**
1. FAISS retrieves a broad candidate pool (top ~200) by `global` CLIP similarity — cheap,
   catches "definitely this general style."
2. That pool is **re-ranked** by a weighted fusion of all five signals
   (`config.FUSION_WEIGHTS`), which is what actually surfaces items that match on fine detail
   — same border pattern, same colour combination, similar fabric — rather than just "also a
   saree." Because the pool is close to exhaustive for a 1k-item catalogue, this re-rank is
   effectively an exact search over the fused score, not an approximation.

Each result comes back with a **score breakdown per signal**, not just one number — both so the
agent can describe *why* something matched, and so a reviewer can sanity-check the ranking.

### Trade-offs / assumptions made here
- **Region crops are heuristic (fixed fractions), not a segmentation model.** True garment
  segmentation (isolating the actual pallu/border pixels regardless of pose/drape) would be more
  precise, but adds a model + labeled data dependency out of scope for this build. Fixed
  top/bottom/center crops work reasonably well because catalogue photography here is fairly
  consistent (full-length, centered garment shots) — this is the main place a real garment
  segmentation model would improve results further.
- **`ViT-B-32` over `ViT-L-14`** by default, purely for CPU build/inference speed on free
  hosting. Set `CLIP_MODEL_NAME=ViT-L-14` as an env var and rebuild the index for a quality bump
  if you have more compute/time budget.
- **Fusion weights are hand-set, not learned** (no labeled "these two sarees are similar" pairs
  to train a re-ranker against). They're exposed in `config.FUSION_WEIGHTS` and easy to tune.

---

## Repo layout

```
config.py                  # all tunables: model choice, crop fractions, fusion weights
data/products.csv          # provided catalogue (name, sku, price, image_url, product link)
src/download_images.py     # CSV -> data/images/*.jpg + data/manifest.csv  (needs internet)
src/embeddings.py          # the 5-vector embedding pipeline described above
src/build_index.py         # manifest -> indexes/global.index, store.npz, metadata.parquet
src/search.py              # the actual search tool: FAISS candidates -> fused re-rank
src/agent.py                # Gemini tool-calling loop (system prompt + tool schema)
app.py                      # Streamlit chat frontend
.github/workflows/build_index.yml  # CI job that (re)builds the index on a runner with open internet
```

---

## Setup (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Download the catalogue images (needs normal internet access)
python -m src.download_images

# 2. Build the embeddings + FAISS index (one-time, ~10-20 min on CPU for ~1k images)
python -m src.build_index

# 3. Run the app
# Get a free API key (no billing required) at https://aistudio.google.com/app/apikey
export GOOGLE_API_KEY=AIza...
streamlit run app.py
```

If you don't want to run the index build locally, push the repo to GitHub and manually trigger
`.github/workflows/build_index.yml` (Actions tab → "Build search index" → Run workflow) — it
downloads the images and builds the index on GitHub's runner, then commits `indexes/` back to
the repo. This also re-runs automatically if `data/products.csv` changes.

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub, **with a pre-built `indexes/` folder committed** (either build it
   locally and commit it, or run the GitHub Action above first — building the index live at
   app-boot is intentionally avoided, it's slow and can hit platform health-check timeouts).
2. On [share.streamlit.io](https://share.streamlit.io), point at this repo, `app.py` as the
   entrypoint.
3. Add `GOOGLE_API_KEY` as a secret (Settings → Secrets) — get a free key (no billing
   required) at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey):
   ```toml
   GOOGLE_API_KEY = "AIza..."
   ```
4. Deploy. `packages.txt` handles the one system dependency (`libgl1`, for OpenCV).

(Hugging Face Spaces works the same way — Streamlit SDK, same secret, same pre-built `indexes/`.)

---

## Using it

- Paste an image URL or upload a photo in the sidebar.
- Chat normally — e.g. *"find me something similar to this"*, *"I like the border on this one,
  show close matches"*. The agent decides whether the request calls for a search; if you ask to
  search without an attached image, it'll ask you for one instead of guessing.
- Each result card shows the catalogue image, name, price, a link to the product page, overall
  score, and an expandable per-signal breakdown (global / top / bottom / center / colour).

## Known limitations
- Search quality is bounded by catalogue photography consistency — a query photo shot at a very
  different angle/lighting than the catalogue's studio shots will embed less reliably.
- No garment segmentation, so region crops occasionally include background/model skin rather
  than pure garment (mitigated but not eliminated by the colour-histogram background masking).
- Fusion weights are global, not per-query-adaptive (e.g. no automatic "the user seems to care
  about colour more than fabric" detection).
