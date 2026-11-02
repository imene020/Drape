---
title: Drape
emoji: 👗
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: mit
short_description: Snap an outfit, shop every piece with fashion AI
models:
  - Marqo/marqo-fashionSigLIP
  - mattmdjaga/segformer_b2_clothes
datasets:
  - Marqo/polyvore
tags:
  - fashion
  - visual-search
  - retrieval
  - multimodal
---

# Drape — shop the look

**Snap an outfit, shop every piece.** Drape finds each garment in a photo, matches it to real products in a catalog, lets you tweak a piece in plain words ("this jacket, but in black leather"), and suggests what would complete the look.

**Live demo:** https://huggingface.co/spaces/Imene-ee/Drape  
<!-- After deploying, add a screenshot: docs/screenshot.png -->

| | |
|---|---|
| **Find the pieces** | Pixel-level garment parsing (top, dress, skirt, pants, shoes, bag, hat, scarf…) |
| **Match to products** | Fashion-tuned image embeddings, searched only within the right category |
| **Refine with words** | Blend the photo with a text instruction; you choose the balance |
| **Complete the look** | Style-coherent suggestions for the categories the outfit is missing |
| **Search by description** | "chunky white sneakers" searches product images, not keywords |
| **Free to run** | No GPU and no paid API at query time: a few seconds per photo on CPU |

## How it works

```
photo ─► SegFormer clothes parsing ─► one mask per piece
                                         │
                                         ▼
                    cut out onto white, square-padded (looks like a product shot)
                                         │
                                         ▼
          Marqo-FashionSigLIP embedding (each piece + the whole look)
                                         │
          ┌──────────────────────────────┼───────────────────────────────┐
          ▼                              ▼                               ▼
  nearest products in the        image + text scores, z-scored      nearest items to the whole look
  piece's category (MMR          and blended ("…but in black")      in the categories it lacks
  removes near-duplicates)                                          (shoes, bag, a layer…)
```

Design choices worth knowing:

- **Cut-out onto white.** Catalog products are shot on white; street photos are not. Masking each garment onto a white square narrows that domain gap before embedding.
- **Category routing.** Each detected garment is searched only within its own shelves (a top within tops and outerwear), so a shoe never returns a handbag.
- **Score fusion that respects scale.** Image–image cosines sit around 0.7 and image–text cosines around 0.1. Adding them raw lets the image side drown out the text, so each is z-scored over the candidates before blending.
- **Diversity.** Results are re-ranked with maximal marginal relevance and exact near-duplicates are skipped, so you see ten different products rather than one product ten times.
- **Graceful fallbacks.** No person in the photo? Drape treats it as one product and routes it to a category with zero-shot text prompts. Catalog or model missing at startup? The app still loads and says what is wrong.

## Benchmark

<!-- Replace this section with the contents of benchmark/results.md after running scripts/benchmark.py -->

Known-item retrieval under photo conditions: catalog products are degraded the way a garment looks when cropped out of a real photo (off-centre crop, tilt, colour cast, blur, JPEG, and in the *wild* setting a non-white background plus a partial occlusion), then searched against a 5,000-product gallery. Same gallery and same queries for every model: the fashion-tuned model Drape ships, the same architecture with generic pre-training, and a classic CLIP.

Run it yourself: `python scripts/benchmark.py --index-dir index_out` (about 10 minutes on a free Colab T4).

## Project layout

```
app.py                  Gradio app (Hugging Face Space entry point)
drape/
  taxonomy.py           garment kinds, catalog groups, keyword + zero-shot routing
  models.py             open_clip embedder, SegFormer segmenter (torch imported lazily)
  garments.py           masks -> clean crops, overlay rendering
  index.py              catalog index: exact cosine search, filters, fusion, MMR
  pipeline.py           analyze / refine / complete the look / text search
  augment.py            photo-condition degradations for the benchmark
scripts/
  build_index.py        embed a Hugging Face image dataset into a catalog index
  benchmark.py          retrieval benchmark vs. generic baselines
notebooks/
  build_index_colab.ipynb   one-click index build + benchmark on a free GPU
tests/                  pytest suite with fake models (no GPU, no downloads)
```

## Run locally

```bash
git clone https://github.com/<your-username>/drape && cd drape
pip install -r requirements.txt
python app.py            # downloads the models and the catalog index on first run
pytest -q                # tests use fake models, no downloads needed
```

## Rebuild the catalog index

The Space loads a prebuilt index (embeddings + thumbnails) from the Hub dataset `Imene-ee/drape-catalog`. To rebuild it, or to index a different catalog, open `notebooks/build_index_colab.ipynb` in Google Colab with a GPU runtime and run all cells, or:

```bash
pip install -r requirements-build.txt
huggingface-cli login
python scripts/build_index.py --dataset Marqo/polyvore --max-items 40000 --push-repo <you>/drape-catalog
```

Any Hugging Face dataset with an image column works; the builder picks up title and category columns when they exist and falls back to zero-shot categorisation when they do not.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `DRAPE_INDEX_REPO` | `Imene-ee/drape-catalog` | Hub dataset holding the index |
| `DRAPE_INDEX_DIR` | — | Local index folder (overrides the Hub) |
| `DRAPE_EMBED_MODEL` | `hf-hub:Marqo/marqo-fashionSigLIP` | Must match the model the index was built with |
| `DRAPE_SEG_MODEL` | `mattmdjaga/segformer_b2_clothes` | Garment parser |
| `DRAPE_MAX_SIDE` | `1024` | Photos are downscaled to this before parsing |

## Limitations

- "Complete the look" is zero-shot style similarity, not a trained compatibility model.
- The benchmark uses synthetic photo degradations of catalog images, not real street-to-shop pairs, and the fashion model saw Polyvore images during its own training (see caveats in the results).
- The garment parser does not separate a jacket from the top under it; both are searched together.
- Catalog images come from the public Polyvore research dataset and are used for demonstration only.

## Roadmap

- Outfit-compatibility model trained on Polyvore Outfits, evaluated on the standard fill-in-the-blank task.
- Consumer-to-shop evaluation on real photos (DeepFashion2).
- Bring-your-own-catalog: an API that indexes a retailer's product feed.
- Multilingual search (French, Arabic) for North African e-commerce.

## Credits

Marqo-FashionSigLIP (Marqo, Apache-2.0) · SegFormer clothes parser by mattmdjaga (fine-tuned on ATR) · Polyvore product images via `Marqo/polyvore` · built with open_clip, Transformers and Gradio.

Built by **Imene Guizani** · [LinkedIn](https://linkedin.com/in/imene-guizani)
