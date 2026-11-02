"""Build Drape's catalog index from a Hugging Face image dataset.

Run it where there is a GPU (a free Google Colab T4 takes ~10 minutes for
40k products), then push the result to a Hub dataset repo the Space loads:

    python scripts/build_index.py --push-repo <you>/drape-catalog

Output folder (default ``index_out/``):
    embeddings.npy    float16, L2-normalized, one row per product
    catalog.parquet   item_id, title, category, group, group_kw, group_zs, thumb (JPEG bytes)
    manifest.json     model, dataset, counts, build date
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drape.imaging import thumbnail_bytes  # noqa: E402
from drape.index import CatalogIndex  # noqa: E402
from drape.models import DEFAULT_EMBED_MODEL, OpenClipEmbedder, l2_normalize  # noqa: E402
from drape.taxonomy import GROUP_PROMPTS, GROUPS, OTHER, group_from_text  # noqa: E402

TEXT_COLS = ("text", "title", "name", "description", "productDisplayName")
CATEGORY_COLS = ("category", "category3", "category2", "sub-category", "subCategory", "category1", "articleType")


def pick_columns(features) -> tuple[str, str | None, list[str]]:
    from datasets import Image as HFImage

    image_col = next((c for c, f in features.items() if isinstance(f, HFImage)), None)
    if image_col is None:
        raise SystemExit(f"No image column found in {list(features)}")
    text_col = next((c for c in TEXT_COLS if c in features), None)
    cat_cols = [c for c in CATEGORY_COLS if c in features]
    return image_col, text_col, cat_cols


def zero_shot_groups(embedder, embs: np.ndarray) -> list[str]:
    names = [*GROUPS, OTHER]
    protos = np.stack([l2_normalize(embedder.embed_texts(GROUP_PROMPTS[g]).mean(axis=0)) for g in names])
    return [names[i] for i in np.argmax(embs @ protos.T, axis=1)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="Marqo/polyvore")
    ap.add_argument("--split", default=None, help="default: the dataset's first split")
    ap.add_argument("--max-items", type=int, default=40000)
    ap.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--thumb", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="index_out")
    ap.add_argument("--push-repo", default=None, help="Hub dataset repo to upload to, e.g. you/drape-catalog")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    from datasets import load_dataset

    t0 = time.time()
    dsd = load_dataset(args.dataset)
    split = args.split or next(iter(dsd.keys()))
    ds = dsd[split]
    image_col, text_col, cat_cols = pick_columns(ds.features)
    print(f"[build] {args.dataset}/{split}: {len(ds):,} rows | image={image_col} text={text_col} category={cat_cols}")

    ds = ds.shuffle(seed=args.seed)
    if args.max_items and len(ds) > args.max_items:
        ds = ds.select(range(args.max_items))

    embedder = OpenClipEmbedder(args.model, batch_size=args.batch_size)
    print(f"[build] model {embedder.name} on {embedder.device}, dim {embedder.dim}")

    rows, embs = [], []
    n = len(ds)
    for start in range(0, n, args.batch_size):
        batch = ds[start : start + args.batch_size]
        images, keep = [], []
        for j, img in enumerate(batch[image_col]):
            try:
                thumb = thumbnail_bytes(img, side=args.thumb)
            except Exception as exc:  # corrupt image: skip it
                print(f"[build] skip row {start + j}: {exc}")
                continue
            images.append(Image.open(io.BytesIO(thumb)))
            title = str(batch[text_col][j]) if text_col else ""
            cats = [str(batch[c][j]) for c in cat_cols if batch[c][j] is not None]
            rows.append(
                {
                    "item_id": f"{args.dataset.split('/')[-1]}-{start + j}",
                    "title": title,
                    "category": " / ".join(cats),
                    "group_kw": group_from_text(*cats, title),
                    "thumb": thumb,
                }
            )
            keep.append(j)
        if images:
            embs.append(embedder.embed_images(images))
        done = min(start + args.batch_size, n)
        if (start // args.batch_size) % 20 == 0 or done == n:
            print(f"[build] {done:,}/{n:,} embedded ({time.time() - t0:.0f}s)")

    emb = np.concatenate(embs).astype(np.float32)
    meta = pd.DataFrame(rows)
    meta["group_zs"] = zero_shot_groups(embedder, emb)
    meta["group"] = meta["group_kw"].fillna(meta["group_zs"])

    keep = (meta["group"] != OTHER).to_numpy()
    print(f"[build] dropping {int((~keep).sum()):,} non-fashion items (beauty, home, ...)")
    meta, emb = meta[keep].reset_index(drop=True), emb[keep]

    # Exact-duplicate removal (same product listed twice).
    rounded = np.round(emb, 3)
    _, first = np.unique(rounded, axis=0, return_index=True)
    first = np.sort(first)
    if len(first) < len(meta):
        print(f"[build] dropping {len(meta) - len(first):,} exact duplicates")
        meta, emb = meta.iloc[first].reset_index(drop=True), emb[first]

    labelled = meta["group_kw"].notna()
    agree = float((meta.loc[labelled, "group_kw"] == meta.loc[labelled, "group_zs"]).mean()) if labelled.any() else None
    manifest = {
        "drape_index_version": 1,
        "embed_model": embedder.name,
        "dim": int(emb.shape[1]),
        "dataset": args.dataset,
        "split": split,
        "seed": args.seed,
        "items": int(len(meta)),
        "groups": meta["group"].value_counts().to_dict(),
        "group_source": {"keyword": int(labelled.sum()), "zero_shot": int((~labelled).sum())},
        "zero_shot_agreement_with_keywords": agree,
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "note": "Product images come from a public research dataset; demo use only.",
    }
    index = CatalogIndex(emb, meta, manifest)
    index.save(args.out)
    print(f"[build] saved {len(index):,} items to {args.out}/ in {time.time() - t0:.0f}s")
    print(f"[build] groups: {index.group_counts()}")
    if agree is not None:
        print(f"[build] zero-shot group agrees with keyword label on {agree:.1%} of {int(labelled.sum()):,} items")

    if args.push_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_repo, repo_type="dataset", exist_ok=True, private=args.private)
        (Path(args.out) / "README.md").write_text(
            "---\nlicense: other\ntags: [fashion, retrieval, drape]\n---\n"
            f"# Drape catalog index\n\nEmbeddings ({embedder.name}) and thumbnails for {len(index):,} products "
            f"from `{args.dataset}`, built by `scripts/build_index.py`. "
            "Images belong to their original owners; research and demonstration use only.\n"
        )
        api.upload_folder(folder_path=args.out, repo_id=args.push_repo, repo_type="dataset",
                          commit_message=f"Index: {len(index):,} items, {embedder.name}")
        print(f"[build] pushed to https://huggingface.co/datasets/{args.push_repo}")


if __name__ == "__main__":
    main()
