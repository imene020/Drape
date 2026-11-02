"""Retrieval benchmark for Drape.

Known-item search under photo conditions: take catalog products, degrade them
the way a garment looks when cropped out of a real photo (see
``drape/augment.py``), and check whether the original product comes back.
Every model is evaluated on the same gallery and the same degraded queries.

    python scripts/benchmark.py --index-dir index_out

Writes ``benchmark/results.json`` and ``benchmark/results.md``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drape.augment import augment  # noqa: E402
from drape.index import CatalogIndex  # noqa: E402
from drape.models import DEFAULT_EMBED_MODEL, OpenClipEmbedder, l2_normalize  # noqa: E402
from drape.taxonomy import GROUP_PROMPTS, GROUPS, OTHER  # noqa: E402

DEFAULT_MODELS = [
    DEFAULT_EMBED_MODEL,  # fashion-tuned SigLIP (what the app ships)
    "ViT-B-16-SigLIP|webli",  # same architecture, generic web pre-training
    "ViT-B-32|laion2b_s34b_b79k",  # classic generic CLIP
]
NICE = {
    DEFAULT_EMBED_MODEL: "Marqo-FashionSigLIP",
    "ViT-B-16-SigLIP|webli": "SigLIP ViT-B/16 (generic)",
    "ViT-B-32|laion2b_s34b_b79k": "CLIP ViT-B/32 (generic)",
}


def ranks_of_targets(q: np.ndarray, g: np.ndarray, targets: np.ndarray, allowed: list[np.ndarray] | None) -> np.ndarray:
    """1-based rank of each query's target in the gallery."""
    sims = q @ g.T  # (Q, G)
    ranks = np.empty(len(q), dtype=np.int64)
    for i in range(len(q)):
        s = sims[i]
        if allowed is not None:
            s = np.where(allowed[i], s, -np.inf)
        ranks[i] = int((s > s[targets[i]]).sum()) + 1
    return ranks


def metrics(ranks: np.ndarray) -> dict:
    n = len(ranks)
    out = {f"recall@{k}": float((ranks <= k).mean()) for k in (1, 5, 10)}
    out["mrr"] = float((1.0 / ranks).mean())
    p = out["recall@1"]
    out["recall@1_ci95"] = float(1.96 * np.sqrt(p * (1 - p) / n))
    return out


def zero_shot_accuracy(embedder, embs: np.ndarray, labels: np.ndarray) -> float:
    names = [*GROUPS, OTHER]
    protos = np.stack([l2_normalize(embedder.embed_texts(GROUP_PROMPTS[g]).mean(axis=0)) for g in names])
    scores = embs @ protos.T
    scores[:, names.index(OTHER)] = -np.inf  # the catalog has no "other" items left
    pred = np.array(names)[np.argmax(scores, axis=1)]
    return float((pred == labels).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-dir", default="index_out")
    ap.add_argument("--gallery", type=int, default=5000)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--out-dir", default="benchmark")
    args = ap.parse_args()

    index = CatalogIndex.load(args.index_dir)
    rng = np.random.default_rng(args.seed)
    gallery_rows = np.sort(rng.choice(len(index), size=min(args.gallery, len(index)), replace=False))
    q_local = rng.choice(len(gallery_rows), size=min(args.queries, len(gallery_rows)), replace=False)
    groups = index.groups[gallery_rows]
    allowed = [groups == groups[t] for t in q_local]
    kw = index.meta["group_kw"].to_numpy()[gallery_rows] if "group_kw" in index.meta else np.array([None] * len(gallery_rows))
    labelled = np.array([v is not None and v == v for v in kw])  # not None / NaN

    print(f"[bench] gallery {len(gallery_rows):,} items, {len(q_local):,} queries, seed {args.seed}")
    gallery_imgs = [index.thumbnail(int(r)) for r in gallery_rows]
    queries = {}
    for setting in ("clean", "wild"):
        queries[setting] = [
            augment(gallery_imgs[t], np.random.default_rng([args.seed, i, setting == "wild"]), wild=(setting == "wild"))
            for i, t in enumerate(q_local)
        ]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for setting in ("clean", "wild"):  # save a few examples for the README
        strip = queries[setting][:6]
        from PIL import Image

        canvas = Image.new("RGB", (6 * 160, 2 * 160), (255, 255, 255))
        for j, t in enumerate(q_local[:6]):
            canvas.paste(gallery_imgs[t].resize((160, 160)), (j * 160, 0))
            q = strip[j].copy()
            q.thumbnail((160, 160))
            canvas.paste(q, (j * 160, 160))
        canvas.save(Path(args.out_dir) / f"queries_{setting}.jpg", quality=88)

    results = {"config": vars(args) | {"date": dt.date.today().isoformat()}, "models": {}}
    for spec in args.models:
        t0 = time.time()
        emb = OpenClipEmbedder(spec)
        g = emb.embed_images(gallery_imgs)
        res = {"device": emb.device}
        for setting, imgs in queries.items():
            q = emb.embed_images(imgs)
            res[setting] = {
                "all_catalog": metrics(ranks_of_targets(q, g, q_local, None)),
                "category_routed": metrics(ranks_of_targets(q, g, q_local, allowed)),
            }
        if labelled.any():
            res["zero_shot_group_accuracy"] = zero_shot_accuracy(emb, g[labelled], kw[labelled].astype(str))
            res["zero_shot_n"] = int(labelled.sum())
        res["seconds"] = round(time.time() - t0, 1)
        results["models"][spec] = res
        print(f"[bench] {NICE.get(spec, spec)}: {json.dumps(res, indent=None)[:400]}")
        del emb

    (Path(args.out_dir) / "results.json").write_text(json.dumps(results, indent=2))
    (Path(args.out_dir) / "results.md").write_text(render_markdown(results, index))
    print((Path(args.out_dir) / "results.md").read_text())


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def render_markdown(results: dict, index: CatalogIndex) -> str:
    cfg = results["config"]
    lines = [
        "## Benchmark",
        "",
        f"Known-item retrieval: {cfg['queries']:,} catalog products, degraded to look like garments cropped from "
        f"real photos, searched against a {cfg['gallery']:,}-product gallery (seed {cfg['seed']}, {cfg['date']}). "
        "*Clean*: off-centre crop, tilt, lighting/colour cast, blur, JPEG. *Wild*: also a non-white background "
        "and a 6–20% occlusion. *Routed*: search restricted to the garment's category, as the app does.",
        "",
        "| Model | Setting | R@1 (±95% CI) | R@5 | R@10 | MRR | R@1 routed | R@10 routed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for spec, res in results["models"].items():
        for setting in ("clean", "wild"):
            a, r = res[setting]["all_catalog"], res[setting]["category_routed"]
            lines.append(
                f"| {NICE.get(spec, spec)} | {setting} | {_pct(a['recall@1'])} ± {_pct(a['recall@1_ci95'])} | "
                f"{_pct(a['recall@5'])} | {_pct(a['recall@10'])} | {a['mrr']:.3f} | "
                f"{_pct(r['recall@1'])} | {_pct(r['recall@10'])} |"
            )
    zs = [(spec, res) for spec, res in results["models"].items() if "zero_shot_group_accuracy" in res]
    if zs:
        lines += [
            "",
            "**Zero-shot category routing** (used when a photo has no person in it): accuracy against "
            f"keyword-derived labels on {zs[0][1]['zero_shot_n']:,} gallery items, {len(GROUPS)} categories.",
            "",
            "| Model | Accuracy |",
            "|---|---|",
        ]
        lines += [f"| {NICE.get(s, s)} | {_pct(r['zero_shot_group_accuracy'])}% |" for s, r in zs]
    lines += [
        "",
        "**Caveats.** Marqo-FashionSigLIP was fine-tuned on data that includes Polyvore product images, so its "
        "absolute scores on this catalog may be optimistic; the generic models were not. Synthetic degradations "
        "approximate, but are not, real street photos — a consumer-to-shop evaluation is on the roadmap.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
