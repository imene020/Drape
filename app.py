"""Drape — snap an outfit, shop every piece.  (Gradio app for Hugging Face Spaces)"""

from __future__ import annotations

import inspect
import logging
import random
import traceback
from pathlib import Path

import gradio as gr
import numpy as np

from drape import __version__
from drape.config import settings
from drape.index import CatalogIndex, Match
from drape.taxonomy import GROUP_LABELS, GROUPS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("drape")

ROOT = Path(__file__).parent
MAX_SLOTS = 6

# ---------------------------------------------------------------------------
# ZeroGPU: free Hugging Face accounts can only host Gradio apps on ZeroGPU
# hardware, which refuses to start unless at least one function is decorated
# with @spaces.GPU. Drape runs comfortably on CPU, so this stub is never called
# and visitors never spend GPU quota.
# ---------------------------------------------------------------------------
try:
    import spaces  # type: ignore

    @spaces.GPU
    def _zerogpu_placeholder():  # pragma: no cover
        return None

except ImportError:  # local runs / CI
    pass


# ---------------------------------------------------------------------------
# Startup: load the catalog and the models once.
# ---------------------------------------------------------------------------
ENGINE = None
INDEX: CatalogIndex | None = None
STARTUP_ERROR: str | None = None


def _load_engine():
    global ENGINE, INDEX, STARTUP_ERROR
    try:
        if settings.index_dir:
            log.info("Loading catalog index from %s", settings.index_dir)
            INDEX = CatalogIndex.load(settings.index_dir)
        else:
            log.info("Loading catalog index from the Hub: %s", settings.index_repo)
            INDEX = CatalogIndex.from_hub(settings.index_repo)
        log.info("Catalog: %d items %s", len(INDEX), INDEX.group_counts())

        from drape.models import OpenClipEmbedder, SegformerClothesSegmenter
        from drape.pipeline import ShopTheLook

        embedder = OpenClipEmbedder(settings.embed_model, device="cpu")
        if INDEX.manifest.get("embed_model") and INDEX.manifest["embed_model"] != embedder.name:
            raise RuntimeError(
                f"Index was built with {INDEX.manifest['embed_model']} but the app loads {embedder.name}. "
                "Rebuild the index or set DRAPE_EMBED_MODEL to match."
            )
        if embedder.dim != INDEX.dim:
            raise RuntimeError(f"Embedding size mismatch: model {embedder.dim} vs index {INDEX.dim}")
        segmenter = SegformerClothesSegmenter(settings.seg_model, device="cpu")
        ENGINE = ShopTheLook(embedder, segmenter, INDEX, max_side=settings.max_side)
        ENGINE._group_prototypes()  # warm up
        log.info("Drape ready.")
    except Exception as exc:  # keep the UI up and explain what is missing
        STARTUP_ERROR = f"{type(exc).__name__}: {exc}"
        log.error("Startup failed:\n%s", traceback.format_exc())


_load_engine()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _gallery(matches: list[Match], prefix: str = "") -> list[tuple]:
    return [(INDEX.thumbnail(m.row), f"{prefix}{m.caption()}") for m in matches]


def _require_engine():
    if ENGINE is None:
        raise gr.Error(
            "Drape is not ready: " + (STARTUP_ERROR or "still loading, try again in a minute.")
        )


def _empty_slots():
    out = []
    for _ in range(MAX_SLOTS):
        out += [gr.update(visible=False), gr.update(value=""), gr.update(value=[])]
    return out


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def on_analyze(image, k):
    _require_engine()
    if image is None:
        raise gr.Error("Upload a photo first, or press “Surprise me”.")
    try:
        result = ENGINE.analyze(image, k=int(k))
    except Exception as exc:
        log.error(traceback.format_exc())
        raise gr.Error(f"Could not analyse this photo: {exc}") from exc

    garments = result.garments[:MAX_SLOTS]
    if result.mode == "outfit":
        found = ", ".join(g.name.lower() for g in garments)
        status = f"**Found {len(garments)} piece{'s' if len(garments) != 1 else ''}:** {found}."
    else:
        status = (
            f"**No person detected**, so I treated the photo as a single product "
            f"(looks like **{garments[0].name.lower()}**)."
        )
    status += f"  <span class='drape-muted'>{result.timings['total_s']:.1f}s on CPU</span>"

    slots = []
    for i in range(MAX_SLOTS):
        if i < len(garments):
            g = garments[i]
            shelves = " / ".join(GROUP_LABELS[s].lower() for s in g.kind.search_groups)
            title = f"### {i + 1} · {g.name}\n<span class='drape-muted'>searched in {shelves}</span>"
            slots += [gr.update(visible=True), gr.update(value=title), gr.update(value=_gallery(g.matches))]
        else:
            slots += [gr.update(visible=False), gr.update(value=""), gr.update(value=[])]

    state = {
        "garments": [
            {"name": g.name, "groups": list(g.kind.search_groups), "emb": g.embedding.tolist()} for g in garments
        ],
        "look": result.look_embedding.tolist(),
        "present": result.present_groups,
    }
    choices = [f"{i + 1} · {g.name}" for i, g in enumerate(garments)]
    piece = gr.update(choices=choices, value=choices[0] if choices else None, interactive=bool(choices))
    looks = _complete(state, "")
    return [result.overlay, status, state, piece, gr.update(value=[]), looks, *slots]


def _complete(state, style_text):
    if not state:
        return gr.update(value=[])
    groups = ENGINE.complete_look(
        np.asarray(state["look"], dtype=np.float32), state["present"], k_per_group=4, style_text=style_text
    )
    items = []
    for group, matches in groups.items():
        items += _gallery(matches, prefix=f"{GROUP_LABELS[group]} · ")
    return gr.update(value=items)


def on_complete(state, style_text):
    _require_engine()
    if not state:
        raise gr.Error("Analyse a photo first.")
    return _complete(state, style_text)


def on_refine(state, piece, text, alpha, k):
    _require_engine()
    if not state or not piece:
        raise gr.Error("Analyse a photo first, then pick a piece to refine.")
    idx = int(piece.split("·")[0].strip()) - 1
    g = state["garments"][idx]
    matches = ENGINE.refine(
        np.asarray(g["emb"], dtype=np.float32), tuple(g["groups"]), text, alpha=float(alpha), k=int(k)
    )
    return gr.update(value=_gallery(matches))


def on_text_search(query, group_label, k):
    _require_engine()
    if not (query or "").strip():
        raise gr.Error("Describe what you are looking for, e.g. “red satin slip dress”.")
    group = None if group_label == "Everything" else _label_to_group[group_label]
    return gr.update(value=_gallery(ENGINE.text_search(query, k=int(k), group=group)))


def on_surprise():
    _require_engine()
    rows = INDEX.rows_for(("dress", "top", "outerwear", "bottom", "shoes", "bag"))
    return INDEX.thumbnail(int(random.choice(rows)))


_label_to_group = {GROUP_LABELS[g]: g for g in GROUPS}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_sm,
).set(
    button_primary_background_fill="#1c1917",
    button_primary_background_fill_hover="#44403c",
    button_primary_text_color="#fafaf9",
    button_primary_background_fill_dark="#fafaf9",
    button_primary_background_fill_hover_dark="#e7e5e4",
    button_primary_text_color_dark="#1c1917",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500&display=swap');
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
#drape-hero { padding: 28px 4px 8px; border-bottom: 1px solid var(--border-color-primary); margin-bottom: 8px; }
#drape-hero h1 { font-family: 'Cormorant Garamond', Georgia, serif; font-size: 3.4rem; font-weight: 600;
  letter-spacing: 0.02em; margin: 0; line-height: 1; }
#drape-hero h1 em { font-style: italic; font-weight: 500; color: #b45309; }
#drape-hero p { margin: 10px 0 0; font-size: 1.05rem; opacity: 0.8; }
.drape-muted { opacity: 0.6; font-size: 0.9em; }
.drape-section h3, .drape-section h2 { font-family: 'Cormorant Garamond', Georgia, serif; font-size: 1.6rem;
  font-weight: 600; margin: 4px 0; }
.drape-slot { border-top: 1px solid var(--border-color-primary); padding-top: 6px; }
footer { opacity: 0.6; }
"""


def _hero_md() -> str:
    if INDEX is not None:
        stats = f"{len(INDEX):,} products · Marqo-FashionSigLIP embeddings · SegFormer garment parsing · runs on CPU"
    else:
        stats = "Catalog not loaded yet"
    return (
        "<div id='drape-hero'><h1>Drape <em>shop the look</em></h1>"
        "<p>Upload any outfit photo. Drape finds each piece, matches it to real products, "
        "and suggests what completes the look.</p>"
        f"<p class='drape-muted'>{stats}</p></div>"
    )


def _how_it_works() -> str:
    md = (ROOT / "docs" / "how_it_works.md").read_text(encoding="utf-8")
    results = ROOT / "benchmark" / "results.md"
    if results.exists():
        md += "\n\n" + results.read_text(encoding="utf-8")
    else:
        md += "\n\n_Benchmark results will appear here once `scripts/benchmark.py` has been run._"
    return md


def build_ui() -> gr.Blocks:
    examples = sorted(p for p in (ROOT / "examples").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})

    with gr.Blocks(title="Drape · shop the look", **_blocks_style_kwargs()) as demo:
        gr.HTML(_hero_md())
        if STARTUP_ERROR:
            gr.Markdown(
                f"> **Setup needed.** The app started but could not load its catalog or models.\n>\n"
                f"> `{STARTUP_ERROR}`\n>\n> See *Deploy* in the README."
            )
        state = gr.State(None)

        with gr.Tab("Shop the look"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    image_in = gr.Image(type="pil", label="Your outfit photo", height=460, sources=["upload", "webcam", "clipboard"])
                    with gr.Row():
                        surprise_btn = gr.Button("Surprise me", variant="secondary")
                        analyze_btn = gr.Button("Find the pieces", variant="primary")
                    k_slider = gr.Slider(4, 16, value=settings.k, step=1, label="Matches per piece")
                    if examples:
                        gr.Examples(examples=[[str(p)] for p in examples], inputs=[image_in], label="Try a photo")
                with gr.Column(scale=6):
                    overlay_out = gr.Image(label="Detected pieces", height=460, interactive=False)
                    status_md = gr.Markdown("Upload a photo of a person wearing an outfit (full body works best).")

            slot_components = []
            with gr.Column(elem_classes="drape-section"):
                gr.Markdown("## Matching products")
                for _ in range(MAX_SLOTS):
                    with gr.Column(visible=False, elem_classes="drape-slot") as col:
                        title = gr.Markdown()
                        gal = gr.Gallery(columns=8, height=210, object_fit="contain", show_label=False, allow_preview=True)
                    slot_components += [col, title, gal]

            with gr.Row(equal_height=False):
                with gr.Column(elem_classes="drape-section"):
                    gr.Markdown("## Refine a piece\n<span class='drape-muted'>Same piece, with a twist: “in black leather”, “cropped”, “with gold buttons”.</span>")
                    with gr.Row():
                        piece_dd = gr.Dropdown(label="Piece", choices=[], interactive=False, scale=1)
                        refine_txt = gr.Textbox(label="Change it to…", placeholder="in emerald green", scale=2)
                    alpha_slider = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Look ↔ words (weight of your text)")
                    refine_btn = gr.Button("Refine", variant="primary")
                    refine_gal = gr.Gallery(columns=4, height=300, object_fit="contain", show_label=False)
                with gr.Column(elem_classes="drape-section"):
                    gr.Markdown("## Complete the look\n<span class='drape-muted'>Pieces the outfit is missing, chosen to match its colours and style.</span>")
                    with gr.Row():
                        style_txt = gr.Textbox(label="Optional style direction", placeholder="minimal, evening, streetwear…", scale=3)
                        complete_btn = gr.Button("Restyle", variant="secondary", scale=1)
                    complete_gal = gr.Gallery(columns=4, height=360, object_fit="contain", show_label=False)

        with gr.Tab("Search the catalog"):
            gr.Markdown("Describe a piece in your own words. Drape searches images, not keywords.")
            with gr.Row():
                q_txt = gr.Textbox(label="Looking for…", placeholder="oversized camel wool coat", scale=4)
                q_group = gr.Dropdown(["Everything", *[GROUP_LABELS[g] for g in GROUPS]], value="Everything", label="In", scale=1)
                q_k = gr.Slider(4, 24, value=12, step=1, label="Results", scale=1)
            q_btn = gr.Button("Search", variant="primary")
            q_gal = gr.Gallery(columns=6, height=520, object_fit="contain", show_label=False)
            gr.Examples(
                examples=[["red satin slip dress"], ["chunky white sneakers"], ["black leather crossbody bag"], ["linen wide-leg trousers"]],
                inputs=[q_txt],
            )

        with gr.Tab("How it works"):
            gr.Markdown(_how_it_works())

        gr.Markdown(
            f"<p class='drape-muted'>Drape v{__version__} · built by Imene Guizani · "
            "catalog images from the public Polyvore dataset, used for research demonstration only.</p>"
        )

        analyze_outputs = [overlay_out, status_md, state, piece_dd, refine_gal, complete_gal, *slot_components]
        analyze_btn.click(on_analyze, [image_in, k_slider], analyze_outputs)
        surprise_btn.click(on_surprise, None, image_in).then(on_analyze, [image_in, k_slider], analyze_outputs)
        refine_btn.click(on_refine, [state, piece_dd, refine_txt, alpha_slider, k_slider], refine_gal)
        refine_txt.submit(on_refine, [state, piece_dd, refine_txt, alpha_slider, k_slider], refine_gal)
        complete_btn.click(on_complete, [state, style_txt], complete_gal)
        style_txt.submit(on_complete, [state, style_txt], complete_gal)
        q_btn.click(on_text_search, [q_txt, q_group, q_k], q_gal)
        q_txt.submit(on_text_search, [q_txt, q_group, q_k], q_gal)
    return demo


# Gradio 6 moved theme/css from Blocks() to launch(); support both.
_LAUNCH_TAKES_STYLE = "css" in inspect.signature(gr.Blocks.launch).parameters


def _blocks_style_kwargs() -> dict:
    return {} if _LAUNCH_TAKES_STYLE else {"theme": THEME, "css": CSS}


def _launch_style_kwargs() -> dict:
    return {"theme": THEME, "css": CSS} if _LAUNCH_TAKES_STYLE else {}


demo = build_ui()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(ssr_mode=False, **_launch_style_kwargs())
