## How it works

```
photo ──► SegFormer clothes parsing ──► one mask per piece (top, pants, shoes, bag…)
                                           │
                                           ▼
                       cut out onto white, square-padded (looks like a product shot)
                                           │
                                           ▼
             Marqo-FashionSigLIP image embedding  (+ one for the whole look)
                                           │
               ┌───────────────────────────┼─────────────────────────────┐
               ▼                           ▼                             ▼
     nearest products inside       "this, but in black leather":   complete the look: nearest
     the piece's own category      image + text scores, z-scored    items to the whole look in the
     (MMR to avoid duplicates)     and blended                     categories the outfit lacks
```

1. **Find the pieces.** A SegFormer-B2 model fine-tuned for clothes parsing labels every pixel (upper clothes, skirt, pants, dress, shoes, bag, hat, scarf, sunglasses, belt). Speckles are removed and each piece becomes a mask.
2. **Make it look like a catalog photo.** Each piece is cut out onto a white, square canvas with a feathered edge, because catalog products are shot on white. This closes much of the gap between a street photo and a product image.
3. **Embed.** [Marqo-FashionSigLIP](https://huggingface.co/Marqo/marqo-fashionSigLIP) maps images and text into one space where similar products sit close together.
4. **Search inside the right shelf.** A shoe is only compared with shoes. Search is exact cosine similarity over the whole catalog, then re-ranked with maximal marginal relevance so the results are not ten copies of the same item.
5. **Refine with words.** Image and text similarities live on different scales (≈0.7 vs ≈0.1), so each is z-scored over the candidates before blending with the weight you choose.
6. **Complete the look.** The whole outfit's embedding captures its palette, materials and formality. Within each category the outfit is missing (e.g. shoes, bag, a layer), Drape suggests the items closest to that style. This is zero-shot: no outfit-compatibility training yet.
7. **No person in the photo?** Drape treats the image as a single product and routes it to a category by comparing it with text prompts ("a pair of boots", "a handbag", …).

Everything runs on a free CPU: no GPU and no paid API at query time.
