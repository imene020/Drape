# Deploying Drape (free, ~1 hour)

Three free services: **GitHub** (code), **Google Colab** (one-time GPU job that builds the catalog), **Hugging Face** (the live app).

Replace `YOUR-GH` with your GitHub username. Your Hugging Face username is assumed to be `Imene-ee`.

---

## 0. One-time setup (5 min)

1. Install **Git** and **Git LFS** (Git for Windows already includes LFS). Then run once: `git lfs install`
2. Hugging Face → Settings → Access Tokens → **Create new token** → type **Write** → copy it somewhere safe.

## 1. Push to GitHub (10 min)

1. Go to https://github.com/new → name `drape` → **Public** → do **not** add a README, .gitignore or licence → Create.
2. Unzip `drape.zip`, open a terminal inside the `drape` folder:

```bash
git init
git add .
git commit -m "Drape v0.1: shop-the-look with fashion embeddings"
git branch -M main
git remote add origin https://github.com/YOUR-GH/drape.git
git push -u origin main
```

If Git asks for a password, GitHub wants a personal access token, not your password (GitHub → Settings → Developer settings → Personal access tokens), or install GitHub CLI and run `gh auth login` first.

3. Open the repo's **Actions** tab: the *tests* workflow should go green in ~2 minutes.

## 2. Build the catalog index on Colab (25 min, mostly waiting)

1. https://colab.research.google.com → File → Open notebook → **GitHub** tab → paste `https://github.com/YOUR-GH/drape` → open `notebooks/build_index_colab.ipynb`.
2. Runtime → Change runtime type → **T4 GPU**.
3. Click the **key icon** (Secrets) in the left sidebar → add `HF_TOKEN` = your write token → switch on **Notebook access**.
4. In cell 1, put your GitHub username in `GITHUB_REPO`.
5. Runtime → **Run all**.

Check the build cell's last lines: the item count per category, and "zero-shot group agrees with keyword label on X%". The index is now at `huggingface.co/datasets/Imene-ee/drape-catalog`.

6. The last cell downloads `benchmark_results.zip`. Unzip it into the repo so you have `benchmark/results.md`, then:

```bash
git add benchmark
git commit -m "Add benchmark results"
git push
```

Also paste the contents of `benchmark/results.md` into the **Benchmark** section of `README.md`.

## 3. Create the live Space (15 min)

1. https://huggingface.co/new-space → name `Drape` → SDK **Gradio** → hardware **ZeroGPU** (the only free option for Gradio on a free account; this is your 2nd of 2 slots, LabLens is the 1st) → **Public** → Create.
2. From the `drape` folder:

```bash
git remote add space https://huggingface.co/spaces/Imene-ee/Drape
git push --force space main
```

Username: `Imene-ee`. Password: your **write token**. (`--force` replaces the placeholder commit Hugging Face created.)

3. Watch the **Logs** tab: build (~5 min), then first start downloads the two models (~1–2 min). You want to see `Drape ready.`
4. Test: **Surprise me** → *Find the pieces*; upload a full-body outfit photo; the *Search the catalog* tab.

From now on, every update is: `git push` (GitHub) and `git push space main` (live app).

## 4. Polish (15 min)

- Add 2–4 outfit photos to `examples/` (your own, or Pexels/Unsplash), push to both remotes.
- Take a screenshot of a good result → `docs/screenshot.png` → uncomment the line in `README.md`.
- GitHub repo → ⚙ next to *About* → website = the Space URL; then pin the repo on your profile.

---

## Troubleshooting

| What you see | Fix |
|---|---|
| App shows **Setup needed** with `RepositoryNotFoundError` | The index repo name differs or is private. Make the dataset public, or Space → Settings → Variables → `DRAPE_INDEX_REPO` = `your-name/your-repo`, then Restart. |
| Space build fails on the Gradio version | In `README.md`, set `sdk_version` to the version Hugging Face pre-filled when it created the Space, push again. |
| `push rejected … binary files` | Run `git lfs install`, then `git add --renormalize .`, commit and push again. |
| Colab: `HF_TOKEN` not found | Secrets panel → turn on *Notebook access* for `HF_TOKEN`. |
| Colab: build says *No image column found* | Paste the printed column list to Claude; the builder needs one image column. |
| Space says *Sleeping* | Free Spaces sleep after ~48 h without visitors; the first visit wakes it in about a minute. |
