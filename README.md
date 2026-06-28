# Pyrochrome

> **Predict the glaze before the kiln.**

Pyrochrome is an open-source tool that predicts the **post-firing render** of a
ceramic glaze — colour family, surface (glossy / matte / satin) and transparency
— from its **composition** (a recipe or its UMF oxide chemistry) and **firing
conditions** (Orton cone + atmosphere). Every prediction comes with a
**confidence index** and the **nearest real recipes** (with their photos), so a
ceramicist can reduce the number of test firings — without pretending to replace
them.

The problem is real: the render of a home-made glaze is notoriously "random"
depending on temperature and kiln atmosphere. Pyrochrome learns the
composition → render relationship from thousands of real recipes.

## Status

Proof of concept **validated**. From UMF chemistry + cone alone (no atmosphere,
noisy colour labels):

| Target | Naïve baseline | Model (RF / GradBoost) |
|---|---|---|
| Surface (Glossy/Matte/Satin), n=8,397 | 60 % | **78 %** (F1-macro 0.68) |
| Colour family (10 classes), n=5,511 | 45 % | **65 % top-1 / 80 % top-2** |

The link is learnable. A 5-fold CV comparison (RandomForest, ExtraTrees,
HistGradientBoosting, LightGBM, XGBoost, MLP, LogReg) selected
**`HistGradientBoostingClassifier`** as the server model — see
[MODEL_CARD.md](MODEL_CARD.md) and [results.md](results.md) for the full tables
and rationale.

**Lever #1 (atmosphere) — measured.** Multi-hot atmosphere features (from the
YAML dump) were added and evaluated. The aggregate gain is negligible (contrary
to the brief's expectation), though the effect is real where it matters (copper:
reduction → red, oxidation → green). The binding ceiling is **colour-label noise**.

**Lever #2 (colour) — done.** Colour is also modelled as **CIELAB regression**
(`make color`), reporting error as **ΔE** (best ΔE ≈ 33, R² ≈ 0.40 — a 36% cut
vs the floor, with the large residual quantifying the label-noise ceiling). A
**k-NN nearest-real-recipe index** (`make neighbors`) surfaces real fired tiles
alongside any prediction — the honest way to handle that noise. See
[results.md](results.md) for the full analysis.

## Architecture

```
[data pipeline]  download → parse YAML → join atmosphere → clean → features (UMF + cone + atm)
       │
[models]         surface / transparency / colour-family (classification) + colour Lab (regression)
                 RF / GradientBoosting baseline, compact MLP for the browser; k-NN for real examples
       │
[API]            FastAPI: POST recipe/chemistry → {surface, transparency, colour ± confidence, neighbours}
       │
[render]         procedural engine (Canvas/WebGL, client-side): attributes → stylised tile
       │
[frontend]       sober SPA (Vite + TypeScript): input → prediction + confidence + tile + real neighbours
```

## Quickstart

```bash
# 1. Install the pinned Python environment (needs uv + Python 3.12)
make setup

# 2. Download the Glazy dataset and build features
make data

# 3. Train and persist the selected model (HistGradientBoosting) per target
make train

# 4. Run the quality gates (lint + types + tests)
make check
```

Other useful targets: `make baseline` (reproduce the reference RF/GradBoost
baseline), `make compare` (cross-validate all candidate models), `make help`.

The web frontend lives in [`web/`](web/) (needs Node ≥ 20):

```bash
cd web && npm install && npm run dev
```

## Repository structure

```
pyrochrome/
├── src/pyrochrome/
│   ├── pipeline/      # download, cone mapping, atmosphere (YAML), feature engineering
│   ├── models/        # baseline training, evaluation, export
│   └── api/           # FastAPI prediction service
├── web/               # Vite + TypeScript SPA (sober, French UI)
├── prototypes/        # original reference artifacts (baseline + renderer + predictor demo)
├── tests/             # pytest
├── reports/           # versioned evaluation reports + figures
├── data/              # Glazy data (git-ignored; re-download with `make data`)
├── DATA.md            # data provenance, cleaning, feature engineering, splits, license
├── MODEL_CARD.md      # intended use, data, metrics, limitations, how to reproduce
└── results.md         # per-run metrics log
```

## Honesty notice

Pyrochrome **never** shows a single, falsely-precise colour. It always shows a
confidence index, a range, and real example tiles. Cone and atmosphere are
**required** inputs. See the "How it works" section of the site for the full
list of limitations (photos from non-standardised conditions, atmosphere, kiln
variance).

## Data & attribution

Trained on the [Glazy public dataset](https://github.com/derekphilipau/glazy-data)
(CC BY-NC-SA 4.0). Methodology reference:
[GlazyBench, Zhai et al., arXiv:2605.06641](https://arxiv.org/abs/2605.06641).
See [DATA.md](DATA.md) and [LICENSE](LICENSE).

## License

[CC BY-NC-SA 4.0](LICENSE) — attribution, non-commercial, share-alike.
