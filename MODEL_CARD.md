# MODEL CARD — Pyrochrome render predictor

## Intended use

Predict the post-firing render of a ceramic glaze (surface, transparency, colour
family / Lab colour) from its UMF chemistry, Orton cone and kiln atmosphere, to
**help ceramicists reduce test firings**. Decision-support, **not** a replacement
for firing tests. Predictions are always shown with a confidence index, a range,
and nearest real recipes.

**Out of scope**: food-safety / toxicity / durability judgements; commercial use
(data license is NonCommercial); guaranteeing an exact colour.

## Data

Glazy public dataset (~12,900 glazes), CC BY-NC-SA 4.0. See [DATA.md](DATA.md)
for provenance, cleaning, feature engineering and splits.

## Models

| Role | Model | Where |
|---|---|---|
| **Selected server model** (surface / colour family) | `HistGradientBoostingClassifier` | `pyrochrome.models.selected` → `pyrochrome.models.train` |
| **Colour (Lab regression)** | `ExtraTreesRegressor` (multi-output L*a*b*) | `pyrochrome.models.color_regression` |
| **Nearest real recipes** | k-NN over standardised chemistry features | `pyrochrome.models.neighbors` |
| Reference baseline | `RandomForestClassifier` + `HistGradientBoostingClassifier` vs naïve | `pyrochrome.models.baseline` |
| Model-selection harness | RF / ExtraTrees / HistGB / MLP / LogReg (CV) | `pyrochrome.models.compare` |
| In-browser predictor | compact MLP exported to JSON | `pyrochrome.models.export` → `web/` (TODO) |

Model docs:
[`HistGradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html),
[`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html),
[`MLPClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) /
[`MLPRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html).

### Selected model & rationale
**`HistGradientBoostingClassifier`** (params in `pyrochrome/models/selected.py`),
chosen from a 5-fold stratified CV comparison (full table in [`results.md`](results.md)):

- The tree ensembles (RandomForest, ExtraTrees, HistGB, LightGBM, XGBoost) are
  **statistically tied** — every difference is within the ±0.008–0.011 inter-fold std.
- HistGB **wins/ties on the harder colour target** (acc 0.664, top-2 0.794, macro-F1
  0.511) and has the **best surface macro-F1** (0.658).
- It matches LightGBM / XGBoost **without a heavy native dependency** → stays
  pure-sklearn and fully reproducible from a single `uv.lock`.
- **Log-loss gradient boosting calibrates better than RandomForest**, which matters
  because we surface a confidence index to users.
- No GPU; handles mixed scales and missing values; supports feature importances.

A **compact MLP** will additionally be exported to JSON for the in-browser predictor
(the `glaze_predictor.html` prototype proves that flow); it trades a few points of
accuracy for client-side, zero-backend inference.

## Results

Reference baseline (held-out 20%, `make baseline`):

| Target | n | Naïve | RandomForest | GradientBoosting | RF Top-2 |
|---|---|---|---|---|---|
| Surface (Glossy/Matte/Satin) | 8,397 | 60 % | 78.0 % (F1 0.671) | 77.3 % | — |
| Colour family (10 classes) | 5,511 | 45 % | 64.6 % | 64.6 % | 79.5 % |

Selected model — `HistGradientBoostingClassifier` + early stopping, **with the
atmosphere feature** (held-out 20%, `make train`):

| Target | n | accuracy | macro-F1 | top-2 |
|---|---|---|---|---|
| Surface (Glossy/Matte/Satin) | 8,397 | **76.9 %** | 0.650 | 91.3 % |
| Colour family (10 classes) | 5,511 | 65.7 % | 0.495 | **79.9 %** |

Per-run, dated metrics live in [`results.md`](results.md). The full evaluation
report — confusion matrices, per-class precision/recall/F1, **probability
calibration** (ECE ≤ 0.06, so the confidence % is trustworthy), permutation
feature importances and Lab diagnostics — is in
[`reports/REPORT.md`](reports/REPORT.md). Regenerate with `make train && make eval`.

### Lever #1 (atmosphere): measured, mostly a null result
Adding multi-hot atmosphere features (from the YAML dump, 80% coverage) gives a
**negligible aggregate gain** (colour acc +0.6%, top-2 +0.3%; surface ~0) —
contrary to the brief's expectation that atmosphere was the biggest lever. The
physical effect is nonetheless real: on copper recipes, reduction yields **Rouge
35%** (copper-red) while oxidation yields turquoise/green — exactly as expected.
The effect is just diluted because copper-with-known-atmosphere is ~13% of the
colour set. The features are **kept** (they help the redox-sensitive cases the
product cares about and don't hurt surface), but the binding ceiling is label
noise, so **lever #2 is re-prioritised**. Full analysis in [`results.md`](results.md).

### Lever #2 (colour labels): Lab regression + nearest real recipes
Colour is now also modelled as **CIELAB regression** (`ExtraTreesRegressor`),
reporting error as **ΔE**. Best CV ΔE ≈ **32.9** (mean), R² ≈ **0.40** — a 36%
cut vs the mean-prediction floor (ΔE 51.4), but a residual ΔE ≈ 33 that directly
quantifies the **photo-label-noise ceiling**. Because a single predicted colour
is therefore unreliable, a **k-NN nearest-real-recipe index** (over chemistry
features) surfaces real fired tiles (id, name, RGB, Lab) alongside any
prediction — the honest way to communicate the uncertainty. Both regenerate via
`make color` / `make neighbors`. Full tables in [`results.md`](results.md).

## Limitations & known biases

- **Colour labels are noisy** (photos under non-standardised lighting) → this is
  the binding ceiling on colour accuracy (**now the #1 lever**: clean labels /
  predict in Lab space).
- **Atmosphere tag is set-valued** — it lists the atmospheres a recipe is
  associated with, not the single one under which its recorded photo was fired,
  so it is only weakly informative per sample.
- **Kiln variance** between real kilns is a fundamental source of variance.
- Families with < 40 samples are dropped → no prediction for rare colours.

## How to reproduce

```bash
make setup && make data && make train && make eval
```

Seeds are fixed (`random_state=42`). Record the dataset file name/commit and the
resulting metrics in `results.md` for every run.
