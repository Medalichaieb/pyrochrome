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
| **Selected server model** | `HistGradientBoostingClassifier` | `pyrochrome.models.selected` → `pyrochrome.models.train` |
| Reference baseline | `RandomForestClassifier` + `HistGradientBoostingClassifier` vs naïve | `pyrochrome.models.baseline` |
| Model-selection harness | RF / ExtraTrees / HistGB / MLP / LogReg (CV) | `pyrochrome.models.compare` |
| In-browser predictor | compact MLP exported to JSON | `pyrochrome.models.export` → `web/` (TODO) |
| Nearest real recipes | k-NN in chemistry / colour space | TODO |

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

Selected model — `HistGradientBoostingClassifier` (held-out 20%, `make train`):

| Target | n | accuracy | macro-F1 | top-2 |
|---|---|---|---|---|
| Surface (Glossy/Matte/Satin) | 8,397 | **76.7 %** | 0.665 | 90.8 % |
| Colour family (10 classes) | 5,511 | 64.9 % | 0.489 | **79.8 %** |

Per-run, dated metrics live in [`results.md`](results.md) and detailed reports
(confusion matrices, feature importances, calibration, error analysis) in
[`reports/`](reports/). Regenerate with `make train && make eval`.

## Limitations & known biases

- **Atmosphere** not yet in the baseline → copper green↔red and other
  redox-sensitive colorants are undecidable (priority lever #1).
- **Colour labels are noisy** (photos under non-standardised lighting) →
  ceiling on colour accuracy (priority lever #2).
- **Kiln variance** between real kilns is a fundamental source of variance.
- Families with < 40 samples are dropped → no prediction for rare colours.

## How to reproduce

```bash
make setup && make data && make train && make eval
```

Seeds are fixed (`random_state=42`). Record the dataset file name/commit and the
resulting metrics in `results.md` for every run.
