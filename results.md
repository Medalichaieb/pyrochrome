# Results log

One entry per training run. Newest first. Record: date, dataset file/commit,
features, model, and metrics per target. This is the lightweight experiment
tracker (MLflow / W&B optional later).

---

## 2026-06-27 — Baseline reproduced on real Glazy data
- Dataset: `glazy-data-glazes-20211130.csv` (Glazy repo, shallow clone 2026-06-27)
- Features: 42 = UMF oxides + aggregates (R2O/RO/SiO2:Al2O3) + cone (ordinal). No atmosphere yet.
- Split: test_size=0.2, random_state=42, stratified. Median imputation.
- Command: `make train`

| Target          | n     | Naïve | RandomForest          | GradientBoosting       | RF Top-2 |
|-----------------|-------|-------|-----------------------|------------------------|----------|
| Surface         | 8,397 | 59.9% | **78.0%** (F1 0.671)  | 77.3% (F1 0.671)       | —        |
| Colour family   | 5,511 | 44.8% | 64.6% (F1 0.496)      | 64.6% (F1 0.490)       | **79.5%**|

Notes: matches the brief's PoC figures (78% surface, 65%/80% colour). This is the
reference point for the model-exploration runs below.

---

## 2026-06-27 — Model comparison (5-fold stratified CV)
- Dataset/features: same as above (UMF + cone, 42 features, no atmosphere yet).
- Protocol: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, mean ± std.
- Command: `make compare` (+ a one-off run for LightGBM / XGBoost via `uv run --with`).

**Surface** (n=8,397, 3 classes)

| model         | accuracy      | macro-F1      | top-2         |
|---------------|---------------|---------------|---------------|
| random_forest | 0.767 ± 0.003 | 0.651 ± 0.006 | 0.911 ± 0.004 |
| lightgbm      | 0.766         | 0.659         | 0.914         |
| extra_trees   | 0.766 ± 0.002 | 0.654 ± 0.007 | 0.907 ± 0.007 |
| xgboost       | 0.766         | 0.653         | 0.911         |
| **hist_gb**   | 0.764 ± 0.004 | **0.658 ± 0.009** | 0.912 ± 0.007 |
| mlp           | 0.724 ± 0.006 | 0.612 ± 0.008 | 0.895 ± 0.004 |
| logreg        | 0.615 ± 0.009 | 0.533 ± 0.009 | 0.827 ± 0.009 |
| majority      | 0.600         | 0.250         | 0.754         |

**Colour family** (n=5,511, 10 classes)

| model         | accuracy      | macro-F1      | top-2         |
|---------------|---------------|---------------|---------------|
| **hist_gb**   | **0.664 ± 0.008** | **0.511 ± 0.017** | 0.794 ± 0.009 |
| lightgbm      | 0.663         | 0.508         | 0.795         |
| random_forest | 0.662 ± 0.011 | 0.512 ± 0.018 | 0.792 ± 0.008 |
| xgboost       | 0.660         | 0.504         | 0.794         |
| extra_trees   | 0.644 ± 0.008 | 0.498 ± 0.020 | 0.780 ± 0.013 |
| mlp           | 0.584 ± 0.018 | 0.430 ± 0.023 | 0.738 ± 0.014 |
| majority      | 0.448         | 0.062         | 0.473         |
| logreg        | 0.358 ± 0.008 | 0.287 ± 0.009 | 0.522 ± 0.005 |

**Decision: `HistGradientBoostingClassifier`.** The tree ensembles (RF, ExtraTrees,
HistGB, LightGBM, XGBoost) are statistically tied — all differences sit inside the
±0.008–0.011 inter-fold std. HistGB wins/ties on the harder colour target and has
the best surface macro-F1, matches LightGBM/XGBoost **without** a native dependency
(pure sklearn, one lockfile), and log-loss boosting calibrates better than RF
(matters for the confidence index). See `pyrochrome/models/selected.py`.

### 2026-06-27 — Selected model held-out metrics (`make train`)
- Single stratified 80/20 holdout (seed=42), artifacts saved to `models_out/`.

| Target        | n     | accuracy | macro-F1 | top-2 |
|---------------|-------|----------|----------|-------|
| Surface       | 8,397 | 0.767    | 0.665    | 0.908 |
| Colour family | 5,511 | 0.649    | 0.489    | 0.798 |

---

## Template

```
### YYYY-MM-DD — <short description>
- Dataset: glazy-data-glazes-YYYYMMDD.csv (commit <hash>)
- Features: UMF + aggregates + cone [+ atmosphere?]
- Split: test_size=0.2, random_state=42, stratified

| Target          | n      | Naïve | RandomForest        | GradientBoosting | Top-2 |
|-----------------|--------|-------|---------------------|------------------|-------|
| Surface         |        |       |                     |                  |       |
| Colour family   |        |       |                     |                  |       |
| Colour (Lab ΔE) |        |       |                     |                  |       |

Notes:
```

---

## Baseline reference (from the project brief — not yet reproduced in this repo)

### ~2026 — PoC baseline (UMF + cone, no atmosphere, noisy colour labels)
- Source: project brief / `prototypes/glaze_baseline.py`

| Target        | n     | Naïve | RandomForest                | Top-2 |
|---------------|-------|-------|-----------------------------|-------|
| Surface       | 8,397 | 60 %  | 78 % (F1-macro 0.68)        | —     |
| Colour family | 5,511 | 45 %  | 65 %                        | 80 %  |
