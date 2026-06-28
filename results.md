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

## 2026-06-28 — Lever #1: atmosphere feature (multi-hot from YAML dump)
- Source: `glazy_20260531.yaml.gz` → `id` + multi-hot `atm_{oxidation,reduction,neutral,salt_soda,wood,raku,luster}` + `atm_known`, joined by `id` (cached in `data/processed/atmospheres.csv`).
- Coverage: 30,026 recipes carry ≥1 atmosphere; **80.3%** of cleaned recipes get a known atmosphere.
- Protocol: 3-fold stratified CV, selected model (HistGB + early stopping), identical for both arms. Features 42 → 50.

| Target  | metric | without atm | with atm | gain |
|---------|--------|-------------|----------|------|
| Surface | acc    | 0.758       | 0.758    | +0.001 |
| Surface | top-2  | 0.907       | 0.907    | −0.001 |
| Colour  | acc    | 0.647       | 0.653    | **+0.006** |
| Colour  | macro-F1 | 0.485     | 0.491    | +0.006 |
| Colour  | top-2  | 0.781       | 0.784    | +0.003 |

**Finding: the aggregate gain is negligible — this contradicts the brief's
hypothesis that atmosphere is the single biggest lever.** But the physical signal
is real and strong where it should be. On copper recipes (`CuO_umf > 0.01`, known
single atmosphere, n=981):

| Copper, oxidation-only (n=558) | Copper, reduction-only (n=158) |
|--------------------------------|--------------------------------|
| Turquoise 22%, Vert 19%, Bleu 9% (greens/blues; no red) | **Rouge 35%** (#1), Violet 14% (copper-red / sang-de-bœuf) |

Interpretation: atmosphere decides copper colour exactly as expected, but
copper-with-known-atmosphere is only ~13% of the colour set, so the aggregate
metric barely moves. Two compounding causes: (1) the **label-noise ceiling**
(photo RGB) is the binding constraint — lever #2; (2) the `Atmospheres` tag is a
*set* the recipe is associated with, not the single atmosphere under which the
recorded photo was fired, so it is only weakly informative per-sample.

**Decision:** keep the atmosphere features — small positive on colour, strong on
the redox-sensitive subset the product cares about, cheap, and they don't hurt
surface. But **re-prioritise lever #2 (clean colour labels / Lab space)** as the
real ceiling.

### 2026-06-28 — Selected model held-out (with atmosphere + early stopping, `make train`)

| Target        | n     | accuracy | macro-F1 | top-2 |
|---------------|-------|----------|----------|-------|
| Surface       | 8,397 | 0.769    | 0.650    | 0.913 |
| Colour family | 5,511 | 0.657    | 0.495    | 0.799 |

---

## 2026-06-28 — Lever #2: Lab colour regression + nearest real recipes
Move colour out of the noisy 10-family classification into **CIELAB regression**,
reporting error as **ΔE** (CIE76), and add a **k-NN nearest-real-recipe** index so
the product can show real fired tiles alongside any prediction.

**Lab regression** (5-fold CV, n=5,542 recipes with valid RGB, 50 features):

| model         | ΔE mean | ΔE median | MAE L | MAE a | MAE b | R²    |
|---------------|---------|-----------|-------|-------|-------|-------|
| **extra_trees** | **32.90** | 23.48   | 16.51 | 15.18 | 17.39 | **0.405** |
| random_forest | 34.09   | 25.05     | 16.91 | 15.81 | 18.27 | 0.395 |
| knn           | 40.70   | 32.48     | 20.74 | 18.56 | 21.29 | 0.201 |
| mean (floor)  | 51.43   | 32.38     | 29.46 | 21.02 | 23.44 | −0.003 |

ExtraTrees selected (`make color` → `models_out/colour_lab.joblib`). It cuts mean
ΔE by **36%** vs the mean-prediction floor (51.4 → 32.9) and explains ~40% of Lab
variance. **But the residual ΔE ≈ 33 is large** (ΔE > 5 is clearly visible to the
eye): this quantifies the colour-label-noise ceiling directly — the model is about
as good as the noisy photo labels allow. This is exactly why we surface real
neighbours rather than a single confident colour.

**Nearest real recipes** (`make neighbors` → `models_out/neighbors.joblib`):
k-NN over standardised chemistry features (UMF + cone + atmosphere) across 5,542
recipes with photos. A query returns the closest real, fireable recipes (id, name,
RGB, Lab, distance). Note: chemically-close recipes can have very different colours
(colorant/atmosphere/label noise), which is precisely the uncertainty the
neighbours communicate. Refinement idea: weight colorant oxides, or search in a
chemistry+predicted-colour space.

---

## 2026-06-28 — Evaluation report (`make eval`)
Full report + figures in [`reports/REPORT.md`](reports/REPORT.md) (confusion
matrices, per-class P/R/F1, calibration, permutation importances, Lab diagnostics).

| Target        | accuracy | macro-F1 | top-2 | ECE (top label) |
|---------------|----------|----------|-------|-----------------|
| Surface       | 0.774    | 0.660    | 0.919 | 0.044           |
| Transparency  | 0.658    | 0.610    | 0.851 | 0.047           |
| Colour family | 0.663    | 0.519    | 0.791 | 0.064           |
| Colour (Lab)  | ΔE 32.6 mean / 23 median | — | — | R² 0.42 |

Notes: **probabilities are well calibrated** (ECE ≤ 0.06 — the displayed confidence
% is trustworthy). Confusion matrices confirm the expected failure modes: a bias
toward "Blanc" on colour (photo-background label noise) and Satin↔Glossy/Matte
overlap on surface. Permutation importance: silica/alumina/flux balance drive
surface; colorant oxides + cone drive colour.

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
