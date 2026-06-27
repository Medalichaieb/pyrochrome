# DATA — datasheet

Provenance, cleaning, feature engineering, splits and license of the data used
to train Pyrochrome.

## Source

- **Dataset**: Glazy public data — <https://github.com/derekphilipau/glazy-data>
- **Version**: pinned by clone date / commit (record the exact file name, e.g.
  `glazy-data-glazes-20211130.csv`, in `results.md` for every training run).
- **Volume**: ~12,900 glazes in the flat CSV.

### Files

| File | Content | Use |
|---|---|---|
| `glazy-data-glazes-YYYYMMDD.csv` | ~12,900 glazes, named columns, composition precomputed as `*_percent`, `*_umf`, `*_mol`, aggregates (`R2O_umf`, `RO_umf`, `SiO2_Al2O3_ratio_umf`), `from/to_orton_cone`, `surface_type`, `transparency_type`, `material_type`, `rgb_r/g/b`. | Primary feature + label source. |
| `glazy_YYYYMMDD.yaml.gz` | Full dump including the **`Atmospheres`** field (Oxidation / Reduction) — **absent from the flat CSV**. | Parsed and joined by `id` to add the atmosphere feature (priority lever #1). |
| `machine-learning/` | Maintainer's examples (cone regression). | Reference only. |

## Cleaning

Rows kept must satisfy (from the baseline): `is_analysis == 0`,
`is_primitive == 0`, `is_theoretical == 0` — i.e. real, non-theoretical recipes.

- **Cone**: Orton cones are **not linear** → mapped to an ordinal scale
  (see `pyrochrome.pipeline.cones`). The `&#189;` HTML entity is normalised to `.5`.
- **Surface label**: collapsed to `Glossy` / `Matte` / `Satin` by prefix match.
- **Colour label**: bucketed from `rgb_r/g/b` into perceptual families via HSV
  (see `pyrochrome.pipeline` / baseline `color_family`). **Known to be noisy**
  (photos taken under non-standardised conditions). Families with < 40 samples
  are dropped. Lever #2: clean these (GlazyBench approach) and move to Lab-space
  regression + k-NN nearest real tiles.
- **Atmosphere** (lever #1): parsed from the YAML, joined by `id`, encoded as a
  categorical feature.

## Feature engineering

Validated representation:

```
UMF oxides (precomputed by Glazy)
  + aggregates R2O_umf, RO_umf, SiO2_Al2O3_ratio_umf
  + cone (ordinal)
  + atmosphere (categorical)   ← to be added
```

All-zero UMF columns are dropped. Missing values imputed with the column median.

## Splits

- `train_test_split`, `test_size=0.2`, `random_state=42`, **stratified** on the
  target. A held-out test set is used for all reported metrics.
- TODO: add a validation split / cross-validation for model selection and a
  hand-annotated colour test set (GlazyBench-style) for honest colour metrics.

## License

Glazy data is licensed **CC BY-NC-SA 4.0**:

- **Attribution** — credit Glazy (Derek Au) and link the dataset.
- **NonCommercial** — no commercial use.
- **ShareAlike** — derivatives under the same license.

Raw recipes are not copyrightable, but descriptions, photos and metadata are.
The whole Pyrochrome project is released under CC BY-NC-SA 4.0 to comply
(see [LICENSE](LICENSE)). Methodology reference:
[GlazyBench](https://arxiv.org/abs/2605.06641).
