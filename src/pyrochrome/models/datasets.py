"""Prepare model-ready (X, y) datasets per target from the Glazy CSV.

Centralises loading + cleaning + feature building + label derivation so that the
baseline, the model-comparison harness and the final trainer all share *exactly*
the same preprocessing (no silent drift between experiments).

Inputs : path to the Glazy glazes CSV (default: latest under ``data/raw/``).
Outputs: a :class:`TargetData` per target, each holding the feature matrix
         ``X`` (median-imputed), the label vector ``y`` and the feature names.

Targets:
    - ``surface`` : Glossy / Matte / Satin (3 classes)
    - ``colour``  : colour family (10 classes, rare families dropped)

The colour labels are noisy (photo-derived RGB); this is the PoC label scheme
that priority lever #2 will replace with cleaned Lab-space colour.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pyrochrome.pipeline.download import find_glazes_csv
from pyrochrome.pipeline.features import build_features, clean_recipes
from pyrochrome.pipeline.labels import color_family, surface_family

MIN_CLASS_COUNT = 40  # drop colour families with fewer samples than this


@dataclass
class TargetData:
    """A model-ready dataset for one prediction target.

    Attributes:
        name: Target identifier (``"surface"`` or ``"colour"``).
        title: Human-readable title for reports.
        X: Feature matrix, median-imputed, one row per labelled recipe.
        y: Label vector aligned to ``X``.
        feature_names: Ordered feature column names.
    """

    name: str
    title: str
    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]


def _impute(features: pd.DataFrame) -> pd.DataFrame:
    """Coerce to numeric and fill missing values with each column's median."""
    numeric = features.apply(pd.to_numeric, errors="coerce")
    imputed: pd.DataFrame = numeric.fillna(numeric.median())
    return imputed


def load_targets(csv_path: str | None = None) -> dict[str, TargetData]:
    """Load and prepare the (X, y) dataset for every target.

    Args:
        csv_path: Explicit path to the Glazy CSV, or ``None`` to auto-detect the
            latest one under ``data/raw/glazy-data/``.

    Returns:
        Mapping ``target name -> TargetData``.
    """
    path = csv_path or str(find_glazes_csv())
    df = clean_recipes(pd.read_csv(path, low_memory=False))
    feat_cols = build_features(df)

    targets: dict[str, TargetData] = {}

    # Surface.
    surface = df["surface_type"].apply(surface_family)
    mask = surface.notna()
    targets["surface"] = TargetData(
        name="surface",
        title="Surface (Glossy / Matte / Satin)",
        X=_impute(df.loc[mask, feat_cols]),
        y=surface[mask].astype("object"),
        feature_names=feat_cols,
    )

    # Colour family (drop rare classes).
    colour = df.apply(lambda r: color_family(r["rgb_r"], r["rgb_g"], r["rgb_b"]), axis=1)
    counts = colour.value_counts()
    colour = colour.where(~colour.isin(counts[counts < MIN_CLASS_COUNT].index))
    mask = colour.notna()
    targets["colour"] = TargetData(
        name="colour",
        title="Colour family (10 classes, from RGB)",
        X=_impute(df.loc[mask, feat_cols]),
        y=colour[mask].astype("object"),
        feature_names=feat_cols,
    )

    return targets
