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

import numpy as np
import pandas as pd

from pyrochrome.pipeline.atmosphere import (
    ATMOSPHERE_COLUMNS,
    ATMOSPHERE_KNOWN_COLUMN,
    join_atmosphere,
    load_atmospheres,
)
from pyrochrome.pipeline.color import srgb_to_lab
from pyrochrome.pipeline.download import find_glazes_csv
from pyrochrome.pipeline.features import build_features, clean_recipes
from pyrochrome.pipeline.labels import color_family, surface_family, transparency_family

MIN_CLASS_COUNT = 40  # drop colour families with fewer samples than this
RGB_COLUMNS = ["rgb_r", "rgb_g", "rgb_b"]


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


@dataclass
class RegressionData:
    """A model-ready dataset for the Lab colour-regression target.

    Attributes:
        X: Feature matrix, median-imputed, one row per recipe with valid RGB.
        Y: Target matrix of shape ``(n, 3)`` = predicted CIELAB (L*, a*, b*).
        feature_names: Ordered feature column names.
        reference: Aligned frame with ``id``, ``name`` and ``rgb_*`` for display
            and nearest-recipe lookups.
    """

    X: pd.DataFrame
    Y: np.ndarray
    feature_names: list[str]
    reference: pd.DataFrame


def _impute(features: pd.DataFrame) -> pd.DataFrame:
    """Coerce to numeric and fill missing values with each column's median."""
    numeric = features.apply(pd.to_numeric, errors="coerce")
    imputed: pd.DataFrame = numeric.fillna(numeric.median())
    return imputed


def _feature_frame(
    csv_path: str | None, *, with_atmosphere: bool
) -> tuple[pd.DataFrame, list[str]]:
    """Load, clean, build features and (optionally) join atmosphere.

    Shared by every loader so all experiments use identical preprocessing.

    Args:
        csv_path: Explicit Glazy CSV path, or ``None`` to auto-detect.
        with_atmosphere: Whether to join the multi-hot atmosphere features.

    Returns:
        The recipes dataframe and the ordered feature column names.
    """
    path = csv_path or str(find_glazes_csv())
    df = clean_recipes(pd.read_csv(path, low_memory=False))
    feat_cols = build_features(df)
    if with_atmosphere:
        df = join_atmosphere(df, load_atmospheres())
        feat_cols = [*feat_cols, *ATMOSPHERE_COLUMNS, ATMOSPHERE_KNOWN_COLUMN]
    return df, feat_cols


def load_targets(
    csv_path: str | None = None, *, with_atmosphere: bool = True
) -> dict[str, TargetData]:
    """Load and prepare the (X, y) dataset for every classification target.

    Args:
        csv_path: Explicit path to the Glazy CSV, or ``None`` to auto-detect the
            latest one under ``data/raw/glazy-data/``.
        with_atmosphere: If ``True`` (default), join the multi-hot atmosphere
            features (lever #1) from the YAML dump. Set ``False`` to reproduce
            the chemistry-only setup and measure the atmosphere gain.

    Returns:
        Mapping ``target name -> TargetData``.
    """
    df, feat_cols = _feature_frame(csv_path, with_atmosphere=with_atmosphere)

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

    # Transparency.
    transparency = df["transparency_type"].apply(transparency_family)
    mask = transparency.notna()
    targets["transparency"] = TargetData(
        name="transparency",
        title="Transparency (Opaque / Semi-opaque / Translucent / Transparent)",
        X=_impute(df.loc[mask, feat_cols]),
        y=transparency[mask].astype("object"),
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


def load_colour_regression(
    csv_path: str | None = None, *, with_atmosphere: bool = True
) -> RegressionData:
    """Load the Lab colour-regression dataset (lever #2).

    Uses every recipe with a valid RGB triple (no rare-class dropping, so more
    data than the classification target) and converts RGB → CIELAB as the
    regression target.

    Args:
        csv_path: Explicit Glazy CSV path, or ``None`` to auto-detect.
        with_atmosphere: Whether to include the atmosphere features.

    Returns:
        A :class:`RegressionData` bundle (X, Lab Y, feature names, reference rows).
    """
    df, feat_cols = _feature_frame(csv_path, with_atmosphere=with_atmosphere)

    rgb = df[RGB_COLUMNS].apply(pd.to_numeric, errors="coerce")
    mask = rgb.notna().all(axis=1)
    lab = srgb_to_lab(rgb[mask].to_numpy())

    reference_cols = [c for c in ["id", "name", *RGB_COLUMNS] if c in df.columns]
    return RegressionData(
        X=_impute(df.loc[mask, feat_cols]),
        Y=lab,
        feature_names=feat_cols,
        reference=df.loc[mask, reference_cols].reset_index(drop=True),
    )
