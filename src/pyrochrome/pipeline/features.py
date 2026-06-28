"""Feature engineering: build the model input matrix from a Glazy dataframe.

Inputs : a cleaned Glazy dataframe (real recipes only — see ``clean_recipes``).
Outputs: the list of feature column names, plus the dataframe augmented with a
         ``cone_num`` ordinal column.

Feature representation (validated):

    UMF oxides (precomputed by Glazy, columns ending in ``_umf``)
      + aggregates R2O_umf, RO_umf, SiO2_Al2O3_ratio_umf  (already ``_umf`` cols)
      + cone (ordinal, ``cone_num``)
      + atmosphere (categorical)   ← added once joined from the YAML dump

All-zero UMF columns are dropped (they carry no signal). Missing numeric values
are imputed with the column median at training time (see ``models.baseline``).
"""

from __future__ import annotations

import pandas as pd

from pyrochrome.pipeline.cones import cone_to_ordinal

# Colorant oxides of interest, for interpretability / examples (UMF columns).
COLORANTS: dict[str, str] = {
    "CoO_umf": "cobalt",
    "CuO_umf": "copper",
    "Fe2O3_umf": "iron",
    "Cr2O3_umf": "chromium",
    "MnO_umf": "manganese",
    "NiO_umf": "nickel",
    "TiO2_umf": "titanium",
    "SnO2_umf": "tin",
    "ZrO2_umf": "zirconium",
}


def clean_recipes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only real, fired glaze recipes.

    Drops chemical analyses, primitive materials and theoretical entries, plus
    recipes with no silica in their UMF — overglazes such as lusters and metallics
    that carry no glass chemistry and would pollute both the targets and the
    nearest-recipe search (they have no oxides to compare on).

    Args:
        df: Raw Glazy glazes dataframe.

    Returns:
        A filtered copy of ``df``.
    """
    silica = pd.to_numeric(df["SiO2_umf"], errors="coerce").fillna(0)
    mask = (
        (df["is_analysis"] == 0)
        & (df["is_primitive"] == 0)
        & (df["is_theoretical"] == 0)
        & (silica > 0)
    )
    return df[mask].copy()


def build_features(df: pd.DataFrame) -> list[str]:
    """Add the ordinal cone column and return the feature column list.

    Mutates ``df`` in place by adding ``cone_num``. Selects all ``*_umf``
    columns (which already include the R2O/RO aggregates and the SiO2:Al2O3
    ratio) plus ``cone_num``, then drops columns that are entirely zero.

    Args:
        df: Cleaned Glazy dataframe (see :func:`clean_recipes`).

    Returns:
        The ordered list of feature column names to feed the model.
    """
    df["cone_num"] = df["from_orton_cone"].apply(cone_to_ordinal)
    umf_cols = [c for c in df.columns if c.endswith("_umf")]
    feat_cols = list(dict.fromkeys([*umf_cols, "cone_num"]))
    numeric = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    # Drop features that are all-zero / all-missing (no signal).
    return [c for c in feat_cols if numeric[c].fillna(0).abs().sum() > 0]
