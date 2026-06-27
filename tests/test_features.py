import pandas as pd

from pyrochrome.pipeline.features import build_features, clean_recipes


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "is_analysis": [0, 1, 0],
            "is_primitive": [0, 0, 0],
            "is_theoretical": [0, 0, 1],
            "from_orton_cone": ["6", "10", "04"],
            "SiO2_umf": [3.0, 2.5, 4.0],
            "Al2O3_umf": [0.4, 0.3, 0.5],
            "ZeroOxide_umf": [0.0, 0.0, 0.0],  # all-zero → must be dropped
        }
    )


def test_clean_recipes_drops_non_real() -> None:
    cleaned = clean_recipes(_sample_df())
    # Only id=1 is a real, non-theoretical, non-analysis recipe.
    assert cleaned["id"].tolist() == [1]


def test_build_features_adds_cone_and_drops_zero_columns() -> None:
    df = clean_recipes(
        pd.DataFrame(
            {
                "id": [1, 2],
                "is_analysis": [0, 0],
                "is_primitive": [0, 0],
                "is_theoretical": [0, 0],
                "from_orton_cone": ["6", "04"],
                "SiO2_umf": [3.0, 4.0],
                "ZeroOxide_umf": [0.0, 0.0],
            }
        )
    )
    feat_cols = build_features(df)
    assert "cone_num" in feat_cols
    assert "SiO2_umf" in feat_cols
    assert "ZeroOxide_umf" not in feat_cols  # dropped (all zero)
    assert df["cone_num"].notna().all()
