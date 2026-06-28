import gzip
from pathlib import Path

import pandas as pd

from pyrochrome.pipeline.atmosphere import (
    ATMOSPHERE_COLUMNS,
    ATMOSPHERE_KNOWN_COLUMN,
    join_atmosphere,
    normalise_atmospheres,
    parse_atmospheres,
)


def test_normalise_handles_lists_multilabel_and_aliases() -> None:
    assert normalise_atmospheres(["Oxidation", "Reduction"]) == {"oxidation", "reduction"}
    assert normalise_atmospheres(["Salt & Soda"]) == {"salt_soda"}
    assert normalise_atmospheres("Reducing") == {"reduction"}
    assert normalise_atmospheres(["Wood", "Raku", "Luster"]) == {"wood", "raku", "luster"}
    assert normalise_atmospheres(None) == set()
    assert normalise_atmospheres(["something else"]) == set()


def test_parse_atmospheres_multihot_from_gzipped_yaml(tmp_path: Path) -> None:
    yaml_text = (
        "-\n"
        "  ID: 1\n"
        "  Atmospheres: ['Oxidation', 'Reduction']\n"
        "-\n"
        "  ID: 2\n"
        "  Atmospheres: ['Salt & Soda']\n"
        "-\n"
        "  ID: 3\n"  # no atmosphere field -> not emitted
    )
    path = tmp_path / "glazy_test.yaml.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(yaml_text)

    df = parse_atmospheres(path)
    assert list(df.columns) == ["id", *ATMOSPHERE_COLUMNS]
    assert set(df["id"]) == {1, 2}  # id=3 has no atmosphere, dropped
    row1 = df[df["id"] == 1].iloc[0]
    assert row1["atm_oxidation"] == 1 and row1["atm_reduction"] == 1
    assert row1["atm_wood"] == 0
    row2 = df[df["id"] == 2].iloc[0]
    assert row2["atm_salt_soda"] == 1 and row2["atm_oxidation"] == 0


def test_join_atmosphere_marks_unknown() -> None:
    features = pd.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30]})
    atmospheres = pd.DataFrame(
        {
            "id": [1, 2],
            **{c: [1 if c == "atm_oxidation" else 0 for _ in range(2)] for c in ATMOSPHERE_COLUMNS},
        }
    )
    joined = join_atmosphere(features, atmospheres)
    # Known rows keep their flags; unknown row gets all-zero + atm_known=0.
    assert joined.loc[joined["id"] == 1, "atm_oxidation"].item() == 1
    assert joined.loc[joined["id"] == 3, ATMOSPHERE_KNOWN_COLUMN].item() == 0
    assert joined.loc[joined["id"] == 1, ATMOSPHERE_KNOWN_COLUMN].item() == 1
    assert joined.loc[joined["id"] == 3, ATMOSPHERE_COLUMNS].to_numpy().sum() == 0
