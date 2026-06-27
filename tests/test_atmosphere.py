import gzip
from pathlib import Path

import pandas as pd

from pyrochrome.pipeline.atmosphere import (
    _normalise_atmosphere,
    join_atmosphere,
    parse_atmospheres,
)


def test_normalise_handles_strings_lists_dicts() -> None:
    assert _normalise_atmosphere("Oxidation") == "oxidation"
    assert _normalise_atmosphere("reducing") == "reduction"
    assert _normalise_atmosphere(["Neutral", "Oxidation"]) == "neutral"
    assert _normalise_atmosphere({"name": "Reduction"}) == "reduction"
    assert _normalise_atmosphere(None) is None
    assert _normalise_atmosphere("unknown-thing") is None


def test_parse_atmospheres_from_gzipped_yaml(tmp_path: Path) -> None:
    yaml_text = (
        "- id: 1\n"
        "  Atmospheres: Oxidation\n"
        "- id: 2\n"
        "  Atmospheres:\n"
        "    - Reduction\n"
        "- id: 3\n"  # no atmosphere field
    )
    path = tmp_path / "glazy_test.yaml.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(yaml_text)

    df = parse_atmospheres(path)
    assert list(df.columns) == ["id", "atmosphere"]
    by_id = dict(zip(df["id"], df["atmosphere"], strict=True))
    assert by_id == {1: "oxidation", 2: "reduction", 3: None}


def test_join_atmosphere_left_join() -> None:
    features = pd.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30]})
    atmospheres = pd.DataFrame({"id": [1, 2], "atmosphere": ["oxidation", "reduction"]})
    joined = join_atmosphere(features, atmospheres)
    assert joined.loc[joined["id"] == 1, "atmosphere"].item() == "oxidation"
    assert pd.isna(joined.loc[joined["id"] == 3, "atmosphere"].item())
