from pathlib import Path

import numpy as np
import pandas as pd

from pyrochrome.models.datasets import RegressionData
from pyrochrome.models.neighbors import build_index, load_index, query, save_index


def _toy_data() -> RegressionData:
    # Four recipes on a line in 2-D chemistry space, distinct colours.
    X = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 9.0], "f2": [0.0, 1.0, 2.0, 9.0]})
    reference = pd.DataFrame(
        {
            "id": [10, 11, 12, 13],
            "name": ["a", "b", "c", "far"],
            "rgb_r": [255, 0, 0, 10],
            "rgb_g": [255, 0, 0, 10],
            "rgb_b": [255, 0, 255, 10],
        }
    )
    return RegressionData(X=X, Y=np.zeros((4, 3)), feature_names=["f1", "f2"], reference=reference)


def test_build_index_adds_lab_columns() -> None:
    index = build_index(_toy_data(), n_neighbors=2)
    assert {"lab_l", "lab_a", "lab_b"}.issubset(index.reference.columns)
    assert index.feature_names == ["f1", "f2"]


def test_query_returns_sorted_neighbors_with_self_first() -> None:
    data = _toy_data()
    index = build_index(data, n_neighbors=4)
    result = query(index, data.X.to_numpy()[0], k=3)
    assert len(result) == 3
    # Nearest to row 0 is itself (distance 0), and distances are non-decreasing.
    assert result.iloc[0]["name"] == "a"
    assert result.iloc[0]["distance"] == 0.0
    assert list(result["distance"]) == sorted(result["distance"])
    # The far outlier should not be among the 3 nearest of row 0.
    assert "far" not in set(result["name"])


def test_save_load_index_roundtrip(tmp_path: Path) -> None:
    data = _toy_data()
    index = build_index(data, n_neighbors=3)
    path = tmp_path / "neighbors.joblib"
    save_index(index, path)
    reloaded = load_index(path)
    assert reloaded.feature_names == index.feature_names
    # Reloaded index still queries identically.
    before = query(index, data.X.to_numpy()[0], k=2)
    after = query(reloaded, data.X.to_numpy()[0], k=2)
    assert list(before["name"]) == list(after["name"])
