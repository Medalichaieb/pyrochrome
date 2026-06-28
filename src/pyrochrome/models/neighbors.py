"""k-NN nearest real recipes (lever #2): show real tiles, not just a prediction.

Because predicted colour is bounded by photo-label noise, the honest product move
is to surface the **nearest real recipes** (with their actual photos/RGB) next to
any prediction. This module builds a k-NN index over standardised chemistry
features and, for a query composition, returns the closest real Glazy recipes.

We search in **chemistry space** (UMF + cone + atmosphere) so neighbours are
recipes a ceramicist could actually mix and fire — "recipes like yours", whose
real fired results bracket the likely outcome.

Run: ``uv run python -m pyrochrome.models.neighbors`` (``make neighbors``) to
build + persist the index and print a demo query.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from pyrochrome.models.datasets import RegressionData, load_colour_regression
from pyrochrome.pipeline.color import srgb_to_lab

OUTPUT_DIR = Path("models_out")
DEFAULT_K = 8

# UMF oxide columns carried on each neighbour so the frontend can place the
# query recipe on a radar chart against the real recipes around it.
RADAR_OXIDES = [
    "SiO2_umf",
    "Al2O3_umf",
    "B2O3_umf",
    "Na2O_umf",
    "K2O_umf",
    "CaO_umf",
    "MgO_umf",
    "Fe2O3_umf",
    "CuO_umf",
    "CoO_umf",
    "MnO_umf",
    "Cr2O3_umf",
]


@dataclass
class RecipeIndex:
    """A fitted nearest-recipe index plus the reference recipes it returns.

    Attributes:
        scaler: Fitted standardiser for the chemistry features.
        nn: Fitted NearestNeighbors model.
        feature_names: Feature columns the index expects (in order).
        reference: Recipes (id, name, rgb_*, lab_*) aligned to the index rows.
    """

    scaler: StandardScaler
    nn: NearestNeighbors
    feature_names: list[str]
    reference: pd.DataFrame


def build_index(data: RegressionData, n_neighbors: int = DEFAULT_K) -> RecipeIndex:
    """Fit the k-NN index over standardised chemistry features.

    Args:
        data: The regression dataset (X + reference rows with id/name/rgb).
        n_neighbors: Default neighbour count the index is fitted for.

    Returns:
        A fitted :class:`RecipeIndex`.
    """
    scaler = StandardScaler().fit(data.X.to_numpy())
    scaled = scaler.transform(data.X.to_numpy())
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(scaled)

    reference = data.reference.reset_index(drop=True).copy()
    lab = srgb_to_lab(reference[["rgb_r", "rgb_g", "rgb_b"]].to_numpy())
    reference[["lab_l", "lab_a", "lab_b"]] = lab
    # Carry the radar oxides (those present) so neighbours include their chemistry.
    radar_cols = [c for c in RADAR_OXIDES if c in data.X.columns]
    reference[radar_cols] = data.X[radar_cols].reset_index(drop=True)
    return RecipeIndex(scaler=scaler, nn=nn, feature_names=data.feature_names, reference=reference)


def save_index(index: RecipeIndex, path: str | Path) -> None:
    """Persist the index as a plain dict (not the dataclass).

    Dumping a plain dict avoids the pickle pitfall where a dataclass defined in a
    module run via ``python -m`` is recorded under ``__main__`` and then fails to
    load in another process (e.g. the API server).

    Args:
        index: The fitted :class:`RecipeIndex`.
        path: Destination ``.joblib`` path.
    """
    joblib.dump(
        {
            "scaler": index.scaler,
            "nn": index.nn,
            "feature_names": index.feature_names,
            "reference": index.reference,
        },
        path,
    )


def load_index(path: str | Path) -> RecipeIndex:
    """Load an index persisted by :func:`save_index`.

    Args:
        path: Path to the ``.joblib`` written by :func:`save_index`.

    Returns:
        The reconstructed :class:`RecipeIndex`.
    """
    bundle = joblib.load(path)
    return RecipeIndex(**bundle)


def query(index: RecipeIndex, x_row: np.ndarray, k: int = DEFAULT_K) -> pd.DataFrame:
    """Return the ``k`` nearest real recipes to a query feature vector.

    Args:
        index: A fitted :class:`RecipeIndex`.
        x_row: A 1-D feature vector in the index's ``feature_names`` order.
        k: Number of neighbours to return.

    Returns:
        A dataframe of the nearest recipes (reference columns + ``distance``),
        ordered nearest first.
    """
    scaled = index.scaler.transform(np.asarray(x_row, dtype=float).reshape(1, -1))
    distances, indices = index.nn.kneighbors(scaled, n_neighbors=k)
    out = index.reference.iloc[indices[0]].copy()
    out["distance"] = distances[0]
    result: pd.DataFrame = out.reset_index(drop=True)
    return result


def main() -> None:
    """Entry point for ``make neighbors``: build, persist and demo the index."""
    data = load_colour_regression()
    index = build_index(data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "neighbors.joblib"
    save_index(index, path)
    print(f"Built nearest-recipe index over {len(data.Y)} recipes -> {path}")

    # Demo: neighbours of the first recipe (should include itself at distance 0).
    demo = query(index, data.X.to_numpy()[0], k=5)
    cols = [c for c in ["name", "rgb_r", "rgb_g", "rgb_b", "distance"] if c in demo.columns]
    print("\nDemo — 5 nearest recipes to row 0:")
    print(demo[cols].to_string(index=False))


if __name__ == "__main__":
    main()
