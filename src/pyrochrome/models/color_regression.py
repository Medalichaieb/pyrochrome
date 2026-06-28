"""Lab colour regression (lever #2): predict CIELAB instead of noisy families.

Predicting (L*, a*, b*) and reporting error as **ΔE** (CIE76) is the
product-meaningful way to express colour error and sidesteps the brittle
10-family classification. This module cross-validates a few multi-output
regressors, reports ΔE / per-channel MAE / R², and trains + persists the best.

The honest framing: colour labels come from non-standardised photos, so even a
perfect model is bounded by that noise. ΔE makes the residual explicit, and the
k-NN neighbours (:mod:`pyrochrome.models.neighbors`) give real example tiles to
show alongside any predicted colour.

Run: ``uv run python -m pyrochrome.models.color_regression`` (``make color``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pyrochrome.models.datasets import RegressionData, load_colour_regression
from pyrochrome.pipeline.color import delta_e_cie76

RANDOM_STATE = 42
N_SPLITS = 5
OUTPUT_DIR = Path("models_out")

RegressorFactory = Callable[[], RegressorMixin]

# Candidate multi-output regressors (all predict the 3 Lab channels at once).
MODELS: dict[str, RegressorFactory] = {
    "mean": lambda: DummyRegressor(strategy="mean"),
    "knn": lambda: make_pipeline(
        StandardScaler(), KNeighborsRegressor(n_neighbors=10, weights="distance")
    ),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
    ),
    "extra_trees": lambda: ExtraTreesRegressor(
        n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE
    ),
}


@dataclass
class RegressionResult:
    """Cross-validated colour-regression metrics for one model.

    Attributes:
        model: Model identifier.
        delta_e_mean: Mean ΔE (CIE76) across folds — the headline metric.
        delta_e_median: Median ΔE across folds.
        mae_l: Mean absolute error on L*.
        mae_a: Mean absolute error on a*.
        mae_b: Mean absolute error on b*.
        r2: Mean R² (uniform-averaged over the 3 channels).
    """

    model: str
    delta_e_mean: float
    delta_e_median: float
    mae_l: float
    mae_a: float
    mae_b: float
    r2: float


def cross_validate(data: RegressionData, factory: RegressorFactory, name: str) -> RegressionResult:
    """Run k-fold CV for one regressor and aggregate ΔE / MAE / R².

    Args:
        data: The Lab regression dataset.
        factory: Callable returning a fresh regressor.
        name: Model identifier for reporting.

    Returns:
        A :class:`RegressionResult`.
    """
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    x_values = data.X.to_numpy()
    y_values = data.Y

    de_means: list[float] = []
    de_medians: list[float] = []
    maes: list[np.ndarray] = []
    r2s: list[float] = []

    for train_idx, test_idx in kf.split(x_values):
        model = factory()
        model.fit(x_values[train_idx], y_values[train_idx])
        pred = np.asarray(model.predict(x_values[test_idx]))
        true = y_values[test_idx]
        de = delta_e_cie76(pred, true)
        de_means.append(float(np.mean(de)))
        de_medians.append(float(np.median(de)))
        maes.append(np.array([mean_absolute_error(true[:, c], pred[:, c]) for c in range(3)]))
        r2s.append(float(r2_score(true, pred, multioutput="uniform_average")))

    mae = np.mean(maes, axis=0)
    return RegressionResult(
        model=name,
        delta_e_mean=float(np.mean(de_means)),
        delta_e_median=float(np.mean(de_medians)),
        mae_l=float(mae[0]),
        mae_a=float(mae[1]),
        mae_b=float(mae[2]),
        r2=float(np.mean(r2s)),
    )


def compare(data: RegressionData) -> list[RegressionResult]:
    """Cross-validate every candidate regressor, sorted by mean ΔE (lower first).

    Args:
        data: The Lab regression dataset.

    Returns:
        Results sorted from best (lowest ΔE) to worst.
    """
    print(f"Colour regression (Lab)  —  n={len(data.Y)}, features={len(data.feature_names)}")
    cols = ["dE_mean", "dE_med", "MAE_L", "MAE_a", "MAE_b", "R2"]
    print(f"{'model':16s} " + " ".join(f"{c:>8s}" for c in cols))
    results = [cross_validate(data, factory, name) for name, factory in MODELS.items()]
    results.sort(key=lambda r: r.delta_e_mean)
    for r in results:
        print(
            f"{r.model:16s} {r.delta_e_mean:9.2f} {r.delta_e_median:9.2f} "
            f"{r.mae_l:7.2f} {r.mae_a:7.2f} {r.mae_b:7.2f} {r.r2:7.3f}"
        )
    return results


def main() -> None:
    """Entry point for ``make color``: compare regressors and persist the best."""
    data = load_colour_regression()
    results = compare(data)
    best = results[0]
    print(f"\nBest by ΔE: {best.model} (ΔE={best.delta_e_mean:.2f})")

    model = MODELS[best.model]()
    model.fit(data.X.to_numpy(), data.Y)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "colour_lab.joblib"
    joblib.dump(
        {
            "target": "colour_lab",
            "model": model,
            "feature_names": data.feature_names,
            "selected_regressor": best.model,
            "cv_delta_e": best.delta_e_mean,
        },
        path,
    )
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
