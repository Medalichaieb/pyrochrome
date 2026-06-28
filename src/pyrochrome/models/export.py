"""Export compact, browser-runnable models to JSON (``make export``).

The deployed site is fully static — no backend — so prediction runs in the
browser. The full pipeline still selects HistGradientBoosting (see the report);
for the client we train small, faithful **MLPs** (the pattern the
``glaze_predictor.html`` prototype proved) whose weights serialise to a few KB,
plus the nearest-recipe data for an in-browser k-NN.

Outputs:
    - ``web/src/model/classifier_<target>.json`` — surface / transparency / colour
    - ``web/src/model/regressor_lab.json``       — CIELAB regressor
    - ``web/public/recipes.json``                — k-NN recipes (lazy-loaded)

Each model JSON carries its feature order, the input scaler (mean/std) and the
dense layers, so the browser rebuilds the exact same feature vector and forward
pass. Run after ``make data``: ``make export``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from pyrochrome.models.datasets import (
    RegressionData,
    TargetData,
    load_colour_regression,
    load_targets,
)
from pyrochrome.models.neighbors import RADAR_OXIDES
from pyrochrome.pipeline.color import delta_e_cie76

RANDOM_STATE = 42
MODEL_DIR = Path("web/src/model")
PUBLIC_DIR = Path("web/public")


def _round(values: Any, decimals: int = 6) -> list[float]:
    """Round a 1-D array to a JSON-friendly list of floats."""
    return [round(float(v), decimals) for v in np.asarray(values).ravel()]


def _dense_layers(mlp: MLPClassifier | MLPRegressor) -> list[dict[str, Any]]:
    """Serialise an sklearn MLP's dense layers (weights + biases)."""
    layers = []
    for weight, bias in zip(mlp.coefs_, mlp.intercepts_, strict=True):
        layers.append(
            {
                "w": [_round(row) for row in np.asarray(weight)],  # (in, out)
                "b": _round(bias),
            }
        )
    return layers


def _scaler(pipeline: Pipeline) -> dict[str, list[float]]:
    scaler: StandardScaler = pipeline.named_steps["standardscaler"]
    return {"mean": _round(scaler.mean_), "std": _round(scaler.scale_)}


def export_classifier(data: TargetData) -> float:
    """Train + export a compact MLP classifier; return held-out accuracy."""
    x_train, x_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=0.2, random_state=RANDOM_STATE, stratify=data.y
    )
    pipe = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(96, 48),
            alpha=1e-3,
            max_iter=1200,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    ).fit(x_train, y_train)
    accuracy = float(accuracy_score(y_test, pipe.predict(x_test)))

    mlp: MLPClassifier = pipe.named_steps["mlpclassifier"]
    payload = {
        "kind": "classifier",
        "target": data.name,
        "feature_names": data.feature_names,
        "scaler": _scaler(pipe),
        "layers": _dense_layers(mlp),
        "classes": [str(c) for c in mlp.classes_],
    }
    (MODEL_DIR / f"classifier_{data.name}.json").write_text(json.dumps(payload), encoding="utf-8")
    return accuracy


def export_regressor(data: RegressionData) -> float:
    """Train + export a compact MLP Lab regressor; return held-out mean ΔE."""
    x_train, x_test, y_train, y_test = train_test_split(
        data.X, data.Y, test_size=0.2, random_state=RANDOM_STATE
    )
    pipe = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(96, 48),
            alpha=1e-3,
            max_iter=1500,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    ).fit(x_train, y_train)
    pred = np.asarray(pipe.predict(x_test))
    mean_delta_e = float(np.mean(delta_e_cie76(pred, y_test)))

    mlp: MLPRegressor = pipe.named_steps["mlpregressor"]
    payload = {
        "kind": "regressor",
        "target": "colour_lab",
        "feature_names": data.feature_names,
        "scaler": _scaler(pipe),
        "layers": _dense_layers(mlp),
        "outputs": ["L", "a", "b"],
    }
    (MODEL_DIR / "regressor_lab.json").write_text(json.dumps(payload), encoding="utf-8")
    return mean_delta_e


def export_neighbours(data: RegressionData) -> int:
    """Export standardised recipe vectors + display data for in-browser k-NN."""
    scaler = StandardScaler().fit(data.X.to_numpy())
    standardised = scaler.transform(data.X.to_numpy())
    radar_cols = [c for c in RADAR_OXIDES if c in data.X.columns]
    radar_values = data.X[radar_cols].to_numpy()
    reference = data.reference

    recipes = []
    for i in range(len(standardised)):
        recipes.append(
            {
                "name": str(reference.iloc[i].get("name", "Untitled")),
                "rgb": [
                    int(reference.iloc[i]["rgb_r"]),
                    int(reference.iloc[i]["rgb_g"]),
                    int(reference.iloc[i]["rgb_b"]),
                ],
                "ox": [round(float(v), 4) for v in radar_values[i]],
                "v": [round(float(v), 2) for v in standardised[i]],
            }
        )

    payload = {
        "feature_names": data.feature_names,
        "scaler": {"mean": _round(scaler.mean_), "std": _round(scaler.scale_)},
        "radar_oxides": radar_cols,
        "recipes": recipes,
    }
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    (PUBLIC_DIR / "recipes.json").write_text(json.dumps(payload), encoding="utf-8")
    return len(recipes)


def main() -> None:
    """Entry point for ``make export``: write all browser model JSON."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    targets = load_targets()
    print("Compact in-browser models (held-out):")
    for name in ("surface", "transparency", "colour"):
        acc = export_classifier(targets[name])
        print(f"  {name:13s} MLP accuracy={acc:.3f}")

    regression = load_colour_regression()
    delta_e = export_regressor(regression)
    print(f"  colour_lab    MLP mean ΔE={delta_e:.2f}")

    count = export_neighbours(regression)
    print(f"  neighbours    {count} recipes -> {PUBLIC_DIR / 'recipes.json'}")


if __name__ == "__main__":
    main()
