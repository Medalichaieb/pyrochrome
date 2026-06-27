"""Train and persist the selected model for every target.

This is the production trainer (wired to ``make train``). For each target it:
    1. loads the shared (X, y) dataset (:mod:`pyrochrome.models.datasets`),
    2. holds out a stratified 20% test set (seed=42),
    3. fits the selected model (:func:`pyrochrome.models.selected.build_selected_model`),
    4. reports held-out accuracy / macro-F1 / (colour) top-2,
    5. saves the fitted estimator + metadata to ``models_out/<target>.joblib``.

The saved artifacts are what :mod:`pyrochrome.api.main` loads to serve predictions.

Inputs : Glazy CSV (auto-detected) via the dataset loader.
Outputs: ``models_out/surface.joblib``, ``models_out/colour.joblib``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split

from pyrochrome.models.datasets import TargetData, load_targets
from pyrochrome.models.selected import SELECTED_MODEL_NAME, build_selected_model

RANDOM_STATE = 42
OUTPUT_DIR = Path("models_out")


@dataclass
class TrainedModel:
    """A fitted model plus the metadata needed to serve it.

    Attributes:
        target: Target name (``"surface"`` / ``"colour"``).
        model: The fitted estimator.
        feature_names: Ordered feature columns the model expects.
        classes: Class labels in the model's output order.
        accuracy: Held-out accuracy.
        f1_macro: Held-out macro-F1.
        top2: Held-out top-2 accuracy (None if < 3 classes).
    """

    target: str
    model: ClassifierMixin
    feature_names: list[str]
    classes: list[str]
    accuracy: float
    f1_macro: float
    top2: float | None


def train_one(data: TargetData) -> TrainedModel:
    """Fit and evaluate the selected model for one target.

    Args:
        data: Prepared target dataset.

    Returns:
        A :class:`TrainedModel` with the fitted estimator and held-out metrics.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=0.2, random_state=RANDOM_STATE, stratify=data.y
    )
    model = build_selected_model()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    top2: float | None = None
    if data.y.nunique() >= 3:
        proba = model.predict_proba(x_test)
        top2 = float(top_k_accuracy_score(y_test, proba, k=2, labels=model.classes_))

    return TrainedModel(
        target=data.name,
        model=model,
        feature_names=data.feature_names,
        classes=[str(c) for c in model.classes_],
        accuracy=float(accuracy_score(y_test, pred)),
        f1_macro=float(f1_score(y_test, pred, average="macro")),
        top2=top2,
    )


def save(trained: TrainedModel, output_dir: Path = OUTPUT_DIR) -> Path:
    """Persist a trained model + metadata to ``<output_dir>/<target>.joblib``.

    Args:
        trained: The fitted model bundle.
        output_dir: Directory to write into (created if missing).

    Returns:
        The path the artifact was written to.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{trained.target}.joblib"
    joblib.dump(
        {
            "target": trained.target,
            "model": trained.model,
            "feature_names": trained.feature_names,
            "classes": trained.classes,
            "selected_model": SELECTED_MODEL_NAME,
        },
        path,
    )
    return path


def main() -> None:
    """Entry point for ``make train``: train, evaluate and persist all targets."""
    print(f"Selected model: {SELECTED_MODEL_NAME}")
    for data in load_targets().values():
        trained = train_one(data)
        top2 = f"   top-2={trained.top2:.3f}" if trained.top2 is not None else ""
        print(
            f"\n{data.title}\n  n={len(data.y)}  classes={len(trained.classes)}\n"
            f"  accuracy={trained.accuracy:.3f}   macro-F1={trained.f1_macro:.3f}{top2}"
        )
        path = save(trained)
        print(f"  saved -> {path}")


if __name__ == "__main__":
    main()
