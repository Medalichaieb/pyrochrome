"""Compare candidate models per target with stratified cross-validation.

For each target and each candidate model we run stratified k-fold CV and report
the mean ± std of accuracy, macro-F1 and (for multi-class colour) top-2 accuracy.
This is the experiment harness behind the model-selection decision documented in
``MODEL_CARD.md`` / ``results.md``.

Models compared (all CPU-only, no GPU):
    - majority      : DummyClassifier (sanity floor)
    - logreg        : StandardScaler + multinomial LogisticRegression
    - random_forest : RandomForestClassifier (the current baseline)
    - extra_trees   : ExtraTreesClassifier
    - hist_gb       : HistGradientBoostingClassifier (tuned)
    - mlp           : StandardScaler + MLPClassifier (compact, browser-portable)

Run: ``uv run python -m pyrochrome.models.compare`` (or ``make compare``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from pyrochrome.models.datasets import TargetData, load_targets
from pyrochrome.models.selected import build_selected_model

RANDOM_STATE = 42
N_SPLITS = 5

# Candidate model factories. Each returns a fresh, unfitted estimator.
ModelFactory = Callable[[], ClassifierMixin]

MODELS: dict[str, ModelFactory] = {
    "majority": lambda: DummyClassifier(strategy="most_frequent"),
    "logreg": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0),
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    ),
    "extra_trees": lambda: ExtraTreesClassifier(
        n_estimators=400,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    ),
    # The selected production model — same instance the trainer/API use.
    "hist_gb": build_selected_model,
    "mlp": lambda: make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            alpha=1e-3,
            max_iter=600,
            random_state=RANDOM_STATE,
        ),
    ),
}


@dataclass
class CVResult:
    """Cross-validated metrics for one (model, target) pair.

    Attributes:
        model: Model identifier.
        acc_mean: Mean accuracy across folds.
        acc_std: Std of accuracy across folds.
        f1_mean: Mean macro-F1 across folds.
        f1_std: Std of macro-F1 across folds.
        top2_mean: Mean top-2 accuracy (None if < 3 classes).
        top2_std: Std of top-2 accuracy (None if < 3 classes).
    """

    model: str
    acc_mean: float
    acc_std: float
    f1_mean: float
    f1_std: float
    top2_mean: float | None
    top2_std: float | None


def _supports_proba(estimator: ClassifierMixin) -> bool:
    """Whether the estimator (or its pipeline final step) exposes predict_proba."""
    final = estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
    return hasattr(final, "predict_proba")


def cross_validate_model(data: TargetData, factory: ModelFactory) -> CVResult:
    """Run stratified k-fold CV for one model on one target.

    Args:
        data: Prepared target dataset.
        factory: Callable returning a fresh estimator.

    Returns:
        A :class:`CVResult` with mean/std of accuracy, macro-F1 and top-2.
    """
    name = next(n for n, f in MODELS.items() if f is factory)
    n_classes = data.y.nunique()
    compute_top2 = n_classes >= 3

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accs: list[float] = []
    f1s: list[float] = []
    top2s: list[float] = []

    x_values = data.X.to_numpy()
    y_values = data.y.to_numpy()

    for train_idx, test_idx in skf.split(x_values, y_values):
        model = factory()
        model.fit(x_values[train_idx], y_values[train_idx])
        pred = model.predict(x_values[test_idx])
        accs.append(accuracy_score(y_values[test_idx], pred))
        f1s.append(f1_score(y_values[test_idx], pred, average="macro"))
        if compute_top2 and _supports_proba(model):
            proba = model.predict_proba(x_values[test_idx])
            top2s.append(
                top_k_accuracy_score(y_values[test_idx], proba, k=2, labels=model.classes_)
            )

    return CVResult(
        model=name,
        acc_mean=float(np.mean(accs)),
        acc_std=float(np.std(accs)),
        f1_mean=float(np.mean(f1s)),
        f1_std=float(np.std(f1s)),
        top2_mean=float(np.mean(top2s)) if top2s else None,
        top2_std=float(np.std(top2s)) if top2s else None,
    )


def compare_target(data: TargetData) -> list[CVResult]:
    """Cross-validate every candidate model on one target, sorted by accuracy.

    Args:
        data: Prepared target dataset.

    Returns:
        Results sorted from best to worst mean accuracy.
    """
    print(f"\n{'=' * 72}\n{data.title}  —  n={len(data.y)}, classes={data.y.nunique()}")
    print(f"{'model':16s} {'accuracy':>16s} {'macro-F1':>16s} {'top-2':>16s}")
    results = [cross_validate_model(data, factory) for factory in MODELS.values()]
    results.sort(key=lambda r: r.acc_mean, reverse=True)
    for r in results:
        top2 = f"{r.top2_mean:.3f}±{r.top2_std:.3f}" if r.top2_mean is not None else "—"
        print(
            f"{r.model:16s} {r.acc_mean:.3f}±{r.acc_std:.3f}   "
            f"{r.f1_mean:.3f}±{r.f1_std:.3f}   {top2:>16s}"
        )
    return results


def main() -> None:
    """Entry point for ``make compare``: compare all models on all targets."""
    targets = load_targets()
    for data in targets.values():
        compare_target(data)


if __name__ == "__main__":
    main()
