"""Baseline render predictor: surface & colour family from UMF chemistry + cone.

This is a clean, importable refactor of ``prototypes/glaze_baseline.py``. It
trains and compares a naïve baseline, a RandomForest and a HistGradientBoosting
classifier on two targets, and reports accuracy / F1-macro / top-2.

Pipeline (per target):
    clean recipes → build features (UMF + cone) → impute medians → stratified
    80/20 split (seed=42) → fit → score on the held-out test set.

Entry point ``main`` is wired to ``make train``. It locates the latest Glazy CSV
via :mod:`pyrochrome.pipeline.download`.

LIMITATIONS (see the brief / MODEL_CARD): no atmosphere yet (lever #1); colour
labels derived from noisy photo RGB (lever #2).
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split

from pyrochrome.models.datasets import TargetData, load_targets

RANDOM_STATE = 42


@dataclass
class TargetResult:
    """Metrics for one trained target.

    Attributes:
        name: Human-readable target name.
        n: Number of labelled samples used.
        n_classes: Number of distinct classes.
        baseline_acc: Accuracy of the majority-class baseline.
        rf_acc: RandomForest accuracy.
        rf_f1_macro: RandomForest macro-averaged F1.
        gb_acc: HistGradientBoosting accuracy.
        gb_f1_macro: HistGradientBoosting macro-averaged F1.
        rf_top2: RandomForest top-2 accuracy (None if not computed).
    """

    name: str
    n: int
    n_classes: int
    baseline_acc: float
    rf_acc: float
    rf_f1_macro: float
    gb_acc: float
    gb_f1_macro: float
    rf_top2: float | None = None


def train_target(
    data: TargetData,
    *,
    compute_top2: bool = False,
) -> tuple[RandomForestClassifier, TargetResult]:
    """Train and score the models for a single target.

    Uses the shared :func:`pyrochrome.models.datasets.load_targets` preprocessing
    (same cleaning, features and imputation as every other experiment).

    Args:
        data: Prepared target dataset (features already imputed).
        compute_top2: Whether to also report RandomForest top-2 accuracy.

    Returns:
        The fitted RandomForest and a :class:`TargetResult` of metrics.
    """
    features, target = data.X, data.y
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE, stratify=target
    )

    dummy = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    baseline_acc = accuracy_score(y_test, dummy.predict(x_test))

    rf = RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    ).fit(x_train, y_train)
    rf_pred = rf.predict(x_test)

    gb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, random_state=RANDOM_STATE
    ).fit(x_train, y_train)
    gb_pred = gb.predict(x_test)

    top2: float | None = None
    if compute_top2:
        proba = rf.predict_proba(x_test)
        top2 = float(top_k_accuracy_score(y_test, proba, k=2, labels=rf.classes_))

    result = TargetResult(
        name=data.title,
        n=int(len(features)),
        n_classes=int(target.nunique()),
        baseline_acc=float(baseline_acc),
        rf_acc=float(accuracy_score(y_test, rf_pred)),
        rf_f1_macro=float(f1_score(y_test, rf_pred, average="macro")),
        gb_acc=float(accuracy_score(y_test, gb_pred)),
        gb_f1_macro=float(f1_score(y_test, gb_pred, average="macro")),
        rf_top2=top2,
    )
    return rf, result


def _print_result(result: TargetResult) -> None:
    """Pretty-print one target's metrics to stdout."""
    print(f"\n{'=' * 62}\n{result.name}\n  n={result.n}   classes={result.n_classes}")
    print(f"  {'Naïve baseline (majority class)':42s} acc={result.baseline_acc:.3f}")
    print(f"  {'Random Forest':42s} acc={result.rf_acc:.3f}   F1-macro={result.rf_f1_macro:.3f}")
    print(
        f"  {'Gradient Boosting':42s} acc={result.gb_acc:.3f}   F1-macro={result.gb_f1_macro:.3f}"
    )
    if result.rf_top2 is not None:
        print(f"  -> Random Forest Top-2 : {result.rf_top2:.3f}")


def run(csv_path: str | None = None) -> list[TargetResult]:
    """Train both targets and return their metrics.

    Args:
        csv_path: Optional explicit path to the Glazy CSV. If ``None``, the
            latest CSV under ``data/raw/glazy-data/`` is used.

    Returns:
        A list of :class:`TargetResult`, one per target.
    """
    targets = load_targets(csv_path)
    feat_count = len(next(iter(targets.values())).feature_names)
    print(f"Features (UMF + ratios + cone): {feat_count}")

    results: list[TargetResult] = []
    # Top-2 is the product-relevant metric for the multi-class colour target.
    for name, data in targets.items():
        _, result = train_target(data, compute_top2=(name == "colour"))
        _print_result(result)
        results.append(result)
    return results


def main() -> None:
    """Entry point for ``make train``."""
    run()


if __name__ == "__main__":
    main()
