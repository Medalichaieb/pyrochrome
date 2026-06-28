"""The selected model and the rationale for choosing it.

Single source of truth for the production model, so the comparison harness
(:mod:`pyrochrome.models.compare`), the trainer (:mod:`pyrochrome.models.train`)
and the API all use the exact same estimator.

Decision (2026-06-27, 5-fold stratified CV on UMF + cone features):
    HistGradientBoostingClassifier.

Why, given the tree ensembles were statistically tied (differences within the
±0.008–0.011 inter-fold std):
    - Best / tied-best on the colour target (the harder, product-relevant one):
      acc 0.664, top-2 0.794, macro-F1 0.511.
    - Best macro-F1 on surface (0.658), accuracy within noise of RandomForest.
    - Matches LightGBM / XGBoost performance **without** a heavy native
      dependency — stays pure-sklearn, fully reproducible from one lockfile.
    - Log-loss gradient boosting is typically better *calibrated* than a
      RandomForest, which matters because we surface a confidence index.

See ``results.md`` and ``MODEL_CARD.md`` for the full comparison tables.

Docs: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier

SELECTED_MODEL_NAME = "hist_gb"

# Hyperparameters chosen during the comparison run. Modest depth + L2 + a low
# learning rate, with early stopping so training halts once the held-out loss
# plateaus (far below max_iter) — faster and avoids overfitting 600 iterations.
HIST_GB_PARAMS = {
    "max_iter": 600,
    "learning_rate": 0.06,
    "max_leaf_nodes": 63,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 15,
    "random_state": 42,
}


def build_selected_model() -> HistGradientBoostingClassifier:
    """Return a fresh, unfitted instance of the selected model.

    Returns:
        A :class:`~sklearn.ensemble.HistGradientBoostingClassifier` configured
        with :data:`HIST_GB_PARAMS`.
    """
    return HistGradientBoostingClassifier(**HIST_GB_PARAMS)
