"""Models: dataset prep, model selection, training, evaluation and export.

- ``datasets``         — shared (X, y) / (X, Lab) prep (single source of truth).
- ``selected``         — the selected classifier + rationale for the choice.
- ``compare``          — cross-validated comparison of candidate classifiers.
- ``train``            — fit & persist the selected classifier (``make train``).
- ``color_regression`` — Lab colour regression (ΔE), lever #2 (``make color``).
- ``neighbors``        — k-NN nearest real recipes, lever #2 (``make neighbors``).
- ``baseline``         — reference RandomForest / HistGradientBoosting reproduction.
- ``evaluate``         — detailed metrics & report generation (TODO).
- ``export``           — compact model → JSON for the in-browser predictor (TODO).
"""
