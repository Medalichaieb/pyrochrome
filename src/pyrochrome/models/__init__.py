"""Models: dataset prep, model selection, training, evaluation and export.

- ``datasets`` — shared (X, y) preparation per target (single source of truth).
- ``selected`` — the selected production model + the rationale for the choice.
- ``compare``  — cross-validated comparison of candidate models (model selection).
- ``train``    — fit & persist the selected model per target (``make train``).
- ``baseline`` — reference RandomForest / HistGradientBoosting reproduction.
- ``evaluate`` — detailed metrics & report generation (TODO).
- ``export``   — compact model → JSON for the in-browser predictor (TODO).
"""
