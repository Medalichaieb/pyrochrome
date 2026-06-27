"""Export a compact model to JSON for the in-browser predictor (skeleton).

The ``prototypes/glaze_predictor.html`` demo runs a small MLP entirely in the
browser, loaded from a JSON blob holding the feature list, the input scaler
(mean/std) and the layer weights. This module will train that compact MLP and
serialise it to ``web/src/model.json``.

Wired to ``make export``. TODO: train an MLPClassifier/MLPRegressor on the same
features and dump features + scaler + weights as JSON.
"""

from __future__ import annotations


def main() -> None:
    """Entry point for ``make export`` (stub).

    Todo:
        - fit sklearn ``MLPClassifier`` / ``MLPRegressor`` on the standard
          feature set (StandardScaler + MLP)
        - serialise {features, scaler: {mean, std}, weights} to web/src/model.json
        - mirror the JSON schema already consumed by glaze_predictor.html
    """
    raise NotImplementedError(
        "Compact-model export is not implemented yet. See prototypes/glaze_predictor.html "
        "for the target JSON schema."
    )


if __name__ == "__main__":
    main()
