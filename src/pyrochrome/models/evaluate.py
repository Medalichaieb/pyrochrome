"""Detailed evaluation & report generation (skeleton).

Produces the versioned, reproducible evaluation report described in the brief
(§11.6): per-target metrics, confusion matrices, feature importances,
probability calibration and error analysis, written to ``reports/`` and
summarised in ``results.md``.

Wired to ``make eval``. Currently a stub that runs the baseline and prints its
metrics; the figures/report generation is TODO.
"""

from __future__ import annotations

from pyrochrome.models.baseline import run


def main() -> None:
    """Entry point for ``make eval`` (stub — extends baseline metrics).

    Todo:
        - confusion matrices (surface, colour) → reports/figures/
        - permutation importance / SHAP per target
        - probability calibration (reliability of the confidence index)
        - error analysis (atmosphere-driven failures, label noise examples)
        - write reports/REPORT-<date>.md and append to results.md
    """
    results = run()
    print("\n[evaluate] Detailed report generation is not implemented yet (TODO).")
    print(f"[evaluate] Trained {len(results)} target(s).")


if __name__ == "__main__":
    main()
