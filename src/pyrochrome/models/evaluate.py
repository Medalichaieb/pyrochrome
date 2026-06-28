"""Generate the reproducible evaluation report (``make eval``).

Writes ``reports/REPORT.md`` plus figures under ``reports/figures/`` covering,
for the selected models on a held-out 20% test split:

    - per-class precision / recall / F1 and accuracy / macro-F1 / top-2,
    - confusion matrices (row-normalised),
    - probability calibration (reliability diagram + expected calibration error),
    - permutation feature importances (which oxides drive each target),
    - colour Lab regression error (ΔE / per-channel MAE / R²) with diagnostics.

Everything uses the same preprocessing and split as training, so the numbers
match ``make train`` / ``make color``. Run: ``make eval``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: render to files, no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from pyrochrome.models.color_regression import MODELS as REGRESSORS  # noqa: E402
from pyrochrome.models.datasets import (
    RegressionData,
    TargetData,
    load_colour_regression,
    load_targets,
)  # noqa: E402
from pyrochrome.models.selected import SELECTED_MODEL_NAME, build_selected_model  # noqa: E402
from pyrochrome.pipeline.color import delta_e_cie76  # noqa: E402

RANDOM_STATE = 42
N_BINS = 10
TOP_FEATURES = 12
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

CLAY = "#bd5b33"
INK = "#1c1a17"


@dataclass
class ClassEval:
    """Evaluation artifacts for one classification target."""

    name: str
    title: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    top2: float | None
    per_class: dict[str, dict[str, float]]
    ece: float
    figures: dict[str, Path] = field(default_factory=dict)


def _style() -> None:
    """Apply a light, restrained figure style consistent with the site."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cfc8bd",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "font.size": 9,
            "axes.titlesize": 11,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
        }
    )


def _expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray
) -> tuple[float, list[float], list[float], list[float]]:
    """Top-label ECE plus per-bin (centres, accuracy, mean confidence).

    Args:
        confidences: Max predicted probability per sample.
        correct: Whether the top prediction was right (bool array).

    Returns:
        ``(ece, bin_centres, bin_accuracy, bin_confidence)``; empty bins skipped.
    """
    edges = np.linspace(0, 1, N_BINS + 1)
    ece = 0.0
    centres: list[float] = []
    accs: list[float] = []
    confs: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        acc = float(correct[mask].mean())
        conf = float(confidences[mask].mean())
        ece += abs(acc - conf) * (mask.mean())
        centres.append((lo + hi) / 2)
        accs.append(acc)
        confs.append(conf)
    return ece, centres, accs, confs


def _plot_confusion(name: str, title: str, labels: list[str], matrix: np.ndarray) -> Path:
    """Save a row-normalised confusion-matrix heatmap; return its path."""
    norm = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(0.6 * len(labels) + 2.2, 0.6 * len(labels) + 2))
    im = ax.imshow(norm, cmap="Oranges", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title} — confusion (row-normalised)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if norm[i, j] > 0.5 else INK,
                fontsize=7,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = FIGURES_DIR / f"confusion_{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_calibration(name: str, title: str, centres: list[float], accs: list[float]) -> Path:
    """Save a reliability diagram; return its path."""
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], "--", color="#b9b1a4", linewidth=1, label="Perfect")
    ax.plot(centres, accs, "o-", color=CLAY, linewidth=1.6, markersize=4, label="Model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence (top class)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"{title} — calibration")
    ax.legend(frameon=False, fontsize=8)
    path = FIGURES_DIR / f"calibration_{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_importance(name: str, title: str, names: list[str], values: np.ndarray) -> Path:
    """Save a horizontal bar chart of the top permutation importances."""
    order = np.argsort(values)[::-1][:TOP_FEATURES][::-1]
    fig, ax = plt.subplots(figsize=(4.6, 0.3 * len(order) + 1.2))
    ax.barh(range(len(order)), values[order], color=CLAY)
    ax.set_yticks(range(len(order)), [names[i] for i in order])
    ax.set_xlabel("Permutation importance (Δ accuracy)")
    ax.set_title(f"{title} — feature importance")
    path = FIGURES_DIR / f"importance_{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def evaluate_classifier(data: TargetData) -> ClassEval:
    """Fit the selected model on a held-out split and compute all class metrics."""
    x_train, x_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=0.2, random_state=RANDOM_STATE, stratify=data.y
    )
    model = build_selected_model().fit(x_train, y_train)
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)

    labels = [str(c) for c in model.classes_]
    report: dict[str, Any] = classification_report(y_test, pred, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, pred, labels=model.classes_)
    top2 = (
        float(top_k_accuracy_score(y_test, proba, k=2, labels=model.classes_))
        if data.y.nunique() >= 3
        else None
    )

    confidences = proba.max(axis=1)
    correct = (pred == y_test.to_numpy()).astype(float)
    ece, centres, accs, _ = _expected_calibration_error(confidences, correct)

    importance = permutation_importance(
        model, x_test, y_test, n_repeats=5, random_state=RANDOM_STATE, scoring="accuracy"
    )

    result = ClassEval(
        name=data.name,
        title=data.title,
        accuracy=float(accuracy_score(y_test, pred)),
        macro_f1=float(f1_score(y_test, pred, average="macro")),
        weighted_f1=float(f1_score(y_test, pred, average="weighted")),
        top2=top2,
        per_class={k: v for k, v in report.items() if k in labels},
        ece=ece,
    )
    result.figures["confusion"] = _plot_confusion(data.name, data.title, labels, matrix)
    result.figures["calibration"] = _plot_calibration(data.name, data.title, centres, accs)
    result.figures["importance"] = _plot_importance(
        data.name, data.title, data.feature_names, importance.importances_mean
    )
    return result


def evaluate_regression(data: RegressionData) -> dict[str, Any]:
    """Fit the colour Lab regressor on a held-out split; metrics + diagnostics."""
    x_train, x_test, y_train, y_test = train_test_split(
        data.X, data.Y, test_size=0.2, random_state=RANDOM_STATE
    )
    model = REGRESSORS["extra_trees"]().fit(x_train, y_train)
    pred = np.asarray(model.predict(x_test))
    delta_e = delta_e_cie76(pred, y_test)

    # ΔE distribution.
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.hist(delta_e, bins=40, color=CLAY, alpha=0.85)
    ax.axvline(float(np.median(delta_e)), color=INK, linestyle="--", linewidth=1, label="median")
    ax.set_xlabel("ΔE (CIE76) on held-out recipes")
    ax.set_ylabel("count")
    ax.set_title("Colour regression — ΔE distribution")
    ax.legend(frameon=False, fontsize=8)
    hist_path = FIGURES_DIR / "regression_delta_e.png"
    fig.savefig(hist_path)
    plt.close(fig)

    # Predicted vs true per channel.
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    for ax, idx, name in zip(axes, range(3), ["L*", "a*", "b*"], strict=True):
        ax.scatter(y_test[:, idx], pred[:, idx], s=4, alpha=0.25, color=CLAY)
        lims = [float(y_test[:, idx].min()), float(y_test[:, idx].max())]
        ax.plot(lims, lims, "--", color="#b9b1a4", linewidth=1)
        ax.set_xlabel(f"true {name}")
        ax.set_ylabel(f"predicted {name}")
    fig.suptitle("Colour regression — predicted vs. true")
    scatter_path = FIGURES_DIR / "regression_scatter.png"
    fig.savefig(scatter_path)
    plt.close(fig)

    return {
        "n": int(len(y_test)),
        "delta_e_mean": float(np.mean(delta_e)),
        "delta_e_median": float(np.median(delta_e)),
        "mae": [float(mean_absolute_error(y_test[:, c], pred[:, c])) for c in range(3)],
        "r2": float(r2_score(y_test, pred, multioutput="uniform_average")),
        "figures": {"delta_e": hist_path, "scatter": scatter_path},
    }


def _rel(path: Path) -> str:
    """Path relative to the reports dir, for embedding in REPORT.md."""
    return str(path.relative_to(REPORTS_DIR))


def _write_report(classes: list[ClassEval], regression: dict[str, Any], timestamp: str) -> Path:
    """Assemble REPORT.md from the evaluation artifacts."""
    lines: list[str] = [
        "# Pyrochrome — evaluation report",
        "",
        f"_Generated {timestamp} · model `{SELECTED_MODEL_NAME}` · held-out 20% test "
        "split (seed 42)._ Regenerate with `make eval`.",
        "",
    ]

    for ev in classes:
        lines += [
            f"## {ev.title}",
            "",
            f"- Accuracy **{ev.accuracy:.3f}** · macro-F1 **{ev.macro_f1:.3f}** · "
            f"weighted-F1 {ev.weighted_f1:.3f}"
            + (f" · top-2 **{ev.top2:.3f}**" if ev.top2 is not None else ""),
            f"- Expected calibration error (top label): **{ev.ece:.3f}** "
            "(lower = the confidence % can be trusted more).",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "|---|---|---|---|---|",
        ]
        for cls, m in sorted(ev.per_class.items(), key=lambda kv: -kv[1]["support"]):
            lines.append(
                f"| {cls} | {m['precision']:.2f} | {m['recall']:.2f} | "
                f"{m['f1-score']:.2f} | {int(m['support'])} |"
            )
        lines += [
            "",
            f"![confusion]({_rel(ev.figures['confusion'])})",
            "",
            f"![calibration]({_rel(ev.figures['calibration'])}) "
            f"![importance]({_rel(ev.figures['importance'])})",
            "",
        ]

    mae = regression["mae"]
    lines += [
        "## Colour — Lab regression",
        "",
        f"- n={regression['n']} · mean ΔE **{regression['delta_e_mean']:.2f}** · "
        f"median ΔE {regression['delta_e_median']:.2f} · R² {regression['r2']:.3f}",
        f"- Per-channel MAE — L* {mae[0]:.2f} · a* {mae[1]:.2f} · b* {mae[2]:.2f}",
        "- The large residual ΔE is the photo-label-noise ceiling; see the Docs page.",
        "",
        f"![delta-e]({_rel(regression['figures']['delta_e'])})",
        "",
        f"![scatter]({_rel(regression['figures']['scatter'])})",
        "",
    ]

    path = REPORTS_DIR / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    """Entry point for ``make eval``: build figures and write reports/REPORT.md."""
    from datetime import UTC, datetime

    _style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    targets = load_targets()
    classes = [evaluate_classifier(targets[name]) for name in ("surface", "transparency", "colour")]
    regression = evaluate_regression(load_colour_regression())
    path = _write_report(classes, regression, timestamp)

    print(f"Wrote {path} and {len(list(FIGURES_DIR.glob('*.png')))} figures to {FIGURES_DIR}/")
    for ev in classes:
        extra = f" top-2={ev.top2:.3f}" if ev.top2 is not None else ""
        print(f"  {ev.name:13s} acc={ev.accuracy:.3f} F1={ev.macro_f1:.3f} ECE={ev.ece:.3f}{extra}")
    print(f"  colour_lab    ΔE={regression['delta_e_mean']:.2f} R²={regression['r2']:.3f}")


if __name__ == "__main__":
    main()
