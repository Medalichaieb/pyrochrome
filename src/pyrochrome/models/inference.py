"""Load the persisted models and serve predictions for a glaze.

Bundles the artifacts written by ``make train`` / ``make color`` /
``make neighbors`` into a single :class:`Predictor` that the API (and any other
caller) uses. Keeping the feature-vector construction here — not in the API —
makes it unit-testable and keeps the web layer thin.

A request is (chemistry in UMF oxides, Orton cone, firing atmosphere). For each
model we assemble the feature vector in *that model's* stored column order:

    - ``cone_num``        ← the cone mapped to its ordinal index
    - ``atm_<name>``      ← multi-hot from the requested atmosphere (+ ``atm_known``)
    - ``<oxide>_umf`` etc ← from the chemistry dict (accepts ``"SiO2"`` or
                            ``"SiO2_umf"`` keys); missing oxides default to 0

Honesty (brief §7): colour is returned as a family + top-2 + confidence *and* a
Lab value *and* the nearest real recipes — never a single over-confident colour.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from pyrochrome.models.neighbors import RecipeIndex, load_index, query
from pyrochrome.pipeline.atmosphere import ATMOSPHERE_KNOWN_COLUMN
from pyrochrome.pipeline.cones import cone_to_ordinal

MODELS_DIR = Path("models_out")
CLASSIFIER_TARGETS = ("surface", "transparency", "colour")


def build_feature_vector(
    feature_names: list[str],
    chemistry_umf: dict[str, float],
    cone: str,
    atmosphere: str,
) -> np.ndarray:
    """Assemble a 1-row feature vector matching ``feature_names``.

    Args:
        feature_names: The model's feature columns, in order.
        chemistry_umf: UMF oxide amounts; keys may be ``"SiO2"`` or ``"SiO2_umf"``.
        cone: Orton cone label (e.g. ``"6"``).
        atmosphere: Canonical atmosphere name (e.g. ``"oxidation"``).

    Returns:
        A float array of shape ``(1, len(feature_names))``.
    """
    cone_ordinal = cone_to_ordinal(cone)
    row = []
    for name in feature_names:
        if name == "cone_num":
            row.append(0.0 if np.isnan(cone_ordinal) else cone_ordinal)
        elif name == ATMOSPHERE_KNOWN_COLUMN:
            row.append(1.0)
        elif name.startswith("atm_"):
            row.append(1.0 if name == f"atm_{atmosphere}" else 0.0)
        else:
            # Oxide / aggregate UMF column: accept the full name or the bare oxide.
            bare = name[:-4] if name.endswith("_umf") else name
            row.append(float(chemistry_umf.get(name, chemistry_umf.get(bare, 0.0))))
    return np.array(row, dtype=float).reshape(1, -1)


@dataclass
class Predictor:
    """Holds the loaded artifacts and serves predictions."""

    classifiers: dict[str, dict[str, Any]]
    colour_lab: dict[str, Any] | None
    neighbors: RecipeIndex | None

    @classmethod
    def load(cls, models_dir: str | Path = MODELS_DIR) -> Predictor:
        """Load all available artifacts from ``models_dir``.

        Missing artifacts are tolerated (the corresponding output is omitted), so
        the API can still answer partially before every model is trained.

        Args:
            models_dir: Directory holding the ``*.joblib`` artifacts.

        Returns:
            A ready :class:`Predictor`.
        """
        base = Path(models_dir)
        classifiers: dict[str, dict[str, Any]] = {}
        for target in CLASSIFIER_TARGETS:
            path = base / f"{target}.joblib"
            if path.exists():
                classifiers[target] = joblib.load(path)
        colour_lab = (
            joblib.load(base / "colour_lab.joblib")
            if (base / "colour_lab.joblib").exists()
            else None
        )
        neighbors = (
            load_index(base / "neighbors.joblib") if (base / "neighbors.joblib").exists() else None
        )
        return cls(classifiers=classifiers, colour_lab=colour_lab, neighbors=neighbors)

    def _classify(
        self, target: str, chemistry: dict[str, float], cone: str, atm: str
    ) -> dict[str, Any]:
        """Predict one classification target with its confidence and top-2."""
        bundle = self.classifiers[target]
        model = bundle["model"]
        x = build_feature_vector(bundle["feature_names"], chemistry, cone, atm)
        proba = model.predict_proba(x)[0]
        order = np.argsort(proba)[::-1]
        classes = model.classes_
        # top-2 as {label, p} so the UI can show each class's probability.
        top2 = [{"label": str(classes[i]), "p": float(proba[i])} for i in order[:2]]
        return {
            "label": str(classes[order[0]]),
            "top2": top2,
            "confidence": float(proba[order[0]]),
        }

    def predict(
        self, chemistry_umf: dict[str, float], cone: str, atmosphere: str, n_neighbors: int = 6
    ) -> dict[str, Any]:
        """Predict surface, transparency, colour (+ Lab) and nearest recipes.

        Args:
            chemistry_umf: UMF oxide amounts.
            cone: Orton cone label.
            atmosphere: Canonical atmosphere name.
            n_neighbors: How many nearest real recipes to return.

        Returns:
            A dict with the per-target predictions, the Lab colour and neighbours
            (each key present only if its model is loaded).
        """
        out: dict[str, Any] = {}
        for target in ("surface", "transparency"):
            if target in self.classifiers:
                out[target] = self._classify(target, chemistry_umf, cone, atmosphere)

        if "colour" in self.classifiers:
            colour = self._classify("colour", chemistry_umf, cone, atmosphere)
            if self.colour_lab is not None:
                x = build_feature_vector(
                    self.colour_lab["feature_names"], chemistry_umf, cone, atmosphere
                )
                colour["lab"] = [float(v) for v in self.colour_lab["model"].predict(x)[0]]
            out["colour"] = colour

        if self.neighbors is not None:
            x = build_feature_vector(self.neighbors.feature_names, chemistry_umf, cone, atmosphere)
            neighbours = query(self.neighbors, x[0], k=n_neighbors)
            out["neighbours"] = neighbours.to_dict(orient="records")

        return out
