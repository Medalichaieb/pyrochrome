from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from pyrochrome.models.datasets import TargetData
from pyrochrome.models.selected import HIST_GB_PARAMS, build_selected_model
from pyrochrome.models.train import save, train_one


def test_build_selected_model_is_configured_hist_gb() -> None:
    model = build_selected_model()
    assert isinstance(model, HistGradientBoostingClassifier)
    assert model.max_iter == HIST_GB_PARAMS["max_iter"]
    assert model.random_state == HIST_GB_PARAMS["random_state"]


def _toy_target() -> TargetData:
    # Two linearly separable classes, enough rows for a stratified 80/20 split.
    rows = []
    for i in range(40):
        rows.append({"f1": float(i), "f2": 0.0, "label": "low" if i < 20 else "high"})
    df = pd.DataFrame(rows)
    return TargetData(
        name="toy",
        title="Toy",
        X=df[["f1", "f2"]],
        y=df["label"].astype("object"),
        feature_names=["f1", "f2"],
    )


def test_train_one_and_save_roundtrip(tmp_path: Path) -> None:
    trained = train_one(_toy_target())
    assert 0.0 <= trained.accuracy <= 1.0
    assert set(trained.classes) == {"low", "high"}
    assert trained.top2 is None  # only 2 classes

    path = save(trained, output_dir=tmp_path)
    assert path.exists()

    bundle = joblib.load(path)
    assert bundle["target"] == "toy"
    assert bundle["feature_names"] == ["f1", "f2"]
    # The reloaded model still predicts.
    preds = bundle["model"].predict(_toy_target().X)
    assert len(preds) == 40
