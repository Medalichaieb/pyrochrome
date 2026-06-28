import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from pyrochrome.models.inference import Predictor, build_feature_vector
from pyrochrome.pipeline.cones import cone_to_ordinal


def test_build_feature_vector_maps_cone_atmosphere_and_oxides() -> None:
    feature_names = [
        "SiO2_umf",
        "CuO_umf",
        "cone_num",
        "atm_oxidation",
        "atm_reduction",
        "atm_known",
    ]
    vec = build_feature_vector(
        feature_names,
        chemistry_umf={"SiO2": 3.2, "CuO_umf": 0.05},  # bare and full-name keys both work
        cone="6",
        atmosphere="reduction",
    )
    assert vec.shape == (1, 6)
    row = vec[0]
    assert row[0] == 3.2  # SiO2 via bare key
    assert row[1] == 0.05  # CuO via _umf key
    assert row[2] == cone_to_ordinal("6")  # cone mapped to ordinal
    assert row[3] == 0.0 and row[4] == 1.0  # multi-hot: reduction set, oxidation not
    assert row[5] == 1.0  # atm_known


def test_build_feature_vector_defaults_missing_to_zero() -> None:
    vec = build_feature_vector(
        ["Al2O3_umf", "cone_num"], chemistry_umf={}, cone="bad", atmosphere="oxidation"
    )
    assert vec[0, 0] == 0.0  # missing oxide -> 0
    assert vec[0, 1] == 0.0  # unknown cone -> 0 (not NaN)


def _toy_classifier_bundle() -> dict[str, object]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 2))
    y = np.where(x[:, 0] + x[:, 1] > 0, "Glossy", "Matte")
    model = HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(x, y)
    return {
        "model": model,
        "feature_names": ["SiO2_umf", "cone_num"],
        "classes": list(model.classes_),
    }


def test_predictor_predict_with_single_classifier() -> None:
    predictor = Predictor(
        classifiers={"surface": _toy_classifier_bundle()}, colour_lab=None, neighbors=None
    )
    out = predictor.predict({"SiO2": 1.0}, cone="6", atmosphere="oxidation")
    assert set(out) == {"surface"}  # only the loaded model is returned
    assert out["surface"]["label"] in {"Glossy", "Matte"}
    assert 0.0 <= out["surface"]["confidence"] <= 1.0
    assert len(out["surface"]["top2"]) == 2
