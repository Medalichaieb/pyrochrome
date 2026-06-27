import math

from pyrochrome.pipeline.cones import CONE_ORDER, cone_to_ordinal


def test_cone_ordering_is_monotonic_and_complete() -> None:
    assert cone_to_ordinal("022") == 0
    assert cone_to_ordinal("14") == len(CONE_ORDER) - 1
    # Hotter cone => strictly larger ordinal.
    assert cone_to_ordinal("6") > cone_to_ordinal("04")
    assert cone_to_ordinal("04") > cone_to_ordinal("022")


def test_cone_half_entity_is_normalised() -> None:
    assert cone_to_ordinal("05&#189;") == cone_to_ordinal("05.5")
    assert cone_to_ordinal(" 6 ") == cone_to_ordinal("6")


def test_unknown_or_missing_cone_is_nan() -> None:
    assert math.isnan(cone_to_ordinal(None))
    assert math.isnan(cone_to_ordinal("not-a-cone"))
    assert math.isnan(cone_to_ordinal(float("nan")))
