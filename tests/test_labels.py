import math

from pyrochrome.pipeline.labels import color_family, surface_family


def _is_nan(value: float | str) -> bool:
    """True if ``value`` is the NaN sentinel returned for missing labels."""
    return isinstance(value, float) and math.isnan(value)


def test_surface_family_prefix_match() -> None:
    assert surface_family("Glossy bright") == "Glossy"
    assert surface_family("Matte") == "Matte"
    assert surface_family("Satin smooth") == "Satin"


def test_surface_family_unknown_is_nan() -> None:
    assert _is_nan(surface_family("Crystalline"))
    assert _is_nan(surface_family(float("nan")))


def test_color_family_basic_buckets() -> None:
    assert color_family(0, 0, 0) == "Noir"
    assert color_family(255, 255, 255) == "Blanc"
    assert color_family(128, 128, 128) == "Gris"
    assert color_family(20, 30, 220) == "Bleu"
    assert color_family(30, 200, 40) == "Vert"


def test_color_family_dark_warm_is_brown() -> None:
    # Dark, saturated orange/red reads as brown, not its pure hue.
    assert color_family(110, 60, 20) == "Brun"


def test_color_family_missing_is_nan() -> None:
    assert _is_nan(color_family(float("nan"), 0, 0))
