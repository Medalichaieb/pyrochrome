import numpy as np

from pyrochrome.pipeline.color import delta_e_cie76, srgb_to_lab


def test_known_lab_values() -> None:
    rgb = np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0]], dtype=float)
    lab = srgb_to_lab(rgb)
    # White -> L=100, a=b=0.
    assert np.allclose(lab[0], [100.0, 0.0, 0.0], atol=0.5)
    # Black -> L=0.
    assert np.allclose(lab[1], [0.0, 0.0, 0.0], atol=0.5)
    # Pure sRGB red -> well-known reference (53.24, 80.09, 67.20).
    assert np.allclose(lab[2], [53.24, 80.09, 67.20], atol=0.5)


def test_neutral_gray_is_achromatic() -> None:
    lab = srgb_to_lab(np.array([128, 128, 128], dtype=float))
    # Gray: a and b near zero, L around mid.
    assert abs(lab[1]) < 0.5 and abs(lab[2]) < 0.5
    assert 50 < lab[0] < 56


def test_delta_e_zero_and_symmetry() -> None:
    a = np.array([50.0, 10.0, -5.0])
    b = np.array([53.0, 14.0, -5.0])
    assert delta_e_cie76(a, a) == 0.0
    assert np.isclose(delta_e_cie76(a, b), delta_e_cie76(b, a))
    assert np.isclose(delta_e_cie76(a, b), 5.0)  # 3-4-5 triangle
