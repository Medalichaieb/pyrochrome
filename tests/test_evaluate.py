import numpy as np

from pyrochrome.models.evaluate import _expected_calibration_error


def test_ece_zero_when_perfectly_calibrated() -> None:
    conf = np.full(100, 1.0)
    correct = np.ones(100)
    ece, centres, accs, confs = _expected_calibration_error(conf, correct)
    assert ece == 0.0
    assert len(centres) == len(accs) == len(confs) == 1


def test_ece_reflects_overconfidence() -> None:
    # All predictions claim 0.9 confidence but only half are right -> ECE 0.4.
    conf = np.full(100, 0.9)
    correct = np.array([1.0] * 50 + [0.0] * 50)
    ece, _, accs, confs = _expected_calibration_error(conf, correct)
    assert np.isclose(ece, 0.4, atol=1e-9)
    assert np.isclose(accs[0], 0.5)
    assert np.isclose(confs[0], 0.9)
