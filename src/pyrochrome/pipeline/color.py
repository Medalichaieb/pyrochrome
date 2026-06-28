"""Colour-space conversions for the Lab-regression target (lever #2).

We move colour prediction out of the noisy 10-family classification scheme and
into **CIELAB**, a perceptually-uniform space where distances approximate human
colour difference (ΔE). The regressor predicts (L*, a*, b*) and we report error
as ΔE, which is the product-meaningful metric.

Conversion path: sRGB (0–255) → linear RGB → CIE XYZ (D65) → CIE L*a*b*.
Reference white: D65 (Xn, Yn, Zn) = (0.95047, 1.0, 1.08883).

Inputs : RGB arrays with channels in 0–255.
Outputs: Lab arrays (L* in 0–100, a*/b* roughly −128..127) and ΔE distances.
"""

from __future__ import annotations

import numpy as np

# sRGB (linear) → XYZ matrix, D65 reference white.
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])
_EPS = 216 / 24389  # CIE standard 0.008856
_KAPPA = 24389 / 27  # CIE standard 903.3


def _srgb_to_linear(channel: np.ndarray) -> np.ndarray:
    """Undo the sRGB gamma companding for channels in [0, 1]."""
    return np.where(channel > 0.04045, ((channel + 0.055) / 1.055) ** 2.4, channel / 12.92)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB colours (0–255) to CIELAB (D65).

    Args:
        rgb: Array of shape ``(..., 3)`` with channels in 0–255.

    Returns:
        Array of the same leading shape with the last axis = (L*, a*, b*).
    """
    rgb = np.asarray(rgb, dtype=float)
    linear = _srgb_to_linear(rgb / 255.0)
    xyz = linear @ _RGB_TO_XYZ.T
    ratio = xyz / _WHITE_D65
    f = np.where(ratio > _EPS, np.cbrt(ratio), (_KAPPA * ratio + 16) / 116)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    lab = np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)
    return lab


def delta_e_cie76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Compute the CIE76 colour difference (Euclidean distance in Lab).

    CIE76 is the simple, standard ΔE used for v1 reporting. It slightly
    overstates differences for saturated colours vs CIEDE2000, but is monotonic
    and sufficient for tracking regression error.

    Args:
        lab1: Array of shape ``(..., 3)``.
        lab2: Array broadcastable to ``lab1``.

    Returns:
        ΔE distances with the last (colour) axis reduced.
    """
    diff = np.asarray(lab1, dtype=float) - np.asarray(lab2, dtype=float)
    return np.sqrt(np.sum(diff**2, axis=-1))
