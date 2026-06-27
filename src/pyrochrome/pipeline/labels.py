"""Derive prediction targets (labels) from Glazy columns.

Targets:
    - surface       : Glossy / Matte / Satin   (from ``surface_type``)
    - colour family : perceptual bucket          (from ``rgb_r/g/b`` via HSV)

The colour families are **noisy**: the RGB comes from photos taken under
non-standardised lighting. This bucketing is the PoC label scheme; priority
lever #2 is to replace it with cleaned Lab-space colour (GlazyBench approach).
"""

from __future__ import annotations

import colorsys
from typing import Any

import pandas as pd


def surface_family(value: Any) -> float | str:
    """Collapse a raw ``surface_type`` string to Glossy / Matte / Satin.

    Args:
        value: Raw surface label (e.g. ``"Glossy bright"``), or missing.

    Returns:
        One of ``"Glossy"``, ``"Matte"``, ``"Satin"``, or ``float('nan')`` if
        the value is missing or matches none of them.
    """
    if pd.isna(value):
        return float("nan")
    text = str(value)
    for family in ("Glossy", "Matte", "Satin"):
        if text.startswith(family):
            return family
    return float("nan")


def color_family(r: Any, g: Any, b: Any) -> float | str:
    """Bucket an RGB triple into a perceptual colour family via HSV.

    The thresholds are hand-tuned for ceramic glaze colours (e.g. dark + warm
    hues collapse to ``"Brun"`` / brown). Labels are kept in French to match the
    public-facing site.

    Args:
        r: Red channel, 0-255 (or missing).
        g: Green channel, 0-255.
        b: Blue channel, 0-255.

    Returns:
        A colour-family string, or ``float('nan')`` if RGB is missing.
    """
    if pd.isna(r) or pd.isna(g) or pd.isna(b):
        return float("nan")
    h, s, v = colorsys.rgb_to_hsv(float(r) / 255, float(g) / 255, float(b) / 255)
    hue = h * 360
    if v < 0.16:
        return "Noir"
    if s < 0.13:
        return "Blanc" if v > 0.78 else "Gris"
    if hue < 22 or hue >= 330:
        base = "Rouge"
    elif hue < 45:
        base = "Orange"
    elif hue < 68:
        base = "Jaune"
    elif hue < 160:
        base = "Vert"
    elif hue < 200:
        base = "Turquoise"
    elif hue < 255:
        base = "Bleu"
    else:
        base = "Violet"
    # Dark, saturated warm hues read as brown rather than their pure hue.
    if base in ("Orange", "Jaune", "Rouge") and v < 0.55 and s > 0.2:
        base = "Brun"
    return base
