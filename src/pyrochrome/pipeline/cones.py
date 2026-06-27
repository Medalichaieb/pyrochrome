"""Orton cone → ordinal index mapping.

Orton pyrometric cones measure heat-work, not temperature, and their numbering
is **not linear** (it runs ...022, 021, ... 01, 1, 2, ... 14, coldest to
hottest). For modelling we map each cone to its position in this ordered list so
that "hotter" is monotonically increasing — an ordinal feature the models can
use directly.

Inputs : a raw cone label as found in Glazy's ``from_orton_cone`` /
         ``to_orton_cone`` columns (string, may contain the ``&#189;`` HTML
         entity for ½).
Outputs: an integer ordinal index, or ``nan`` if the cone is unknown/missing.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

# Orton cones from coldest to hottest. Index in this list == ordinal value.
CONE_ORDER: list[str] = [
    "022",
    "021",
    "020",
    "019",
    "018",
    "017",
    "016",
    "015",
    "014",
    "013",
    "012",
    "011",
    "010",
    "09",
    "08",
    "07",
    "06",
    "05",
    "05.5",
    "04",
    "03",
    "02",
    "01",
    "1",
    "2",
    "3",
    "4",
    "5",
    "5.5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
]

CONE_IDX: dict[str, int] = {cone: i for i, cone in enumerate(CONE_ORDER)}


def cone_to_ordinal(value: Any) -> float:
    """Convert a raw Orton cone label to its ordinal index.

    Args:
        value: Raw cone label (e.g. ``"6"``, ``"05.5"``, ``"04&#189;"``), or a
            missing value. Typed ``Any`` as it comes straight from a pandas cell.

    Returns:
        The ordinal index as a float (so it composes with NaNs in pandas), or
        ``float('nan')`` if the cone is missing or unrecognised.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return float("nan")
    normalised = str(value).replace("&#189;", ".5").replace(" ", "")
    return float(CONE_IDX.get(normalised, float("nan")))
