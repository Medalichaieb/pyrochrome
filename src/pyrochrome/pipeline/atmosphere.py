"""Parse the Glazy YAML dump to recover the firing **atmosphere** per recipe.

This is **priority lever #1**: the atmosphere (Oxidation / Reduction) is present
in ``glazy_YYYYMMDD.yaml.gz`` but **absent from the flat CSV**. Recovering it and
joining by ``id`` is expected to give the largest accuracy gain, because many
colorants (notably copper: green ↔ red) are only decidable given the atmosphere.

Inputs : path to the gzipped YAML dump.
Outputs: a dataframe with columns ``id`` and ``atmosphere`` (normalised to
         ``"oxidation"`` / ``"reduction"`` / ``"neutral"`` / ``None``), ready to
         left-join onto the features table by ``id``.

NOTE: the exact YAML schema must be confirmed against the real dump before this
is trusted (see the TODO below). The field is referenced in the brief as
``Atmospheres``; we look it up defensively under a few likely keys.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Candidate keys under which the atmosphere may appear in a glaze record.
_ATMOSPHERE_KEYS = ("Atmospheres", "atmospheres", "Atmosphere", "atmosphere")

# Map raw Glazy atmosphere labels to a small normalised vocabulary.
_NORMALISE = {
    "oxidation": "oxidation",
    "oxidising": "oxidation",
    "oxidizing": "oxidation",
    "reduction": "reduction",
    "reducing": "reduction",
    "neutral": "neutral",
}


def _normalise_atmosphere(raw: Any) -> str | None:
    """Normalise a raw atmosphere value to oxidation/reduction/neutral/None.

    The dump may store the atmosphere as a string, a list, or a dict. We take
    the first recognisable label.

    Args:
        raw: Raw value pulled from the YAML record.

    Returns:
        A normalised atmosphere string, or ``None`` if unknown/empty.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        raw = raw.get("name") or next(iter(raw.values()), None)
    if isinstance(raw, (list, tuple)):
        for item in raw:
            result = _normalise_atmosphere(item)
            if result is not None:
                return result
        return None
    text = str(raw).strip().lower()
    for key, value in _NORMALISE.items():
        if key in text:
            return value
    return None


def _iter_records(parsed: Any) -> list[dict[str, Any]]:
    """Yield glaze records from the parsed YAML, regardless of top-level shape.

    The dump may be a list of records or a dict keyed by id. This normalises
    both into a flat list of dicts.
    """
    if isinstance(parsed, list):
        return [r for r in parsed if isinstance(r, dict)]
    if isinstance(parsed, dict):
        # Either {id: record} or {"glazes": [...]} style.
        for container_key in ("glazes", "data", "records"):
            inner = parsed.get(container_key)
            if isinstance(inner, list):
                return [r for r in inner if isinstance(r, dict)]
        return [r for r in parsed.values() if isinstance(r, dict)]
    return []


def parse_atmospheres(yaml_gz_path: str | Path) -> pd.DataFrame:
    """Extract ``id`` → ``atmosphere`` from the gzipped Glazy YAML dump.

    Args:
        yaml_gz_path: Path to ``glazy_YYYYMMDD.yaml.gz``.

    Returns:
        A dataframe with columns ``id`` (int) and ``atmosphere`` (str or None),
        one row per recipe that carries an id.

    TODO(lever-1): confirm the real YAML schema (key name, value shape) against
        an actual dump and tighten ``_ATMOSPHERE_KEYS`` / ``_normalise_atmosphere``
        accordingly; add a unit test on a small fixture.
    """
    path = Path(yaml_gz_path)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle)

    rows: list[dict[str, Any]] = []
    for record in _iter_records(parsed):
        recipe_id = record.get("id") or record.get("Id") or record.get("ID")
        if recipe_id is None:
            continue
        raw_atmosphere = next((record[key] for key in _ATMOSPHERE_KEYS if key in record), None)
        rows.append({"id": int(recipe_id), "atmosphere": _normalise_atmosphere(raw_atmosphere)})

    return pd.DataFrame(rows, columns=["id", "atmosphere"])


def join_atmosphere(features: pd.DataFrame, atmospheres: pd.DataFrame) -> pd.DataFrame:
    """Left-join the atmosphere onto a features/recipes dataframe by ``id``.

    Args:
        features: Recipes/features dataframe containing an ``id`` column.
        atmospheres: Output of :func:`parse_atmospheres`.

    Returns:
        ``features`` with an added ``atmosphere`` column (NaN where unknown).
    """
    return features.merge(atmospheres, on="id", how="left")
