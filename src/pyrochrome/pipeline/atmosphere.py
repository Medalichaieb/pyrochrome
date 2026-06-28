"""Parse the Glazy YAML dump to recover the firing **atmosphere** per recipe.

This is **priority lever #1**: the atmosphere is present in
``glazy_YYYYMMDD.yaml.gz`` but **absent from the flat CSV**. Recovering it and
joining by ``id`` is expected to give the largest accuracy gain, because many
colorants (notably copper: green ↔ red) are only decidable given the atmosphere.

Real schema (confirmed against ``glazy_20260531.yaml.gz``): the dump is a YAML
list of records; each record has an ``ID`` and, optionally, an ``Atmospheres``
key holding a **list** of labels drawn from::

    Oxidation, Reduction, Neutral, Salt & Soda, Wood, Raku, Luster

A recipe is frequently tagged with **several** atmospheres (e.g.
``['Oxidation', 'Reduction']``), so we encode it as **multi-hot** — one binary
column per atmosphere — rather than collapsing to a single category (which would
lose information and bias toward the first label). Note the tag is the set of
atmospheres the recipe is *associated with*, not necessarily the single one
under which its recorded photo/RGB was fired — a known source of label noise.

Inputs : path to the gzipped YAML dump.
Outputs: a dataframe with ``id`` + one ``atm_<name>`` 0/1 column per atmosphere,
         ready to left-join onto the features table by ``id``.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pyrochrome.pipeline.download import find_yaml_dump

# Parsed-atmosphere cache. Parsing the 7.7 MB YAML dump is slow (seconds), so we
# build this once and reuse it; it is git-ignored under data/processed/.
PROCESSED_CACHE = Path("data/processed/atmospheres.csv")

# Candidate keys under which the atmosphere list may appear in a record.
_ATMOSPHERE_KEYS = ("Atmospheres", "atmospheres", "Atmosphere", "atmosphere")

# Map a (lowercased) substring of a raw label to its canonical atmosphere name.
# Order here defines the canonical column order below.
_CANONICAL: dict[str, str] = {
    "oxidation": "oxidation",
    "oxidising": "oxidation",
    "oxidizing": "oxidation",
    "reduction": "reduction",
    "reducing": "reduction",
    "neutral": "neutral",
    "salt": "salt_soda",
    "soda": "salt_soda",
    "wood": "wood",
    "raku": "raku",
    "luster": "luster",
    "lustre": "luster",
}

# Canonical atmosphere names, in a stable order. Feature columns are these
# prefixed with ``atm_`` (plus the ``atm_known`` indicator added on join).
ATMOSPHERE_NAMES: list[str] = [
    "oxidation",
    "reduction",
    "neutral",
    "salt_soda",
    "wood",
    "raku",
    "luster",
]
ATMOSPHERE_COLUMNS: list[str] = [f"atm_{name}" for name in ATMOSPHERE_NAMES]
ATMOSPHERE_KNOWN_COLUMN = "atm_known"


def normalise_atmospheres(raw: Any) -> set[str]:
    """Return the set of canonical atmosphere names present in a raw value.

    Handles the real list shape as well as defensive fallbacks (string, dict).

    Args:
        raw: Raw value pulled from the ``Atmospheres`` key.

    Returns:
        A set of canonical names (subset of :data:`ATMOSPHERE_NAMES`); empty if
        nothing is recognised.
    """
    if raw is None:
        return set()
    if isinstance(raw, dict):
        raw = list(raw.values())
    items = raw if isinstance(raw, (list, tuple)) else [raw]

    found: set[str] = set()
    for item in items:
        text = str(item).strip().lower()
        for substring, canonical in _CANONICAL.items():
            if substring in text:
                found.add(canonical)
    return found


def _iter_records(parsed: Any) -> list[dict[str, Any]]:
    """Yield glaze records from the parsed YAML, regardless of top-level shape.

    The dump is a list of records; this also tolerates a dict keyed by id or a
    ``{"glazes": [...]}`` wrapper.
    """
    if isinstance(parsed, list):
        return [r for r in parsed if isinstance(r, dict)]
    if isinstance(parsed, dict):
        for container_key in ("glazes", "data", "records"):
            inner = parsed.get(container_key)
            if isinstance(inner, list):
                return [r for r in inner if isinstance(r, dict)]
        return [r for r in parsed.values() if isinstance(r, dict)]
    return []


def parse_atmospheres(yaml_gz_path: str | Path) -> pd.DataFrame:
    """Extract ``id`` + multi-hot atmosphere columns from the gzipped YAML dump.

    Only records that carry at least one recognised atmosphere are emitted; the
    rest are left for the join to mark as unknown.

    Args:
        yaml_gz_path: Path to ``glazy_YYYYMMDD.yaml.gz``.

    Returns:
        A dataframe with columns ``id`` (int) and the
        :data:`ATMOSPHERE_COLUMNS` (0/1 int), one row per recipe with a known
        atmosphere.
    """
    path = Path(yaml_gz_path)
    # Prefer the libyaml C loader (≈6× faster on this 7.7 MB dump) when present.
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        parsed = yaml.load(handle, Loader=loader)  # noqa: S506 - CSafeLoader is safe

    rows: list[dict[str, Any]] = []
    for record in _iter_records(parsed):
        recipe_id = record.get("ID") or record.get("id") or record.get("Id")
        if recipe_id is None:
            continue
        raw = next((record[key] for key in _ATMOSPHERE_KEYS if key in record), None)
        names = normalise_atmospheres(raw)
        if not names:
            continue
        row: dict[str, Any] = {"id": int(recipe_id)}
        for name in ATMOSPHERE_NAMES:
            row[f"atm_{name}"] = int(name in names)
        rows.append(row)

    # Dedupe defensively so a left-join can never fan out rows.
    return pd.DataFrame(rows, columns=["id", *ATMOSPHERE_COLUMNS]).drop_duplicates(
        subset="id", keep="first"
    )


def load_atmospheres(
    yaml_gz_path: str | Path | None = None,
    cache_path: str | Path = PROCESSED_CACHE,
    *,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Load the parsed atmosphere table, building and caching it on first use.

    On a cache miss (or ``rebuild=True``) the YAML dump is parsed and the result
    is written to ``cache_path`` so subsequent calls are instant.

    Args:
        yaml_gz_path: Explicit path to the YAML dump, or ``None`` to auto-detect.
        cache_path: Where the parsed table is cached (CSV).
        rebuild: Force re-parsing even if the cache exists.

    Returns:
        The atmosphere table (``id`` + :data:`ATMOSPHERE_COLUMNS`). Empty (with
        the right columns) if no YAML dump is available.
    """
    cache = Path(cache_path)
    if cache.exists() and not rebuild:
        return pd.read_csv(cache)

    path = Path(yaml_gz_path) if yaml_gz_path is not None else find_yaml_dump()
    if path is None:
        return pd.DataFrame(columns=["id", *ATMOSPHERE_COLUMNS])

    atmospheres = parse_atmospheres(path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    atmospheres.to_csv(cache, index=False)
    return atmospheres


def join_atmosphere(features: pd.DataFrame, atmospheres: pd.DataFrame) -> pd.DataFrame:
    """Left-join multi-hot atmosphere columns onto a features dataframe by ``id``.

    Recipes with no atmosphere record get all-zero ``atm_*`` columns and
    ``atm_known = 0`` (so the model can tell "unknown" from "tagged oxidation").

    Args:
        features: Recipes/features dataframe containing an ``id`` column.
        atmospheres: Output of :func:`parse_atmospheres`.

    Returns:
        ``features`` with the :data:`ATMOSPHERE_COLUMNS` and
        :data:`ATMOSPHERE_KNOWN_COLUMN` added.
    """
    merged = features.merge(atmospheres, on="id", how="left")
    merged[ATMOSPHERE_KNOWN_COLUMN] = merged[ATMOSPHERE_COLUMNS[0]].notna().astype(int)
    merged[ATMOSPHERE_COLUMNS] = merged[ATMOSPHERE_COLUMNS].fillna(0).astype(int)
    return merged


def main() -> None:
    """Entry point for ``make atmosphere``: build and cache the atmosphere table."""
    atmospheres = load_atmospheres(rebuild=True)
    if atmospheres.empty:
        print("No YAML dump found — atmosphere feature unavailable. Run `make data`.")
        return
    counts = atmospheres[ATMOSPHERE_COLUMNS].sum()
    print(f"Cached {len(atmospheres)} recipes with a known atmosphere -> {PROCESSED_CACHE}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
