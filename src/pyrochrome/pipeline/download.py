"""Locate (and validate) the Glazy dataset files under ``data/raw/``.

The actual clone is done by ``make data`` (``git clone --depth 1`` of the Glazy
repo). This module's job is to find the latest CSV and YAML dump in the cloned
tree, so the rest of the pipeline doesn't hard-code dated file names.

Inputs : the ``data/raw/`` directory (default) containing ``glazy-data/``.
Outputs: resolved paths to the glazes CSV and the gzipped YAML dump.
"""

from __future__ import annotations

from pathlib import Path

RAW_DIR = Path("data/raw")
GLAZY_DIR = RAW_DIR / "glazy-data"


def find_glazes_csv(root: Path = GLAZY_DIR) -> Path:
    """Return the path to the most recent ``glazy-data-glazes-*.csv``.

    Args:
        root: Directory of the cloned Glazy repo.

    Returns:
        Path to the newest (lexicographically last, dates sort correctly) CSV.

    Raises:
        FileNotFoundError: If no matching CSV is present.
    """
    matches = sorted(root.glob("glazy-data-glazes-*.csv"))
    if not matches:
        raise FileNotFoundError(f"No glazy-data-glazes-*.csv under {root}. Run `make data` first.")
    return matches[-1]


def find_yaml_dump(root: Path = GLAZY_DIR) -> Path | None:
    """Return the path to the most recent ``glazy_*.yaml.gz``, if present.

    Args:
        root: Directory of the cloned Glazy repo.

    Returns:
        Path to the newest YAML dump, or ``None`` if absent (the dump is large
        and may not be checked out in a shallow clone).
    """
    matches = sorted(root.glob("glazy_*.yaml.gz"))
    return matches[-1] if matches else None


def main() -> None:
    """Report which Glazy files were found (entry point for ``make data``)."""
    csv = find_glazes_csv()
    print(f"Glazes CSV : {csv}")
    yaml_dump = find_yaml_dump()
    if yaml_dump is None:
        print("YAML dump  : NOT FOUND — atmosphere feature (lever #1) unavailable.")
    else:
        print(f"YAML dump  : {yaml_dump}")


if __name__ == "__main__":
    main()
