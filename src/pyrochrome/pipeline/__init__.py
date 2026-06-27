"""Data pipeline: download → parse YAML → join atmosphere → clean → features.

Each step documents its inputs, outputs and assumptions in its module docstring.
The canonical feature representation is:

    UMF oxides (precomputed by Glazy)
      + aggregates R2O_umf, RO_umf, SiO2_Al2O3_ratio_umf
      + cone (ordinal)
      + atmosphere (categorical)
"""
