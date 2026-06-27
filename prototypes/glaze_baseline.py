#!/usr/bin/env python3
"""
glaze_baseline.py — Proof-of-concept : prédire les propriétés post-cuisson d'un
émail (surface, famille de couleur) à partir de sa composition chimique.

Données  : Glazy public data — https://github.com/derekphilipau/glazy-data
           Fichier utilisé : glazy-data-glazes-<date>.csv (colonnes nommées,
           composition pré-calculée en UMF par Glazy).
Licence  : les données Glazy sont sous CC BY-NC-SA 4.0 (attribution,
           partage à l'identique, USAGE NON COMMERCIAL). Tout produit dérivé
           doit respecter ces termes.
Référence: GlazyBench, Zhai et al., arXiv:2605.06641 (2026) — même tâche,
           méthodo de nettoyage des couleurs et baselines à reproduire/citer.

Usage :
    git clone --depth 1 https://github.com/derekphilipau/glazy-data.git
    pip install pandas scikit-learn
    python glaze_baseline.py glazy-data/glazy-data-glazes-20211130.csv

Ce qu'il fait :
    - charge les recettes, mappe le cône Orton sur une échelle ordinale ;
    - construit le vecteur de features = oxydes UMF + agrégats (R2O, RO,
      ratio SiO2:Al2O3) + cône ;
    - entraîne RandomForest + GradientBoosting sur deux cibles :
        (1) surface  -> Glossy / Matte / Satin
        (2) couleur  -> 10 familles dérivées du RGB (étiquettes bruitées)
    - compare à une baseline naïve et affiche des exemples.

LIMITES CONNUES (à corriger dans le vrai build — voir le brief) :
    - L'ATMOSPHÈRE (oxydation/réduction) n'est PAS dans ce CSV plat. Elle est
      dans glazy_<date>.yaml.gz, jointable par `id`. C'est la feature n°1
      manquante ; sans elle le cuivre (vert<->rouge) est indécidable.
    - Les étiquettes couleur viennent de photos non standardisées -> bruitées.
"""
import sys
import warnings
import colorsys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

warnings.filterwarnings("ignore")

DEFAULT_CSV = "glazy-data/glazy-data-glazes-20211130.csv"

# Cônes Orton, du plus froid au plus chaud -> index ordinal.
CONE_ORDER = ['022', '021', '020', '019', '018', '017', '016', '015', '014',
              '013', '012', '011', '010', '09', '08', '07', '06', '05', '05.5',
              '04', '03', '02', '01', '1', '2', '3', '4', '5', '5.5', '6', '7',
              '8', '9', '10', '11', '12', '13', '14']
CONE_IDX = {c: i for i, c in enumerate(CONE_ORDER)}

COLORANTS = {'CoO_umf': 'cobalt', 'CuO_umf': 'cuivre', 'Fe2O3_umf': 'fer',
             'Cr2O3_umf': 'chrome', 'MnO_umf': 'manganèse', 'NiO_umf': 'nickel',
             'TiO2_umf': 'titane', 'SnO2_umf': 'étain', 'ZrO2_umf': 'zircon'}


def cone_to_num(x):
    if pd.isna(x):
        return np.nan
    x = str(x).replace('&#189;', '.5').replace(' ', '')
    return CONE_IDX.get(x, np.nan)


def color_family(r, g, b):
    """Bucket un RGB en famille de couleur perceptuelle (via HSV)."""
    if pd.isna(r):
        return np.nan
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    h *= 360
    if v < 0.16:
        return 'Noir'
    if s < 0.13:
        return 'Blanc' if v > 0.78 else 'Gris'
    if h < 22 or h >= 330:
        base = 'Rouge'
    elif h < 45:
        base = 'Orange'
    elif h < 68:
        base = 'Jaune'
    elif h < 160:
        base = 'Vert'
    elif h < 200:
        base = 'Turquoise'
    elif h < 255:
        base = 'Bleu'
    else:
        base = 'Violet'
    if base in ('Orange', 'Jaune', 'Rouge') and v < 0.55 and s > 0.2:
        base = 'Brun'
    return base


def surface_family(x):
    if pd.isna(x):
        return np.nan
    x = str(x)
    for k in ('Glossy', 'Matte', 'Satin'):
        if x.startswith(k):
            return k
    return np.nan


def build_features(df):
    df['cone_num'] = df['from_orton_cone'].apply(cone_to_num)
    # Les colonnes *_umf incluent déjà R2O_umf, RO_umf, SiO2_Al2O3_ratio_umf.
    umf_cols = [c for c in df.columns if c.endswith('_umf')]
    feat_cols = list(dict.fromkeys(umf_cols + ['cone_num']))
    X = df[feat_cols].apply(pd.to_numeric, errors='coerce')
    feat_cols = [c for c in feat_cols if X[c].fillna(0).abs().sum() > 0]  # drop zéros
    return feat_cols


def train_target(df, feat_cols, mask, y, name, topk=False):
    X = df.loc[mask, feat_cols].apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())
    yy = y[mask]
    print(f"\n{'=' * 62}\n{name}\n  n={len(X)}   classes={yy.nunique()}")
    print("  Répartition :", dict(yy.value_counts().head(8)))
    Xtr, Xte, ytr, yte = train_test_split(
        X, yy, test_size=0.2, random_state=42, stratify=yy)

    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    print(f"  {'Baseline naïve (classe majoritaire)':42s} "
          f"acc={accuracy_score(yte, dummy.predict(Xte)):.3f}")

    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42,
                                class_weight='balanced_subsample').fit(Xtr, ytr)
    pr = rf.predict(Xte)
    print(f"  {'Random Forest':42s} acc={accuracy_score(yte, pr):.3f}   "
          f"F1-macro={f1_score(yte, pr, average='macro'):.3f}")

    hb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08,
                                        random_state=42).fit(Xtr, ytr)
    ph = hb.predict(Xte)
    print(f"  {'Gradient Boosting':42s} acc={accuracy_score(yte, ph):.3f}   "
          f"F1-macro={f1_score(yte, ph, average='macro'):.3f}")

    if topk:
        proba = rf.predict_proba(Xte)
        t2 = top_k_accuracy_score(yte, proba, k=2, labels=rf.classes_)
        print(f"  -> Random Forest Top-2 : {t2:.3f}")
    return rf, Xte, yte


def show_examples(df, rf, Xte, yte, n=10):
    print(f"\n{'=' * 62}\nEXEMPLES DE PRÉDICTIONS (échantillon test)")
    pred = rf.predict(Xte)
    sample = pd.DataFrame({'pred': pred, 'vrai': yte.values},
                          index=yte.index).sample(n, random_state=5)
    for i, row in sample.iterrows():
        name = str(df.loc[i, 'name'])[:32]
        cols = {lbl: df.loc[i, c] for c, lbl in COLORANTS.items()
                if c in df.columns and df.loc[i, c] > 0.004}
        top = ", ".join(l for l, _ in sorted(cols.items(),
                        key=lambda x: -x[1])[:2]) or "—"
        ok = "OK " if row['pred'] == row['vrai'] else "X  "
        print(f"  {ok}{name:32s} | vrai={str(row['vrai']):9s} "
              f"prédit={str(row['pred']):9s} | colorant: {top}")


def main(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[(df.is_analysis == 0) & (df.is_primitive == 0) &
            (df.is_theoretical == 0)].copy()
    feat_cols = build_features(df)
    print(f"Recettes : {len(df)}   |   Features (UMF + ratios + cône) : {len(feat_cols)}")

    # Cible 1 : surface
    ys = df['surface_type'].apply(surface_family)
    train_target(df, feat_cols, ys.notna(), ys,
                 "CIBLE 1 — SURFACE (Glossy / Matte / Satin)")

    # Cible 2 : famille de couleur (depuis RGB)
    yc = df.apply(lambda r: color_family(r['rgb_r'], r['rgb_g'], r['rgb_b']), axis=1)
    vc = yc.value_counts()
    yc = yc.where(~yc.isin(vc[vc < 40].index))  # retirer familles trop rares
    rf, Xte, yte = train_target(
        df, feat_cols, yc.notna(), yc,
        "CIBLE 2 — FAMILLE DE COULEUR (depuis RGB, étiquettes bruitées)", topk=True)
    show_examples(df, rf, Xte, yte)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV)
