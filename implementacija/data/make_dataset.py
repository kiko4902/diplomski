"""Učitavanje realnih neuravnoteženih skupova podataka s metapodacima."""

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_wine,
    load_iris,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


def _to_binary(y):
    classes, counts = np.unique(y, return_counts=True)
    minority_class = classes[np.argmin(counts)]
    return (y == minority_class).astype(int)


def _compute_ir(y):
    _, counts = np.unique(y, return_counts=True)
    return round(max(counts) / counts.min(), 2)


def _compute_separability(X, y):
    """Proxy separabilnosti: F1 logističke regresije (3-fold CV), bez resampliranja."""
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(lr, X_scaled, y, cv=3, scoring="f1")
        return round(float(scores.mean()), 3)
    except Exception:
        return None


def _make_meta(name, X, y, source):
    ir = _compute_ir(y)
    sep = _compute_separability(X, y)
    return {
        "name": name,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "ir": ir,
        "separability_f1": sep,
        "source": source,
        "type": "real",
    }


def load_sklearn_datasets():
    datasets = {}

    data = load_breast_cancer()
    X, y = data.data.astype(np.float64), data.target.astype(int)
    datasets["breast_cancer"] = (X, y, _make_meta("breast_cancer", X, y, "sklearn"))

    data = load_wine()
    X, y = data.data.astype(np.float64), _to_binary(data.target)
    datasets["wine"] = (X, y, _make_meta("wine", X, y, "sklearn"))

    data = load_iris()
    X, y = data.data.astype(np.float64), _to_binary(data.target)
    datasets["iris"] = (X, y, _make_meta("iris", X, y, "sklearn"))

    return datasets


def load_imblearn_datasets():
    datasets = {}
    try:
        from imblearn.datasets import fetch_datasets
        ds = fetch_datasets()
    except ImportError:
        print("  imbalanced-learn nije instaliran; koristite 'pip install imbalanced-learn'")
        return datasets

    selected = [
        "ecoli",
        "abalone",
        "wine_quality",
        "yeast_me2",
        "optical_digits",
        "satimage",
        "libras_move",
        "us_crime",
    ]

    for name in selected:
        if name not in ds:
            print(f"  Dataset '{name}' not found in imbalanced-learn")
            continue
        bunch = ds[name]
        X = bunch.data.astype(np.float64)
        y = bunch.target.astype(int)
        y = _to_binary(y)
        datasets[name] = (X, y, _make_meta(name, X, y, "imbalanced-learn"))

    return datasets


def load_all_real_datasets():
    datasets = {}
    datasets.update(load_sklearn_datasets())
    datasets.update(load_imblearn_datasets())
    return datasets


def print_dataset_summary(datasets, title="Dataset Summary"):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"{'Name':22s} {'Source':18s} {'Samples':>8s} {'Features':>8s} {'IR':>8s} {'Sep. F1':>8s}")
    print("-" * 70)
    for name, (_, _, meta) in sorted(datasets.items(), key=lambda x: x[1][2]["ir"]):
        print(f"{meta['name']:22s} {meta['source']:18s} "
              f"{meta['n_samples']:8d} {meta['n_features']:8d} "
              f"{meta['ir']:8.1f} {meta.get('separability_f1', '-'):>8}")
    print("-" * 70)
