"""Generator sintetičkih skupova podataka i učitavanje ugrađenih datasetova."""

import numpy as np
from sklearn.datasets import (
    make_classification,
    load_breast_cancer,
    load_wine,
    load_iris,
)
from sklearn.preprocessing import StandardScaler


def generate_synthetic(name, n_samples=500, n_features=10, ir=5.0, noise=0.05, random_state=42):
    """Generira jedan sintetički skup s kontroliranim parametrima."""
    weight_min = 1.0 / (1.0 + ir)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=0,
        n_informative=n_features,
        n_clusters_per_class=1,
        weights=[weight_min, 1.0 - weight_min],
        flip_y=noise,
        random_state=random_state,
    )
    return X, y, {
        "name": name,
        "n_samples": n_samples,
        "n_features": n_features,
        "ir": ir,
        "type": "synthetic",
        "noise": noise,
    }


def generate_synthetic_sets():
    """Generira set sintetičkih skupova za eksperimente."""
    datasets = {}

    configs = [
        ("synth_low_ir",    500,  10,   2.0, 0.05, 42),
        ("synth_medium_ir", 500,  10,   5.0, 0.05, 43),
        ("synth_high_ir",   500,  10,  20.0, 0.05, 44),
        ("synth_low_dim",   500,   3,   5.0, 0.05, 45),
        ("synth_high_dim",  500,  50,   5.0, 0.05, 46),
        ("synth_noisy",     500,  10,   5.0, 0.15, 47),
        ("synth_clean",     500,  10,   5.0, 0.00, 48),
    ]

    for cfg in configs:
        X, y, meta = generate_synthetic(*cfg)
        datasets[meta["name"]] = (X, y, meta)

    return datasets


def load_imb_dataset(name):
    """
    Učitava ugrađeni dataset iz sklearn-a i pretvara ga u binarni problem.
    Vraća (X, y, meta).
    """
    if name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
        return X, y, {
            "name": "breast_cancer",
            "n_samples": len(X),
            "n_features": X.shape[1],
            "ir": round(np.mean(y == 0) / max(np.mean(y == 1), 1e-9), 1),
            "type": "real",
        }

    elif name == "wine":
        data = load_wine()
        X, y = data.data, data.target
        y_bin = (y == 0).astype(int)
        return X, y_bin, {
            "name": "wine",
            "n_samples": len(X),
            "n_features": X.shape[1],
            "ir": round(np.mean(y_bin == 0) / max(np.mean(y_bin == 1), 1e-9), 1),
            "type": "real",
        }

    elif name == "iris":
        data = load_iris()
        X, y = data.data, data.target
        y_bin = (y == 0).astype(int)
        return X, y_bin, {
            "name": "iris",
            "n_samples": len(X),
            "n_features": X.shape[1],
            "ir": round(np.mean(y_bin == 0) / max(np.mean(y_bin == 1), 1e-9), 1),
            "type": "real",
        }

    else:
        raise ValueError(f"Unknown dataset: {name}")


def load_all_datasets():
    """Učitava sve ugrađene datasetove."""
    datasets = {}
    for name in ["breast_cancer", "wine", "iris"]:
        try:
            X, y, meta = load_imb_dataset(name)
            datasets[name] = (X, y, meta)
        except Exception as e:
            print(f"  Skipping {name}: {e}")
    return datasets


def print_dataset_summary(datasets, title="Dataset Summary"):
    """Ispisuje tablicu svih učitanih skupova podataka."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"{'Name':25s} {'Samples':>8s} {'Features':>8s} {'IR':>6s} {'Type':>8s}")
    print("-" * 60)
    items = [(name, X, y, meta) for name, (X, y, meta) in datasets.items()]
    items.sort(key=lambda x: (x[3]["type"], -x[3]["ir"]))
    for name, X, y, meta in items:
        print(f"{meta['name']:25s} {meta['n_samples']:8d} {meta['n_features']:8d} {meta['ir']:6.1f} {meta['type']:>8s}")
    print("-" * 60)
