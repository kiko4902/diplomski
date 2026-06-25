"""Glavna skripta za pokretanje svih eksperimenata — 5.2 SMOTE analiza."""

import sys
import numpy as np
from datetime import datetime
from config import SMOTE_K_VALUES, METRICS
from evaluation.experiment_runner import run_experiment
from data.generate_synthetic import generate_synthetic_sets, print_dataset_summary
from data.make_dataset import load_all_real_datasets


SMOTE_ALL = [
    "SMOTE", "Borderline-SMOTE1", "Borderline-SMOTE2", "ADASYN",
    "SafeLevel-SMOTE", "KMeans-SMOTE", "SVM-SMOTE",
    "SMOTE-ENN", "SMOTE-Tomek", "G-SMOTE", "Random-SMOTE", "PolynomFit-SMOTE",
    "NearMiss-1", "NearMiss-2", "NearMiss-3", "TomekLinks", "ENN",
    "NoOversampling", "RandomOversampling", "RandomUndersampling",
]

BASELINE_ALL = {"NoOversampling", "RandomOversampling", "RandomUndersampling"}

CLASSIFIERS_ALL = ["dt", "rf", "lr", "svm", "knn", "gnb", "mlp"]

# Brzi klasifikatori za glavni run (MLP i SVM su prespori za overnight)
CLASSIFIERS_FAST = ["rf", "lr", "dt", "knn", "gnb", "xgboost"]
# Spori klasifikatori — pokrenuti odvojeno: python run_analysis_slow.py
CLASSIFIERS_SLOW = ["svm", "mlp"]


def main():
    use_smote = sys.argv[1:] if len(sys.argv) > 1 else SMOTE_ALL
    use_classifiers = CLASSIFIERS_FAST
    k_vals = SMOTE_K_VALUES
    skip_datasets = set()

    print("=" * 60)
    print("  SMOTE Experimental Analysis — 5.2")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\n[1] Loading real datasets...")
    real = load_all_real_datasets()
    print_dataset_summary(real, "Real Datasets")

    print("\n[2] Generating synthetic datasets...")
    synth = generate_synthetic_sets()
    print_dataset_summary(synth, "Synthetic Datasets")

    datasets = {}
    datasets.update(real)
    datasets.update(synth)

    n_k_baseline = sum(1 for s in use_smote if s in BASELINE_ALL)
    n_k_smote = len(use_smote) - n_k_baseline
    n_undersampling = sum(1 for s in use_smote if s.startswith("NearMiss") or s in ("TomekLinks", "ENN"))
    expected_rows = (n_k_smote * len(k_vals) + n_k_baseline * 1) * len(use_classifiers) * (len(datasets) - len(skip_datasets)) * len(METRICS)

    from config import CV_FOLDS, CV_REPEATS
    print(f"\n[3] Running experiments...")
    print(f"    Total methods:  {len(use_smote)} ({n_k_smote - n_undersampling} SMOTE + {n_undersampling} undersampling + {n_k_baseline} baseline)")
    print(f"    Classifiers:    {len(use_classifiers)} ({', '.join(use_classifiers)})")
    print(f"    k values:       {k_vals}")
    print(f"    CV:             {CV_FOLDS}-fold x {CV_REPEATS} repeats = {CV_FOLDS * CV_REPEATS} fits/comb")
    print(f"    Datasets:       {len(datasets) - len(skip_datasets)} ({len(real) - len(skip_datasets & set(real))} real + {len(synth)} synthetic), {len(skip_datasets)} skipped")
    print(f"    Expected rows:  {expected_rows}")
    print()

    for name, (X, y, meta) in datasets.items():
        if name in skip_datasets:
            print(f"\n--- {name} SKIPPED (vec gotov) ---")
            continue
        source = meta.get("source", meta.get("type", "?"))
        print(f"\n--- {name} ({meta['n_samples']} samples, d={meta['n_features']}, IR={meta['ir']:.1f}, [{source}]) ---")
        run_experiment(
            X, y, name,
            classifier_names=use_classifiers,
            smote_names=use_smote,
            k_values=k_vals,
        )

    print("\n" + "=" * 60)
    print(f"  Done. Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Results saved in results/raw/")
    print("=" * 60)


if __name__ == "__main__":
    main()
