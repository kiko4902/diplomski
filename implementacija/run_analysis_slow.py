"""Poseban run za spore klasifikatore (MLP i SVM) — optimiziran za brzinu.

Optimizacije:
- SVM: bez probability=True (ne treba za F1/G-Mean/MCC, a 5x brze)
- MLP: hidden_layer_sizes=(50,) umjesto (100,)
- Preskacu se 4 najveca dataseta + 3 trivijalna
- CV_REPEATS=5 (25 foldova umjesto 50 glavnog runa)
"""

import sys
from datetime import datetime

# VAZNO: patchaj config PRIJE importa experiment_runner (default arg u cross_validator)
import config as cfg
cfg.CV_REPEATS = 5  # 25 foldova umjesto 50 — SVM i MLP su prespori za 50

from config import SMOTE_K_VALUES, METRICS
from evaluation.experiment_runner import run_experiment
from data.generate_synthetic import generate_synthetic_sets, print_dataset_summary
from data.make_dataset import load_all_real_datasets

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import classifiers.defaults as clf_defaults
clf_defaults.CLASSIFIERS["svm"] = SVC(kernel="rbf", probability=False, cache_size=500, random_state=42)
clf_defaults.CLASSIFIERS["mlp"] = MLPClassifier(
    hidden_layer_sizes=(50,), max_iter=2000, early_stopping=True, random_state=42
)
clf_defaults._CLASS_WEIGHT_CLFS["svm"] = lambda: SVC(
    kernel="rbf", class_weight="balanced", probability=False, cache_size=500, random_state=42
)
clf_defaults._CLASS_WEIGHT_CLFS["mlp"] = lambda: MLPClassifier(
    hidden_layer_sizes=(50,), max_iter=2000, early_stopping=True, random_state=42
)


SMOTE_ALL = [
    "SMOTE", "Borderline-SMOTE1", "Borderline-SMOTE2", "ADASYN",
    "SafeLevel-SMOTE", "KMeans-SMOTE", "SVM-SMOTE",
    "SMOTE-ENN", "SMOTE-Tomek", "G-SMOTE", "Random-SMOTE", "PolynomFit-SMOTE",
    "NearMiss-1", "NearMiss-2", "NearMiss-3", "TomekLinks", "ENN",
    "NoOversampling", "RandomOversampling", "RandomUndersampling",
]

BASELINE_ALL = {"NoOversampling", "RandomOversampling", "RandomUndersampling"}

# Za SVM: preskoci datasetove > 2000 uzoraka (O(n^2) ubija)
LARGE_DATASETS = {"optical_digits", "satimage", "wine_quality", "abalone"}


def main():
    use_smote = sys.argv[1:] if len(sys.argv) > 1 else SMOTE_ALL
    use_classifiers = ["svm", "mlp"]
    k_vals = SMOTE_K_VALUES
    skip_datasets = LARGE_DATASETS | {"breast_cancer", "wine", "iris"}

    print("=" * 60)
    print("  SMOTE — SLOW classifiers (SVM + MLP) [OPTIMIZED]")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("  SVM: probability=False, cache_size=500")
    print("  MLP: hidden=(50,), early_stopping=True")
    print("  CV:  5-fold x 5 repeats = 25 fits/comb")
    print(f"  Skipping: {sorted(skip_datasets)}")
    print()

    print("\n[1] Loading real datasets...")
    real = load_all_real_datasets()
    print_dataset_summary(real, "Real Datasets")

    print("\n[2] Generating synthetic datasets...")
    synth = generate_synthetic_sets()
    print_dataset_summary(synth, "Synthetic Datasets")

    datasets = {}
    datasets.update(real)
    datasets.update(synth)

    n_datasets_run = len(datasets) - len(skip_datasets)
    n_k_baseline = sum(1 for s in use_smote if s in BASELINE_ALL)
    n_k_smote = len(use_smote) - n_k_baseline
    combos_per_ds = n_k_smote * len(k_vals) + n_k_baseline
    expected_rows = combos_per_ds * len(use_classifiers) * n_datasets_run * len(METRICS)

    print(f"\n[3] Running experiments...")
    print(f"    SMOTE variants: {len(use_smote)} ({n_k_smote} SMOTE + {n_k_baseline} baseline)")
    print(f"    Classifiers:    {len(use_classifiers)} ({', '.join(use_classifiers)})")
    print(f"    k values:       {k_vals}")
    print(f"    Datasets:       {n_datasets_run} / {len(datasets)} (skipped {len(skip_datasets)})")
    print(f"    Expected rows:  {expected_rows}")
    print()

    for name, (X, y, meta) in datasets.items():
        if name in skip_datasets:
            print(f"\n--- {name} SKIPPED ---")
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
