"""Eksperiment 5.3 — Usporedba data-level, algorithm-level i hybrid pristupa."""

import sys
import numpy as np
from evaluation.experiment_runner_53 import run_experiment_53
from evaluation.experiment_runner import _register_smote_variants
from data.generate_synthetic import generate_synthetic_sets, print_dataset_summary
from data.make_dataset import load_all_real_datasets


def main():
    print("=" * 60)
    print("  Experiment 5.3 — Paradigm Comparison")
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

    smote_variants = _register_smote_variants()

    methods = [
        ("Baseline", "none"),
        ("NoOversampling", "NoOversampling"),
        ("ROS", "RandomOversampling"),
        ("RUS", "RandomUndersampling"),
        ("SMOTE", "SMOTE"),
        ("WGAN", "WGAN"),
    ]

    classifier_pairs = [
        ("RF", "rf"),
        ("RF_weighted", "rf_weighted"),
        ("DT", "dt"),
        ("DT_weighted", "dt_weighted"),
        ("LR", "lr"),
        ("LR_weighted", "lr_weighted"),
        ("SVM", "svm"),
        ("SVM_weighted", "svm_weighted"),
    ]

    n = len(methods) * len(classifier_pairs) * len(datasets) * len(np.array([]))
    print(f"\n[3] Running experiments...")
    print(f"    Methods:    {len(methods)}")
    print(f"    Classifier pairs: {len(classifier_pairs)}")
    print(f"    Datasets:   {len(datasets)} ({len(real)} real + {len(synth)} synthetic)")
    print()

    for name, (X, y, meta) in datasets.items():
        print(f"\n--- {name} ({meta['n_samples']} samples, IR={meta['ir']:.1f}) ---")
        run_experiment_53(
            X, y, name,
            smote_variants=smote_variants,
            methods=methods,
            classifier_pairs=classifier_pairs,
        )

    print("\n" + "=" * 60)
    print("  Done. Results saved in results/raw/exp_53/")
    print("=" * 60)


if __name__ == "__main__":
    main()
