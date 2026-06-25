"""Brzi test: isplati li se k=7,10?"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from evaluation.experiment_runner import run_experiment
from data.make_dataset import load_all_real_datasets
from data.generate_synthetic import generate_synthetic_sets

# Samo 3 reprezentativne SMOTE varijante
SMOTE_SUBSET = ["SMOTE", "ADASYN", "Borderline-SMOTE1"]

# Samo 2 klasifikatora
CLASSIFIERS = ["rf", "lr"]

# Raznoliki datasetovi
TEST_DATASETS = ["ecoli", "yeast_me2", "optical_digits", "synth_high_ir", "synth_clean"]

real = load_all_real_datasets()
synth = generate_synthetic_sets()

for name in TEST_DATASETS:
    if name in real:
        X, y, meta = real[name]
    else:
        X, y, meta = synth[name]

    print(f"\n--- {name} (IR={meta['ir']:.1f}, d={meta['n_features']}, n={meta['n_samples']}) ---")
    run_experiment(X, y, name + "_k7_k10",
                   classifier_names=CLASSIFIERS,
                   smote_names=SMOTE_SUBSET,
                   k_values=[7, 10])

print("\nDone! Compare with results_{name}.csv vs results_{name}_k7_k10.csv")
