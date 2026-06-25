"""PUNI RUN — za finalni rad. Svi klasifikatori, k=3,5,7, CV=30, svi datasetovi.
Pokrenuti kada su svi ostali rezultati spremni i treba finalna statistika.
Procijenjeno trajanje: 8-12 sati.
"""

import sys, os
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))

import config
config.SMOTE_K_VALUES.clear()
config.SMOTE_K_VALUES.extend([3, 5, 7])
config.CV_REPEATS = 30

from evaluation.experiment_runner import run_experiment
from data.make_dataset import load_all_real_datasets
from data.generate_synthetic import generate_synthetic_sets

SMOTE_ALL = [
    "SMOTE", "Borderline-SMOTE1", "Borderline-SMOTE2", "ADASYN",
    "SafeLevel-SMOTE", "KMeans-SMOTE", "SVM-SMOTE",
    "SMOTE-ENN", "SMOTE-Tomek", "G-SMOTE", "Random-SMOTE", "PolynomFit-SMOTE",
    "NearMiss-1", "NearMiss-2", "NearMiss-3", "TomekLinks", "ENN",
    "NoOversampling", "RandomOversampling", "RandomUndersampling",
]

ALL_CLASSIFIERS = ["dt", "rf", "lr", "svm", "knn", "gnb", "mlp", "xgboost"]

print("=" * 60)
print("  FINAL RUN — Full Configuration")
print(f"  k = {config.SMOTE_K_VALUES}, CV = {config.CV_REPEATS} repeats")
print(f"  Classifiers: {len(ALL_CLASSIFIERS)} ({', '.join(ALL_CLASSIFIERS)})")
print(f"  Methods: {len(SMOTE_ALL)}")
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

real = load_all_real_datasets()
synth = generate_synthetic_sets()

for name, (X, y, meta) in {**real, **synth}.items():
    print(f"\n--- {name} (n={meta['n_samples']}, d={meta['n_features']}, IR={meta['ir']:.1f}) ---")
    run_experiment(X, y, name,
                   classifier_names=ALL_CLASSIFIERS,
                   smote_names=SMOTE_ALL,
                   k_values=config.SMOTE_K_VALUES)

print(f"\nDone. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Saved to F:\\results\\raw\\")
