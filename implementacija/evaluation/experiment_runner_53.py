"""Eksperiment 5.3 — Usporedba paradigmi: data-level, algorithm-level, hybrid."""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import METRICS, RAW_DIR
from evaluation.cross_validator import stratified_repeated_cv
from evaluation.metrics import compute_metrics
from classifiers.defaults import get_classifier


def _get_oversampler(name, smote_variants):
    if name == "none":
        return None
    return smote_variants[name]


def run_experiment_53(X, y, dataset_name, smote_variants, methods, classifier_pairs):
    """
    Pokreće fokusirani 5.3 eksperiment.

    methods: list of (method_name, oversampler_key)
        npr. [("Baseline", "none"), ("WGAN", "WGAN"), ("ClassWeight", "none")]
            gdje oversampler_key "none" znači bez augmentacije podataka

    classifier_pairs: list of (display_name, classifier_key)
        npr. [("RF", "rf"), ("RF+weights", "rf_weighted")]
    """
    all_rows = []

    for method_name, ovs_key in methods:
        for disp_name, clf_key in classifier_pairs:
            clf = get_classifier(clf_key)
            ovs = _get_oversampler(ovs_key, smote_variants)
            fold_name = f"{method_name} + {disp_name}"
            print(f"  {dataset_name} | {fold_name}", end="", flush=True)

            fold_results = {m: [] for m in METRICS}
            iteration = 0

            for X_train, X_test, y_train, y_test in stratified_repeated_cv(X, y):
                iteration += 1
                try:
                    if ovs is not None:
                        ovs_instance = ovs(k=5, random_state=42 + iteration)
                        X_train_res, y_train_res = ovs_instance.fit_resample(X_train, y_train)
                    else:
                        X_train_res, y_train_res = X_train, y_train

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train_res)
                    X_test_scaled = scaler.transform(X_test)

                    clf.fit(X_train_scaled, y_train_res)
                    y_pred = clf.predict(X_test_scaled)

                    try:
                        y_proba = clf.predict_proba(X_test_scaled)[:, 1]
                    except (AttributeError, IndexError):
                        y_proba = None

                    metrics = compute_metrics(y_test, y_pred, y_proba)
                    for m_name, m_val in metrics.items():
                        fold_results[m_name].append(m_val)
                except Exception:
                    continue

            for m_name in METRICS:
                vals = fold_results[m_name]
                if vals:
                    all_rows.append({
                        "dataset": dataset_name,
                        "method": fold_name,
                        "metric": m_name,
                        "mean": np.mean(vals),
                        "std": np.std(vals),
                    })

            print(" OK")

    if all_rows:
        df = pd.DataFrame(all_rows)
        out_dir = os.path.join(RAW_DIR, "exp_53")
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, f"results_53_{dataset_name}.csv")
        df.to_csv(filepath, index=False)
        print(f"  Saved {len(all_rows)} rows to {filepath}")
        return df

    return None
