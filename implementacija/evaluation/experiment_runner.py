import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import SMOTE_K_VALUES, METRICS, RAW_DIR
from evaluation.cross_validator import stratified_repeated_cv
from evaluation.metrics import compute_metrics
from classifiers.defaults import get_classifier


SMOTE_VARIANTS = {}


BASELINE_NAMES = {"NoOversampling", "RandomOversampling", "RandomUndersampling", "WGAN"}


def _register_smote_variants():
    from smote_variants.smote import SMOTE
    from smote_variants.borderline import BorderlineSMOTE
    from smote_variants.adasyn import ADASYN
    from smote_variants.safe_level import SafeLevelSMOTE
    from smote_variants.kmeans_smote import KMeansSMOTE
    from smote_variants.svm_smote import SVMSMOTE
    from smote_variants.smote_enn import SMOTEENN, SMOTETomek
    from smote_variants.g_smote import GeometricSMOTE
    from smote_variants.random_smote import RandomSMOTE
    from smote_variants.polynom_fit import PolynomFitSMOTE
    from smote_variants.baselines import (
        NoOversampling,
        RandomOversampling,
        RandomUndersampling,
    )
    from smote_variants.undersampling import (
        NearMiss1,
        NearMiss2,
        NearMiss3,
        TomekLinks,
        ENN,
    )
    from smote_variants.gan import WGAN

    variants = {
        "SMOTE": SMOTE,
        "Borderline-SMOTE1": lambda **kw: BorderlineSMOTE(kind="BS1", **kw),
        "Borderline-SMOTE2": lambda **kw: BorderlineSMOTE(kind="BS2", **kw),
        "ADASYN": ADASYN,
        "SafeLevel-SMOTE": SafeLevelSMOTE,
        "KMeans-SMOTE": KMeansSMOTE,
        "SVM-SMOTE": SVMSMOTE,
        "SMOTE-ENN": SMOTEENN,
        "SMOTE-Tomek": SMOTETomek,
        "G-SMOTE": GeometricSMOTE,
        "Random-SMOTE": RandomSMOTE,
        "PolynomFit-SMOTE": PolynomFitSMOTE,
        "NoOversampling": NoOversampling,
        "RandomOversampling": RandomOversampling,
        "RandomUndersampling": RandomUndersampling,
        "NearMiss-1": NearMiss1,
        "NearMiss-2": NearMiss2,
        "NearMiss-3": NearMiss3,
        "TomekLinks": TomekLinks,
        "ENN": ENN,
        "WGAN": WGAN,
    }
    return variants


def run_experiment(X, y, dataset_name, classifier_names=None, smote_names=None, k_values=None):
    if classifier_names is None:
        classifier_names = ["rf"]
    if smote_names is None:
        smote_names = ["SMOTE"]
    if k_values is None:
        k_values = SMOTE_K_VALUES

    variants = _register_smote_variants()
    all_rows = []

    for smote_name in smote_names:
        smote_cls = variants[smote_name]
        k_loop = [0] if smote_name in BASELINE_NAMES else k_values
        for k_val in k_loop:
            for clf_name in classifier_names:
                print(f"  {dataset_name} | {smote_name} (k={k_val}) | {clf_name}", end="", flush=True)

                clf = get_classifier(clf_name)
                fold_results = {m: [] for m in METRICS}
                iteration = 0

                for X_train, X_test, y_train, y_test in stratified_repeated_cv(X, y):
                    iteration += 1
                    try:
                        smote = smote_cls(k=k_val, random_state=42 + iteration)
                        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

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
                    except Exception as e:
                        continue

                for m_name in METRICS:
                    vals = fold_results[m_name]
                    if vals:
                        all_rows.append({
                            "dataset": dataset_name,
                            "smote": smote_name,
                            "k": k_val,
                            "classifier": clf_name,
                            "metric": m_name,
                            "mean": np.mean(vals),
                            "std": np.std(vals),
                        })

                print(" OK")

    if all_rows:
        df = pd.DataFrame(all_rows)
        os.makedirs(RAW_DIR, exist_ok=True)
        filepath = os.path.join(RAW_DIR, f"results_{dataset_name}.csv")
        df.to_csv(filepath, index=False)
        print(f"  Saved {len(all_rows)} rows to {filepath}")
        return df

    return None
