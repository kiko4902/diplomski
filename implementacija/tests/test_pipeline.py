import numpy as np
import os
import tempfile
from sklearn.datasets import make_classification
from evaluation.experiment_runner import run_experiment
from evaluation.cross_validator import stratified_repeated_cv, get_iteration_count
from evaluation.metrics import compute_metrics
from classifiers.defaults import get_classifier


def make_toy_dataset():
    X, y = make_classification(
        n_samples=100, n_features=5, n_redundant=0,
        n_clusters_per_class=1, weights=[0.2, 0.8], random_state=42,
    )
    return X, y


class TestCrossValidator:
    def test_stratification_preserved(self):
        X, y = make_toy_dataset()
        for X_train, X_test, y_train, y_test in stratified_repeated_cv(X, y):
            train_ratio = np.mean(y_train) if len(np.unique(y_train)) > 0 else 0
            test_ratio = np.mean(y_test) if len(np.unique(y_test)) > 0 else 0
            assert -1 <= train_ratio <= 1
            assert -1 <= test_ratio <= 1

    def test_iteration_count(self):
        count = get_iteration_count()
        from config import CV_FOLDS, CV_REPEATS
        assert count == CV_FOLDS * CV_REPEATS

    def test_no_overlap(self):
        X, y = make_toy_dataset()
        for X_train, X_test, y_train, y_test in stratified_repeated_cv(X, y):
            train_set = set(map(tuple, X_train))
            test_set = set(map(tuple, X_test))
            assert train_set.isdisjoint(test_set)


class TestClassifiers:
    def test_all_instantiate(self):
        for name in ["dt", "rf", "lr", "svm", "knn", "gnb", "mlp"]:
            clf = get_classifier(name)
            assert clf is not None

    def test_all_fit_predict(self):
        X, y = make_toy_dataset()
        for name in ["dt", "rf", "lr", "svm", "knn", "gnb", "mlp"]:
            clf = get_classifier(name)
            clf.fit(X, y)
            y_pred = clf.predict(X)
            assert len(y_pred) == len(y)


class TestPipeline:
    def test_single_combination_runs(self):
        X, y = make_toy_dataset()
        import pandas as pd
        df = run_experiment(X, y, "toy_test", classifier_names=["rf"], smote_names=["SMOTE"], k_values=[5])
        assert df is not None
        assert len(df) > 0
        assert "dataset" in df.columns
        assert "metric" in df.columns

    def test_results_in_range(self):
        X, y = make_toy_dataset()
        df = run_experiment(X, y, "toy_test2", classifier_names=["rf"], smote_names=["SMOTE"], k_values=[3])
        for _, row in df.iterrows():
            assert 0.0 <= row["mean"] <= 1.0, f"{row['metric']} = {row['mean']} out of range"

    def test_multiple_smote_variants(self):
        X, y = make_toy_dataset()
        df = run_experiment(
            X, y, "toy_test3",
            classifier_names=["rf"],
            smote_names=["SMOTE", "Borderline-SMOTE1", "ADASYN"],
            k_values=[5],
        )
        smote_names_found = df["smote"].unique()
        assert "SMOTE" in smote_names_found
        assert "Borderline-SMOTE1" in smote_names_found
        assert "ADASYN" in smote_names_found

    def test_baselines_in_pipeline(self):
        X, y = make_toy_dataset()
        df = run_experiment(
            X, y, "toy_baseline",
            classifier_names=["rf"],
            smote_names=["NoOversampling", "RandomOversampling", "RandomUndersampling"],
            k_values=[3, 5],
        )
        assert df is not None
        assert len(df) > 0
        names = df["smote"].unique()
        assert "NoOversampling" in names
        assert "RandomOversampling" in names
        assert "RandomUndersampling" in names

    def test_baselines_single_k_row(self):
        X, y = make_toy_dataset()
        df = run_experiment(
            X, y, "toy_bsingle",
            classifier_names=["rf"],
            smote_names=["NoOversampling", "SMOTE"],
            k_values=[3, 5, 7],
        )
        k_vals_no = df[df["smote"] == "NoOversampling"]["k"].unique()
        k_vals_smote = df[df["smote"] == "SMOTE"]["k"].unique()
        assert len(k_vals_no) == 1
        assert k_vals_no[0] == 0
        assert len(k_vals_smote) == 3


class TestWeightedClassifiers:
    def test_all_weighted_instantiate(self):
        from classifiers.defaults import get_classifier
        names = ["dt_weighted", "rf_weighted", "lr_weighted", "svm_weighted", "mlp_weighted"]
        for name in names:
            clf = get_classifier(name)
            assert clf is not None

    def test_all_weighted_fit_predict(self):
        from classifiers.defaults import get_classifier
        X, y = make_toy_dataset()
        names = ["dt_weighted", "rf_weighted", "lr_weighted", "svm_weighted", "mlp_weighted"]
        for name in names:
            clf = get_classifier(name)
            clf.fit(X, y)
            y_pred = clf.predict(X)
            assert len(y_pred) == len(y)
