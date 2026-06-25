import numpy as np
from evaluation.metrics import compute_metrics


class TestMetrics:
    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])

        m = compute_metrics(y_true, y_pred, y_proba)
        assert abs(m["f1"] - 1.0) < 1e-10
        assert abs(m["balanced_accuracy"] - 1.0) < 1e-10
        assert abs(m["mcc"] - 1.0) < 1e-10
        assert abs(m["g_mean"] - 1.0) < 1e-10
        assert abs(m["auc_roc"] - 1.0) < 1e-10
        assert abs(m["auc_pr"] - 1.0) < 1e-10

    def test_worst_classifier(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        y_proba = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])

        m = compute_metrics(y_true, y_pred, y_proba)
        assert abs(m["f1"] - 0.0) < 1e-10
        assert abs(m["balanced_accuracy"] - 0.0) < 1e-10
        assert abs(m["mcc"] + 1.0) < 1e-10

    def test_imbalanced_accuracy_trap(self):
        y_true = np.array([0] * 99 + [1])
        y_pred = np.array([0] * 100)
        y_proba = np.array([0.01] * 100)

        m = compute_metrics(y_true, y_pred, y_proba)
        assert abs(m["f1"] - 0.0) < 1e-10
        assert abs(m["g_mean"] - 0.0) < 1e-10
        assert abs(m["mcc"] - 0.0) < 1e-10

    def test_returns_all_keys(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([0.3, 0.6, 0.4, 0.7])

        m = compute_metrics(y_true, y_pred, y_proba)
        for key in ["f1", "g_mean", "auc_roc", "auc_pr", "balanced_accuracy", "mcc", "f2"]:
            assert key in m
            assert not np.isnan(m[key])
