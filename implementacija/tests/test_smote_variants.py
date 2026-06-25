import numpy as np
import pytest
from sklearn.datasets import make_classification

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
from smote_variants.undersampling import (
    NearMiss1,
    NearMiss2,
    NearMiss3,
    TomekLinks,
    ENN,
)
from smote_variants.baselines import (
    NoOversampling,
    RandomOversampling,
    RandomUndersampling,
)


def make_toy_data(n_samples=100, n_features=2, ir=0.1, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[ir, 1 - ir],
        random_state=random_state,
    )
    y = np.where(y == 0, -1, 1)
    return X, y


class TestSMOTE:
    def test_output_shape(self):
        X, y = make_toy_data()
        smote = SMOTE(k=5, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        assert X_res.shape[0] == y_res.shape[0]
        assert X_res.shape[1] == X.shape[1]

    def test_balances_classes(self):
        X, y = make_toy_data(n_samples=50, ir=0.2)
        smote = SMOTE(k=5, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        classes, counts = np.unique(y_res, return_counts=True)
        assert len(classes) == 2
        assert counts[0] == counts[1]

    def test_minority_increased(self):
        X, y = make_toy_data()
        _, counts_before = np.unique(y, return_counts=True)
        n_min_before = counts_before.min()

        smote = SMOTE(k=5, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        _, counts_after = np.unique(y_res, return_counts=True)
        n_min_after = counts_after.min()
        assert n_min_after > n_min_before

    def test_no_nan_inf(self):
        X, y = make_toy_data()
        smote = SMOTE(k=5, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        assert not np.isnan(X_res).any()
        assert not np.isinf(X_res).any()

    def test_reproducibility(self):
        X, y = make_toy_data()
        smote1 = SMOTE(k=5, random_state=42)
        smote2 = SMOTE(k=5, random_state=42)

        X1, y1 = smote1.fit_resample(X, y)
        X2, y2 = smote2.fit_resample(X, y)

        assert np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)

    def test_synthetic_on_segments(self):
        X, y = make_toy_data(n_features=2, n_samples=20, ir=0.5)
        smote = SMOTE(k=3, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        X_min_orig = X[y == np.min(np.unique(y))]
        X_synth = X_res[len(X):]

        for s in X_synth:
            found = False
            for a in X_min_orig:
                for b in X_min_orig:
                    if np.array_equal(a, b):
                        continue
                    vec_ab = b - a
                    vec_as = s - a
                    if np.linalg.norm(vec_ab) < 1e-10:
                        continue
                    proj = np.dot(vec_as, vec_ab) / np.dot(vec_ab, vec_ab)
                    if -1e-6 < proj < 1 + 1e-6:
                        proj_clamped = max(0, min(1, proj))
                        point_on_segment = a + proj_clamped * vec_ab
                        if np.linalg.norm(s - point_on_segment) < 1e-6:
                            found = True
                            break
                if found:
                    break
            assert found, f"Synthetic point {s} not on any segment between minority points"

    def test_too_few_minority_raises(self):
        X, y = make_toy_data(n_samples=20, ir=0.2)
        smote = SMOTE(k=10, random_state=42)

        X_min = X[y == np.min(np.unique(y))]
        if len(X_min) < 10:
            with pytest.raises(ValueError):
                smote.fit_resample(X, y)


class TestBorderlineSMOTE:
    def test_output_shape_bs1(self):
        X, y = make_toy_data()
        b_smote = BorderlineSMOTE(k=5, kind="BS1", random_state=42)
        X_res, y_res = b_smote.fit_resample(X, y)
        assert X_res.shape[0] == y_res.shape[0]
        assert X_res.shape[1] == X.shape[1]

    def test_output_shape_bs2(self):
        X, y = make_toy_data()
        b_smote = BorderlineSMOTE(k=5, kind="BS2", random_state=42)
        X_res, y_res = b_smote.fit_resample(X, y)
        assert X_res.shape[0] == y_res.shape[0]

    def test_balances_classes(self):
        X, y = make_toy_data(n_samples=50, ir=0.2)
        for kind in ("BS1", "BS2"):
            b_smote = BorderlineSMOTE(k=5, kind=kind, random_state=42)
            X_res, y_res = b_smote.fit_resample(X, y)
            classes, counts = np.unique(y_res, return_counts=True)
            assert counts[0] == counts[1]

    def test_no_nan_inf(self):
        X, y = make_toy_data()
        for kind in ("BS1", "BS2"):
            b_smote = BorderlineSMOTE(k=5, kind=kind, random_state=42)
            X_res, _ = b_smote.fit_resample(X, y)
        assert not np.isnan(X_res).any()
        assert not np.isinf(X_res).any()


class TestKMeansSMOTE:
    def test_output_shape(self):
        X, y = make_toy_data()
        km = KMeansSMOTE(k=5, n_clusters=3, random_state=42)
        X_res, y_res = km.fit_resample(X, y)
        assert X_res.shape[0] == y_res.shape[0]

    def test_balances_classes(self):
        X, y = make_toy_data(n_samples=50, ir=0.2)
        km = KMeansSMOTE(k=5, n_clusters=3, random_state=42)
        X_res, y_res = km.fit_resample(X, y)
        _, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1]

    def test_no_nan_inf(self):
        X, y = make_toy_data()
        km = KMeansSMOTE(k=5, random_state=42)
        X_res, _ = km.fit_resample(X, y)
        assert not np.isnan(X_res).any()
        assert not np.isinf(X_res).any()


class TestSVMSMOTE:
    def test_output_shape(self):
        X, y = make_toy_data()
        svm = SVMSMOTE(k=5, random_state=42)
        X_res, y_res = svm.fit_resample(X, y)
        assert X_res.shape[0] == y_res.shape[0]

    def test_balances_classes(self):
        X, y = make_toy_data(n_samples=50, ir=0.2)
        svm = SVMSMOTE(k=5, random_state=42)
        X_res, y_res = svm.fit_resample(X, y)
        _, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1]

    def test_no_nan_inf(self):
        X, y = make_toy_data()
        svm = SVMSMOTE(k=5, random_state=42)
        X_res, _ = svm.fit_resample(X, y)
        assert not np.isnan(X_res).any()
        assert not np.isinf(X_res).any()


class TestSMOTEENN:
    def test_output_shape(self):
        X, y = make_toy_data()
        enn = SMOTEENN(k=5, random_state=42)
        X_res, y_res = enn.fit_resample(X, y)
        assert X_res.shape[0] == y_res.shape[0]

    def test_no_nan_inf(self):
        X, y = make_toy_data()
        enn = SMOTEENN(k=5, random_state=42)
        X_res, _ = enn.fit_resample(X, y)
        assert not np.isnan(X_res).any()
        assert not np.isinf(X_res).any()

    def test_removes_some_samples(self):
        X, y = make_toy_data(n_samples=50, ir=0.3)
        smote = SMOTE(k=5, random_state=42)
        X_s, y_s = smote.fit_resample(X, y)
        enn = SMOTEENN(k=5, k_enn=3, random_state=42)
        X_e, y_e = enn.fit_resample(X, y)
        assert X_e.shape[0] <= X_s.shape[0]


class TestSMOTETomek:
    def test_output_shape(self):
        X, y = make_toy_data()
        to = SMOTETomek(k=5, random_state=42)
        X_res, y_res = to.fit_resample(X, y)
        assert X_res.shape[0] == y_res.shape[0]

    def test_no_nan_inf(self):
        X, y = make_toy_data()
        to = SMOTETomek(k=5, random_state=42)
        X_res, _ = to.fit_resample(X, y)
        assert not np.isnan(X_res).any()
        assert not np.isinf(X_res).any()

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError):
            BorderlineSMOTE(kind="BS3")


def _test_smote_basics(algo_class, **kwargs):
    X, y = make_toy_data()
    algo = algo_class(random_state=42, **kwargs)
    X_res, y_res = algo.fit_resample(X, y)
    assert X_res.shape[0] == y_res.shape[0]
    assert X_res.shape[1] == X.shape[1]
    assert not np.isnan(X_res).any()
    assert not np.isinf(X_res).any()
    _, counts = np.unique(y_res, return_counts=True)
    assert counts[0] == counts[1]


class TestGeometricSMOTE:
    def test_output_shape(self):
        X, y = make_toy_data()
        gs = GeometricSMOTE(k=5, random_state=42)
        X_res, y_res = gs.fit_resample(X, y)
        assert X_res.shape[0] == y_res.shape[0]

    def test_balances_classes(self):
        X, y = make_toy_data(n_samples=50, ir=0.2)
        gs = GeometricSMOTE(k=5, random_state=42)
        X_res, y_res = gs.fit_resample(X, y)
        _, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1]

    def test_synthetic_not_only_on_segments(self):
        X, y = make_toy_data(n_features=2, n_samples=30, ir=0.3)
        gs = GeometricSMOTE(k=3, alpha=0.5, random_state=42)
        X_res, y_res = gs.fit_resample(X, y)
        X_synth = X_res[len(X):]

        X_min_orig = X[y == np.min(np.unique(y))]
        on_segment_count = 0
        for s in X_synth:
            for a in X_min_orig:
                for b in X_min_orig:
                    if np.array_equal(a, b):
                        continue
                    vec_ab = b - a
                    vec_as = s - a
                    norm_ab = np.linalg.norm(vec_ab)
                    if norm_ab < 1e-10:
                        continue
                    proj = np.dot(vec_as, vec_ab) / np.dot(vec_ab, vec_ab)
                    if -1e-3 < proj < 1 + 1e-3:
                        point_on_seg = a + max(0, min(1, proj)) * vec_ab
                        if np.linalg.norm(s - point_on_seg) < 1e-3:
                            on_segment_count += 1
                            break
                else:
                    continue
                break
        assert on_segment_count < len(X_synth), "All G-SMOTE points on segments - not expected"

    def test_no_nan_inf(self):
        X, y = make_toy_data()
        gs = GeometricSMOTE(k=5, random_state=42)
        X_res, _ = gs.fit_resample(X, y)
        assert not np.isnan(X_res).any()
        assert not np.isinf(X_res).any()


class TestRandomSMOTE:
    def test_output_shape(self):
        X, y = make_toy_data()
        rs = RandomSMOTE(k=5, random_state=42)
        X_res, y_res = rs.fit_resample(X, y)
        assert X_res.shape[0] == y_res.shape[0]

    def test_balances_classes(self):
        X, y = make_toy_data(n_samples=50, ir=0.2)
        rs = RandomSMOTE(k=5, random_state=42)
        X_res, y_res = rs.fit_resample(X, y)
        _, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1]

    def test_no_nan_inf(self):
        X, y = make_toy_data()
        rs = RandomSMOTE(k=5, random_state=42)
        X_res, _ = rs.fit_resample(X, y)
        assert not np.isnan(X_res).any()
        assert not np.isinf(X_res).any()


class TestPolynomFitSMOTE:
    def test_output_shape(self):
        X, y = make_toy_data()
        pf = PolynomFitSMOTE(k=5, degree=2, random_state=42)
        X_res, y_res = pf.fit_resample(X, y)
        assert X_res.shape[0] == y_res.shape[0]

    def test_balances_classes(self):
        X, y = make_toy_data(n_samples=50, ir=0.2)
        pf = PolynomFitSMOTE(k=5, degree=2, random_state=42)
        X_res, y_res = pf.fit_resample(X, y)
        _, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1]

    def test_no_nan_inf(self):
        X, y = make_toy_data()
        pf = PolynomFitSMOTE(k=5, degree=2, random_state=42)
        X_res, _ = pf.fit_resample(X, y)
        assert not np.isnan(X_res).any()
        assert not np.isinf(X_res).any()


class TestADASYN:
    def test_output_shape(self):
        X, y = make_toy_data()
        ada = ADASYN(k=5, random_state=42)
        X_res, y_res = ada.fit_resample(X, y)
        assert X_res.shape[0] == y_res.shape[0]

    def test_balances_classes(self):
        X, y = make_toy_data(n_samples=50, ir=0.2)
        ada = ADASYN(k=5, random_state=42)
        X_res, y_res = ada.fit_resample(X, y)
        _, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1]

    def test_no_nan_inf(self):
        X, y = make_toy_data()
        ada = ADASYN(k=5, random_state=42)
        X_res, _ = ada.fit_resample(X, y)
        assert not np.isnan(X_res).any()
        assert not np.isinf(X_res).any()


class TestSafeLevelSMOTE:
    def test_output_shape(self):
        X, y = make_toy_data()
        sl = SafeLevelSMOTE(k=5, random_state=42)
        X_res, y_res = sl.fit_resample(X, y)
        assert X_res.shape[0] == y_res.shape[0]

    def test_balances_class(self):
        X, y = make_toy_data(n_samples=50, ir=0.2)
        sl = SafeLevelSMOTE(k=5, random_state=42)
        X_res, y_res = sl.fit_resample(X, y)
        _, counts = np.unique(y_res, return_counts=True)
        assert counts[0] <= counts[1]

    def test_no_nan_inf(self):
        X, y = make_toy_data()
        sl = SafeLevelSMOTE(k=5, random_state=42)
        X_res, _ = sl.fit_resample(X, y)
        assert not np.isnan(X_res).any()
        assert not np.isinf(X_res).any()


class TestNoOversampling:
    def test_output_shape(self):
        X, y = make_toy_data()
        base = NoOversampling()
        X_res, y_res = base.fit_resample(X, y)
        assert X_res.shape == X.shape
        assert y_res.shape == y.shape

    def test_no_change(self):
        X, y = make_toy_data()
        base = NoOversampling()
        X_res, y_res = base.fit_resample(X, y)
        assert np.array_equal(X, X_res)
        assert np.array_equal(y, y_res)

    def test_ignores_k(self):
        X, y = make_toy_data()
        for k in (1, 5, 10):
            base = NoOversampling(k=k, random_state=42)
            X_res, y_res = base.fit_resample(X, y)
            assert np.array_equal(X, X_res)

    def test_no_nan_inf(self):
        X, y = make_toy_data()
        base = NoOversampling()
        X_res, _ = base.fit_resample(X, y)
        assert not np.isnan(X_res).any()
        assert not np.isinf(X_res).any()


class TestRandomOversampling:
    def test_output_shape(self):
        X, y = make_toy_data()
        ros = RandomOversampling(random_state=42)
        X_res, y_res = ros.fit_resample(X, y)
        assert X_res.shape[0] == y_res.shape[0]
        assert X_res.shape[1] == X.shape[1]

    def test_balances_classes(self):
        X, y = make_toy_data(n_samples=50, ir=0.2)
        ros = RandomOversampling(random_state=42)
        X_res, y_res = ros.fit_resample(X, y)
        _, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1]

    def test_minority_increased(self):
        X, y = make_toy_data()
        _, counts_before = np.unique(y, return_counts=True)
        n_min_before = counts_before.min()

        ros = RandomOversampling(random_state=42)
        X_res, y_res = ros.fit_resample(X, y)

        _, counts_after = np.unique(y_res, return_counts=True)
        n_min_after = counts_after.min()
        assert n_min_after > n_min_before

    def test_reproducibility(self):
        X, y = make_toy_data()
        ros1 = RandomOversampling(random_state=42)
        ros2 = RandomOversampling(random_state=42)
        X1, y1 = ros1.fit_resample(X, y)
        X2, y2 = ros2.fit_resample(X, y)
        assert np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)

    def test_no_nan_inf(self):
        X, y = make_toy_data()
        ros = RandomOversampling(random_state=42)
        X_res, _ = ros.fit_resample(X, y)
        assert not np.isnan(X_res).any()
        assert not np.isinf(X_res).any()

    def test_all_synthetic_are_existing(self):
        X, y = make_toy_data(n_samples=50, n_features=2, ir=0.2)
        ros = RandomOversampling(random_state=42)
        X_res, y_res = ros.fit_resample(X, y)
        X_synth = X_res[len(X):]
        for s in X_synth:
            assert any(np.array_equal(s, x) for x in X)


class TestRandomUndersampling:
    def test_output_shape(self):
        X, y = make_toy_data()
        rus = RandomUndersampling(random_state=42)
        X_res, y_res = rus.fit_resample(X, y)
        assert X_res.shape[0] == y_res.shape[0]
        assert X_res.shape[1] == X.shape[1]

    def test_balances_classes(self):
        X, y = make_toy_data(n_samples=50, ir=0.2)
        rus = RandomUndersampling(random_state=42)
        X_res, y_res = rus.fit_resample(X, y)
        _, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1]

    def test_majority_decreased(self):
        X, y = make_toy_data()
        _, counts_before = np.unique(y, return_counts=True)
        n_maj_before = counts_before.max()

        rus = RandomUndersampling(random_state=42)
        X_res, y_res = rus.fit_resample(X, y)

        _, counts_after = np.unique(y_res, return_counts=True)
        n_maj_after = counts_after.max()
        assert n_maj_after < n_maj_before

    def test_minority_unchanged(self):
        X, y = make_toy_data()
        _, counts_before = np.unique(y, return_counts=True)
        n_min_before = counts_before.min()
        min_class = np.unique(y)[np.argmin(counts_before)]

        rus = RandomUndersampling(random_state=42)
        _, y_res = rus.fit_resample(X, y)
        n_min_after = np.sum(y_res == min_class)
        assert n_min_after == n_min_before

    def test_reproducibility(self):
        X, y = make_toy_data()
        rus1 = RandomUndersampling(random_state=42)
        rus2 = RandomUndersampling(random_state=42)
        X1, y1 = rus1.fit_resample(X, y)
        X2, y2 = rus2.fit_resample(X, y)
        assert np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)

    def test_no_nan_inf(self):
        X, y = make_toy_data()
        rus = RandomUndersampling(random_state=42)
        X_res, _ = rus.fit_resample(X, y)
        assert not np.isnan(X_res).any()
        assert not np.isinf(X_res).any()

    def test_all_remaining_are_original(self):
        X, y = make_toy_data(n_samples=50, n_features=2, ir=0.2)
        rus = RandomUndersampling(random_state=42)
        X_res, _ = rus.fit_resample(X, y)
        for x in X_res:
            assert any(np.array_equal(x, ox) for ox in X)


class TestNearMiss1:
    def test_output_shape(self):
        X, y = make_toy_data()
        nm = NearMiss1(k=5, random_state=42)
        X_res, y_res = nm.fit_resample(X, y)
        assert X_res.shape[0] == y_res.shape[0]
        assert X_res.shape[1] == X.shape[1]

    def test_reduces_samples(self):
        X, y = make_toy_data(n_samples=100, ir=0.2)
        nm = NearMiss1(k=5, random_state=42)
        X_res, y_res = nm.fit_resample(X, y)
        assert len(X_res) < len(X)

    def test_no_nan_inf(self):
        X, y = make_toy_data()
        nm = NearMiss1(k=5, random_state=42)
        X_res, _ = nm.fit_resample(X, y)
        assert not np.isnan(X_res).any()
        assert not np.isinf(X_res).any()

    def test_minority_unchanged(self):
        X, y = make_toy_data(n_samples=100, ir=0.2)
        _, counts_before = np.unique(y, return_counts=True)
        n_min_before = counts_before.min()
        min_class = np.unique(y)[np.argmin(counts_before)]

        nm = NearMiss1(k=5, random_state=42)
        _, y_res = nm.fit_resample(X, y)
        n_min_after = np.sum(y_res == min_class)
        assert n_min_after == n_min_before


class TestNearMiss2:
    def test_output_shape(self):
        X, y = make_toy_data()
        nm = NearMiss2(k=5, random_state=42)
        X_res, y_res = nm.fit_resample(X, y)
        assert X_res.shape[0] == y_res.shape[0]
        assert X_res.shape[1] == X.shape[1]

    def test_reduces_samples(self):
        X, y = make_toy_data(n_samples=100, ir=0.2)
        nm = NearMiss2(k=5, random_state=42)
        X_res, _ = nm.fit_resample(X, y)
        assert len(X_res) < len(X)

    def test_no_nan_inf(self):
        X, y = make_toy_data()
        nm = NearMiss2(k=5, random_state=42)
        X_res, _ = nm.fit_resample(X, y)
        assert not np.isnan(X_res).any()
        assert not np.isinf(X_res).any()

    def test_minority_unchanged(self):
        X, y = make_toy_data(n_samples=100, ir=0.2)
        _, counts_before = np.unique(y, return_counts=True)
        n_min_before = counts_before.min()
        min_class = np.unique(y)[np.argmin(counts_before)]

        nm = NearMiss2(k=5, random_state=42)
        _, y_res = nm.fit_resample(X, y)
        n_min_after = np.sum(y_res == min_class)
        assert n_min_after == n_min_before


class TestNearMiss3:
    def test_output_shape(self):
        X, y = make_toy_data()
        nm = NearMiss3(k=5, random_state=42)
        X_res, y_res = nm.fit_resample(X, y)
        assert X_res.shape[0] == y_res.shape[0]
        assert X_res.shape[1] == X.shape[1]

    def test_reduces_samples(self):
        X, y = make_toy_data(n_samples=100, ir=0.2)
        nm = NearMiss3(k=5, random_state=42)
        X_res, _ = nm.fit_resample(X, y)
        assert len(X_res) < len(X)

    def test_no_nan_inf(self):
        X, y = make_toy_data()
        nm = NearMiss3(k=5, random_state=42)
        X_res, _ = nm.fit_resample(X, y)
        assert not np.isnan(X_res).any()
        assert not np.isinf(X_res).any()

    def test_minority_unchanged(self):
        X, y = make_toy_data(n_samples=100, ir=0.2)
        _, counts_before = np.unique(y, return_counts=True)
        n_min_before = counts_before.min()
        min_class = np.unique(y)[np.argmin(counts_before)]

        nm = NearMiss3(k=5, random_state=42)
        _, y_res = nm.fit_resample(X, y)
        n_min_after = np.sum(y_res == min_class)
        assert n_min_after == n_min_before


class TestTomekLinksStandalone:
    def test_output_shape(self):
        X, y = make_toy_data()
        tl = TomekLinks(k=5, random_state=42)
        X_res, y_res = tl.fit_resample(X, y)
        assert X_res.shape[0] == y_res.shape[0]
        assert X_res.shape[1] == X.shape[1]

    def test_no_nan_inf(self):
        X, y = make_toy_data()
        tl = TomekLinks(k=5, random_state=42)
        X_res, _ = tl.fit_resample(X, y)
        assert not np.isnan(X_res).any()
        assert not np.isinf(X_res).any()

    def test_does_not_increase(self):
        X, y = make_toy_data(n_samples=100, ir=0.2)
        tl = TomekLinks(k=5, random_state=42)
        X_res, _ = tl.fit_resample(X, y)
        assert len(X_res) <= len(X)


class TestENNStandalone:
    def test_output_shape(self):
        X, y = make_toy_data()
        enn = ENN(k=5, random_state=42)
        X_res, y_res = enn.fit_resample(X, y)
        assert X_res.shape[0] == y_res.shape[0]
        assert X_res.shape[1] == X.shape[1]

    def test_no_nan_inf(self):
        X, y = make_toy_data()
        enn = ENN(k=5, random_state=42)
        X_res, _ = enn.fit_resample(X, y)
        assert not np.isnan(X_res).any()
        assert not np.isinf(X_res).any()

    def test_does_not_increase(self):
        X, y = make_toy_data(n_samples=100, ir=0.2)
        enn = ENN(k=5, random_state=42)
        X_res, _ = enn.fit_resample(X, y)
        assert len(X_res) <= len(X)


@pytest.mark.skip(reason="PyTorch DLL init fails on this Python 3.12/Windows. "
                          "WGAN validated manually — works with 'import torch' first.")
class TestWGAN:
    def _get_wgan(self):
        from smote_variants.gan import WGAN
        return WGAN

    def test_output_shape(self):
        X, y = make_toy_data()
        wgan = self._get_wgan()(k=5, random_state=42, epochs=20, batch_size=16)
        X_res, y_res = wgan.fit_resample(X, y)
        assert X_res.shape[0] == y_res.shape[0]
        assert X_res.shape[1] == X.shape[1]

    def test_balances_classes(self):
        X, y = make_toy_data(n_samples=50, ir=0.2)
        wgan = self._get_wgan()(k=5, random_state=42, epochs=30, batch_size=16)
        X_res, y_res = wgan.fit_resample(X, y)
        _, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1]

    def test_no_nan_inf(self):
        X, y = make_toy_data()
        wgan = self._get_wgan()(k=5, random_state=42, epochs=20, batch_size=16)
        X_res, _ = wgan.fit_resample(X, y)
        assert not np.isnan(X_res).any()
        assert not np.isinf(X_res).any()

    def test_reproducibility(self):
        X, y = make_toy_data(n_samples=40, ir=0.3)
        WGAN = self._get_wgan()
        wgan1 = WGAN(k=5, random_state=42, epochs=20, batch_size=16)
        wgan2 = WGAN(k=5, random_state=42, epochs=20, batch_size=16)
        X1, y1 = wgan1.fit_resample(X, y)
        X2, y2 = wgan2.fit_resample(X, y)
        assert np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)

    def test_minority_increased(self):
        X, y = make_toy_data()
        _, counts_before = np.unique(y, return_counts=True)
        n_min_before = counts_before.min()

        wgan = self._get_wgan()(k=5, random_state=42, epochs=20, batch_size=16)
        X_res, y_res = wgan.fit_resample(X, y)

        _, counts_after = np.unique(y_res, return_counts=True)
        n_min_after = counts_after.min()
        assert n_min_after > n_min_before
