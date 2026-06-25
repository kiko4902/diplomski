import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors

from .base import BaseSMOTE


class SVMSMOTE(BaseSMOTE):
    def __init__(self, k=5, svm_params=None, random_state=None):
        super().__init__(k=k, random_state=random_state)
        self.svm_params = svm_params or {"kernel": "rbf", "probability": False}

    def fit_resample(self, X, y):
        X, y, min_cls, maj_cls, n_min, n_maj = self._validate_input(X, y)
        X_min = X[y == min_cls]
        n_synthetic = n_maj - n_min

        if n_synthetic <= 0:
            return X.copy(), y.copy()

        svm = SVC(**self.svm_params, random_state=self.random_state.randint(0, 2**31 - 1))
        svm.fit(X, y)

        sv_indices = svm.support_
        sv_labels = y[sv_indices]
        sv_min_mask = sv_labels == min_cls

        if not sv_min_mask.any():
            sv_min_mask = np.ones(n_min, dtype=bool)
            X_sv_min = X_min
        else:
            X_sv_min = X[sv_indices][sv_min_mask]

        n_sv = len(X_sv_min)

        synthetic = np.zeros((n_synthetic, X.shape[1]), dtype=np.float64)

        if n_sv < 2:
            noise = self.random_state.randn(n_synthetic, X.shape[1]) * 0.01
            idx = self.random_state.randint(0, n_sv, size=n_synthetic)
            synthetic = X_sv_min[idx] + noise
        else:
            nn = NearestNeighbors(n_neighbors=min(self.k + 1, n_sv))
            nn.fit(X_sv_min)
            _, indices = nn.kneighbors(X_sv_min)
            indices = indices[:, 1:]

            for i in range(n_synthetic):
                idx = self.random_state.randint(0, n_sv)
                if len(indices[idx]) > 0:
                    nn_idx = self.random_state.choice(indices[idx])
                    lam = self.random_state.uniform(0, 1)
                    synthetic[i] = X_sv_min[idx] + lam * (X_sv_min[nn_idx] - X_sv_min[idx])
                else:
                    synthetic[i] = X_sv_min[idx]

        X_resampled = np.vstack([X, synthetic])
        y_resampled = np.hstack([y, np.full(n_synthetic, min_cls)])

        return X_resampled, y_resampled
