import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import BaseSMOTE


class ADASYN(BaseSMOTE):
    def __init__(self, k=5, random_state=None):
        super().__init__(k=k, random_state=random_state)

    def fit_resample(self, X, y):
        X, y, min_cls, maj_cls, n_min, n_maj = self._validate_input(X, y)

        X_min = X[y == min_cls]
        n_synthetic = n_maj - n_min

        if n_synthetic <= 0:
            return X.copy(), y.copy()

        nn = NearestNeighbors(n_neighbors=min(self.k, len(X)))
        nn.fit(X)
        _, indices = nn.kneighbors(X_min)

        gamma = np.zeros(n_min, dtype=np.float64)
        for i in range(n_min):
            gamma[i] = np.sum(y[indices[i]] == maj_cls) / self.k

        total_gamma = gamma.sum()
        if total_gamma == 0:
            gamma = np.ones(n_min)
            total_gamma = n_min

        gamma_norm = gamma / total_gamma
        n_per_point = np.round(gamma_norm * n_synthetic).astype(int)

        diff = n_synthetic - n_per_point.sum()
        if diff > 0:
            remaining = self.random_state.choice(n_min, size=diff, replace=diff > n_min)
            n_per_point[remaining] += 1
        elif diff < 0:
            while n_per_point.sum() > n_synthetic:
                positives = np.where(n_per_point > 0)[0]
                idx = self.random_state.choice(positives)
                n_per_point[idx] -= 1

        min_nn = NearestNeighbors(n_neighbors=min(self.k + 1, n_min))
        min_nn.fit(X_min)
        _, min_indices = min_nn.kneighbors(X_min)
        min_indices = min_indices[:, 1:]

        total = n_per_point.sum()
        synthetic = np.zeros((total, X.shape[1]), dtype=np.float64)
        pos = 0

        for i in range(n_min):
            for _ in range(n_per_point[i]):
                nn_idx = self.random_state.choice(min_indices[i])
                lam = self.random_state.uniform(0, 1)
                synthetic[pos] = X_min[i] + lam * (X_min[nn_idx] - X_min[i])
                pos += 1

        X_resampled = np.vstack([X, synthetic])
        y_resampled = np.hstack([y, np.full(total, min_cls)])

        return X_resampled, y_resampled
