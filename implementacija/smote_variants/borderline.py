import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import BaseSMOTE


class BorderlineSMOTE(BaseSMOTE):
    def __init__(self, k=5, m=10, kind="BS1", random_state=None):
        super().__init__(k=k, random_state=random_state)
        if kind not in ("BS1", "BS2"):
            raise ValueError("kind must be 'BS1' or 'BS2'")
        self.m = m
        self.kind = kind

    def fit_resample(self, X, y):
        X, y, min_cls, maj_cls, n_min, n_maj = self._validate_input(X, y)

        X_min = X[y == min_cls]
        n_synthetic = n_maj - n_min

        if n_synthetic <= 0:
            return X.copy(), y.copy()

        full_nn = NearestNeighbors(n_neighbors=min(self.m, len(X)))
        full_nn.fit(X)
        _, full_indices = full_nn.kneighbors(X_min)

        danger_mask = np.zeros(n_min, dtype=bool)
        for i in range(n_min):
            n_maj_neighbors = np.sum(y[full_indices[i]] == maj_cls)
            if self.m / 2 <= n_maj_neighbors < self.m:
                danger_mask[i] = True

        n_danger = danger_mask.sum()
        if n_danger == 0:
            danger_mask[:] = True
            n_danger = n_min

        synthetic = np.zeros((n_synthetic, X.shape[1]), dtype=np.float64)

        if self.kind == "BS1":
            min_nn = NearestNeighbors(n_neighbors=min(self.k + 1, n_min))
            min_nn.fit(X_min)
            _, min_indices = min_nn.kneighbors(X_min)
            min_indices = min_indices[:, 1:]

            for i in range(n_synthetic):
                danger_idx = self.random_state.choice(np.where(danger_mask)[0])
                nn_idx = self.random_state.choice(min_indices[danger_idx])
                x_nn = X_min[nn_idx]
                lam = self.random_state.uniform(0, 1)
                synthetic[i] = X_min[danger_idx] + lam * (x_nn - X_min[danger_idx])
        else:
            all_nn = NearestNeighbors(n_neighbors=min(self.k, len(X)))
            all_nn.fit(X)
            _, all_indices = all_nn.kneighbors(X_min)

            for i in range(n_synthetic):
                danger_idx = self.random_state.choice(np.where(danger_mask)[0])
                nn_idx = self.random_state.choice(all_indices[danger_idx])
                x_nn = X[nn_idx]
                lam = self.random_state.uniform(0, 0.5)
                synthetic[i] = X_min[danger_idx] + lam * (x_nn - X_min[danger_idx])

        X_resampled = np.vstack([X, synthetic])
        y_resampled = np.hstack([y, np.full(n_synthetic, min_cls)])

        return X_resampled, y_resampled
