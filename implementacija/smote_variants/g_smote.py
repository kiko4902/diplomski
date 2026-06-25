import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import BaseSMOTE


class GeometricSMOTE(BaseSMOTE):
    def __init__(self, k=5, alpha=0.5, truncation_factor=1.0, random_state=None):
        super().__init__(k=k, random_state=random_state)
        self.alpha = alpha
        self.truncation_factor = truncation_factor

    def fit_resample(self, X, y):
        X, y, min_cls, maj_cls, n_min, n_maj = self._validate_input(X, y)
        X_min = X[y == min_cls]
        n_synthetic = n_maj - n_min

        if n_synthetic <= 0:
            return X.copy(), y.copy()

        nn = NearestNeighbors(n_neighbors=min(self.k + 1, n_min))
        nn.fit(X_min)
        _, indices = nn.kneighbors(X_min)
        indices = indices[:, 1:]

        synthetic = np.zeros((n_synthetic, X.shape[1]), dtype=np.float64)

        for i in range(n_synthetic):
            idx = self.random_state.randint(0, n_min)
            nn_idx = self.random_state.choice(indices[idx])
            center = X_min[idx]
            direction = X_min[nn_idx] - center

            t = self.random_state.uniform(0, self.truncation_factor)
            on_line = center + t * direction

            n_dim = X.shape[1]
            basis = np.random.randn(n_dim)
            basis -= np.dot(basis, direction) * direction / (np.dot(direction, direction) + 1e-12)
            norm = np.linalg.norm(basis)
            if norm > 1e-12:
                basis /= norm

            radius = self.alpha * np.linalg.norm(direction) * t
            synthetic[i] = on_line + self.random_state.uniform(-radius, radius) * basis

        X_resampled = np.vstack([X, synthetic])
        y_resampled = np.hstack([y, np.full(n_synthetic, min_cls)])

        return X_resampled, y_resampled
