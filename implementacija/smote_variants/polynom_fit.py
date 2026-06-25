import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import BaseSMOTE


class PolynomFitSMOTE(BaseSMOTE):
    def __init__(self, k=5, degree=2, random_state=None):
        super().__init__(k=k, random_state=random_state)
        self.degree = min(degree, 3)

    def fit_resample(self, X, y):
        X, y, min_cls, maj_cls, n_min, n_maj = self._validate_input(X, y)
        X_min = X[y == min_cls]
        n_synthetic = n_maj - n_min

        if n_synthetic <= 0:
            return X.copy(), y.copy()

        nn = NearestNeighbors(n_neighbors=min(self.k + 1 + self.degree, n_min))
        nn.fit(X_min)
        _, indices = nn.kneighbors(X_min)

        synthetic = np.zeros((n_synthetic, X.shape[1]), dtype=np.float64)

        for i in range(n_synthetic):
            idx = self.random_state.randint(0, n_min)
            n_points = min(self.degree + 1, n_min)
            point_indices = self.random_state.choice(indices[idx, 1:], size=n_points - 1, replace=False)
            all_indices = np.concatenate([[idx], point_indices])
            points = X_min[all_indices]

            t = self.random_state.uniform(0, 1)
            t_powers = np.array([t**d for d in range(self.degree + 1)])
            try:
                A = np.vander(np.linspace(0, 1, n_points), self.degree + 1, increasing=True)
                coeffs = np.linalg.lstsq(A, points, rcond=None)[0]
                synthetic[i] = t_powers @ coeffs
            except np.linalg.LinAlgError:
                synthetic[i] = points[0] + t * (points[-1] - points[0])

        X_resampled = np.vstack([X, synthetic])
        y_resampled = np.hstack([y, np.full(n_synthetic, min_cls)])

        return X_resampled, y_resampled
