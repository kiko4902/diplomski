import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import BaseSMOTE


class SafeLevelSMOTE(BaseSMOTE):
    def __init__(self, k=5, random_state=None):
        super().__init__(k=k, random_state=random_state)

    def fit_resample(self, X, y):
        X, y, min_cls, maj_cls, n_min, n_maj = self._validate_input(X, y)

        X_min = X[y == min_cls]
        n_synthetic = n_maj - n_min

        if n_synthetic <= 0:
            return X.copy(), y.copy()

        min_nn = NearestNeighbors(n_neighbors=min(self.k + 1, n_min))
        min_nn.fit(X_min)
        _, min_indices = min_nn.kneighbors(X_min)
        min_indices = min_indices[:, 1:]

        sl = np.array([len(indices) for indices in min_indices], dtype=int)

        synthetic = np.zeros((n_synthetic, X.shape[1]), dtype=np.float64)
        generated = 0
        max_attempts = n_synthetic * 10

        for _ in range(max_attempts):
            if generated >= n_synthetic:
                break
            i = self.random_state.randint(0, n_min)
            j = self.random_state.choice(min_indices[i])
            sl_p, sl_n = sl[i], sl[j]

            if sl_p == 0 and sl_n == 0:
                continue
            elif sl_p == 0:
                s_new = X_min[j].copy()
            elif sl_n == 0:
                s_new = X_min[i].copy()
            else:
                ratio = float(sl_p) / float(sl_n)
                if ratio <= 1:
                    lam = self.random_state.uniform(0, ratio)
                else:
                    lam = self.random_state.uniform(1 - 1.0 / ratio, 1)
                s_new = X_min[i] + lam * (X_min[j] - X_min[i])

            synthetic[generated] = s_new
            generated += 1

        if generated < n_synthetic:
            synthetic = synthetic[:generated]
            if generated == 0:
                return X.copy(), y.copy()

        X_resampled = np.vstack([X, synthetic])
        y_resampled = np.hstack([y, np.full(generated, min_cls)])

        return X_resampled, y_resampled
