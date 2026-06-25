import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import BaseSMOTE
from .smote import SMOTE


class SMOTEENN(BaseSMOTE):
    def __init__(self, k=5, k_enn=3, random_state=None):
        super().__init__(k=k, random_state=random_state)
        self.k_enn = k_enn

    def fit_resample(self, X, y):
        X, y, min_cls, maj_cls, n_min, n_maj = self._validate_input(X, y)

        smote = SMOTE(k=self.k, random_state=self.random_state.randint(0, 2**31 - 1))
        X_res, y_res = smote.fit_resample(X, y)

        remove_mask = np.ones(len(X_res), dtype=bool)
        nn = NearestNeighbors(n_neighbors=min(self.k_enn + 1, len(X_res)))
        nn.fit(X_res)
        _, indices = nn.kneighbors(X_res)

        for i in range(len(X_res)):
            neighbor_labels = y_res[indices[i, 1:]]
            if np.sum(neighbor_labels != y_res[i]) >= self.k_enn / 2:
                remove_mask[i] = False

        return X_res[remove_mask], y_res[remove_mask]


class SMOTETomek(BaseSMOTE):
    def __init__(self, k=5, random_state=None):
        super().__init__(k=k, random_state=random_state)

    def fit_resample(self, X, y):
        X, y, min_cls, maj_cls, n_min, n_maj = self._validate_input(X, y)

        smote = SMOTE(k=self.k, random_state=self.random_state.randint(0, 2**31 - 1))
        X_res, y_res = smote.fit_resample(X, y)

        remove_mask = np.ones(len(X_res), dtype=bool)
        n = len(X_res)

        if n < 2:
            return X_res, y_res

        nn = NearestNeighbors(n_neighbors=min(2, n))
        nn.fit(X_res)
        distances, indices = nn.kneighbors(X_res)

        for i in range(n):
            if not remove_mask[i]:
                continue
            j = indices[i, 1] if indices.shape[1] > 1 else indices[i, 0]
            if j < n and remove_mask[j] and y_res[i] != y_res[j]:
                if np.array_equal(indices[j, 0], i) or (
                    indices.shape[1] > 1 and np.array_equal(indices[j, 1], i)
                ):
                    remove_mask[i] = False
                    remove_mask[j] = False

        return X_res[remove_mask], y_res[remove_mask]
