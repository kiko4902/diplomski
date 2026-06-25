import numpy as np
from sklearn.neighbors import NearestNeighbors


class NearMiss1:
    def __init__(self, k=5, random_state=None):
        self.k = k
        self.random_state = (
            np.random.RandomState(random_state)
            if random_state is not None
            else np.random.RandomState()
        )

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]
        n_min = counts.min()

        X_min = X[y == minority_class]
        X_maj = X[y == majority_class]

        n_to_keep = n_min
        if n_to_keep >= len(X_maj):
            return X, y

        nn = NearestNeighbors(n_neighbors=min(self.k, len(X_min)))
        nn.fit(X_min)
        distances, _ = nn.kneighbors(X_maj)
        avg_dist = distances.mean(axis=1)

        selected_idx = np.argsort(avg_dist)[:n_to_keep]
        X_maj_selected = X_maj[selected_idx]

        X_resampled = np.vstack([X_min, X_maj_selected])
        y_resampled = np.hstack([
            np.full(n_min, minority_class),
            np.full(n_to_keep, majority_class),
        ])
        return X_resampled, y_resampled


class NearMiss2:
    def __init__(self, k=5, random_state=None):
        self.k = k
        self.random_state = (
            np.random.RandomState(random_state)
            if random_state is not None
            else np.random.RandomState()
        )

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]
        n_min = counts.min()

        X_min = X[y == minority_class]
        X_maj = X[y == majority_class]

        n_to_keep = n_min
        if n_to_keep >= len(X_maj):
            return X, y

        nn = NearestNeighbors(n_neighbors=len(X_min))
        nn.fit(X_min)
        distances, _ = nn.kneighbors(X_maj)
        farthest = distances[:, -min(self.k, len(X_min)):]
        avg_dist = farthest.mean(axis=1)

        selected_idx = np.argsort(avg_dist)[:n_to_keep]
        X_maj_selected = X_maj[selected_idx]

        X_resampled = np.vstack([X_min, X_maj_selected])
        y_resampled = np.hstack([
            np.full(n_min, minority_class),
            np.full(n_to_keep, majority_class),
        ])
        return X_resampled, y_resampled


class NearMiss3:
    def __init__(self, k=5, random_state=None):
        self.k = k
        self.random_state = (
            np.random.RandomState(random_state)
            if random_state is not None
            else np.random.RandomState()
        )

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]
        n_min = counts.min()

        X_min = X[y == minority_class]
        X_maj = X[y == majority_class]

        n_neighbors = min(self.k, len(X_maj))

        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(X_maj)
        _, indices = nn.kneighbors(X_min)

        selected = set()
        for idx_row in indices:
            for idx in idx_row:
                selected.add(idx)
                if len(selected) >= n_min:
                    break
            if len(selected) >= n_min:
                break

        if len(selected) < n_min:
            remaining = set(range(len(X_maj))) - selected
            extra = list(remaining)[:n_min - len(selected)]
            selected.update(extra)

        selected_idx = list(selected)[:n_min]
        X_maj_selected = X_maj[selected_idx]

        X_resampled = np.vstack([X_min, X_maj_selected])
        y_resampled = np.hstack([
            np.full(n_min, minority_class),
            np.full(len(selected_idx), majority_class),
        ])
        return X_resampled, y_resampled


class TomekLinks:
    def __init__(self, k=5, random_state=None):
        self.k = k
        self.random_state = (
            np.random.RandomState(random_state)
            if random_state is not None
            else np.random.RandomState()
        )

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        n = len(X)

        if n < 2:
            return X, y

        keep_mask = np.ones(n, dtype=bool)
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(X)
        _, indices = nn.kneighbors(X)

        for i in range(n):
            if not keep_mask[i]:
                continue
            j = indices[i, 1]
            if y[i] != y[j] and keep_mask[j]:
                if indices[j, 0] == i or (indices.shape[1] > 1 and indices[j, 1] == i):
                    if y[i] == minority_class:
                        keep_mask[j] = False
                    elif y[j] == minority_class:
                        keep_mask[i] = False
                    else:
                        keep_mask[i] = False
                        keep_mask[j] = False

        if keep_mask.sum() < 2:
            return X, y

        return X[keep_mask], y[keep_mask]


class ENN:
    def __init__(self, k=5, random_state=None):
        self.k = k
        self.random_state = (
            np.random.RandomState(random_state)
            if random_state is not None
            else np.random.RandomState()
        )

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        n = len(X)
        keep_mask = np.ones(n, dtype=bool)
        n_neighbors = min(self.k + 1, n)

        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(X)
        _, indices = nn.kneighbors(X)

        for i in range(n):
            neighbor_labels = y[indices[i, 1:]]
            if np.sum(neighbor_labels != y[i]) >= len(neighbor_labels) / 2:
                keep_mask[i] = False

        if keep_mask.sum() < 2:
            return X, y

        return X[keep_mask], y[keep_mask]
