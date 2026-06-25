import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from .base import BaseSMOTE
from .smote import SMOTE


class KMeansSMOTE(BaseSMOTE):
    def __init__(self, k=5, n_clusters=5, random_state=None):
        super().__init__(k=k, random_state=random_state)
        self.n_clusters = n_clusters

    def fit_resample(self, X, y):
        X, y, min_cls, maj_cls, n_min, n_maj = self._validate_input(X, y)
        X_min = X[y == min_cls]
        n_synthetic = n_maj - n_min

        if n_synthetic <= 0:
            return X.copy(), y.copy()

        n_clusters_actual = min(self.n_clusters, n_min)
        kmeans = KMeans(n_clusters=n_clusters_actual, random_state=self.random_state.randint(0, 2**31 - 1), n_init=10)
        cluster_labels = kmeans.fit_predict(X_min)

        cluster_sizes = np.bincount(cluster_labels)
        cluster_weights = np.zeros(n_clusters_actual, dtype=np.float64)
        for c in range(n_clusters_actual):
            if cluster_sizes[c] > 0:
                cluster_weights[c] = 1.0 / cluster_sizes[c]

        total_weight = cluster_weights.sum()
        if total_weight == 0:
            cluster_weights = np.ones(n_clusters_actual)
            total_weight = n_clusters_actual

        cluster_weights /= total_weight
        n_per_cluster = np.round(cluster_weights * n_synthetic).astype(int)

        diff = n_synthetic - n_per_cluster.sum()
        for _ in range(abs(diff)):
            idx = self.random_state.randint(0, n_clusters_actual)
            if diff > 0:
                n_per_cluster[idx] += 1
            else:
                n_per_cluster[idx] = max(0, n_per_cluster[idx] - 1)

        synthetic = np.zeros((n_synthetic, X.shape[1]), dtype=np.float64)
        pos = 0

        for c in range(n_clusters_actual):
            if n_per_cluster[c] == 0:
                continue
            mask = cluster_labels == c
            X_cluster = X_min[mask]
            n_cluster = len(X_cluster)

            if n_cluster < 2:
                for _ in range(n_per_cluster[c]):
                    idx = self.random_state.randint(0, n_cluster)
                    noise = self.random_state.randn(X.shape[1]) * 0.001
                    synthetic[pos] = X_cluster[idx] + noise
                    pos += 1
                continue

            nn = NearestNeighbors(n_neighbors=min(self.k + 1, n_cluster))
            nn.fit(X_cluster)
            _, indices = nn.kneighbors(X_cluster)
            indices = indices[:, 1:]

            for _ in range(n_per_cluster[c]):
                idx = self.random_state.randint(0, n_cluster)
                nn_idx = self.random_state.choice(indices[idx])
                lam = self.random_state.uniform(0, 1)
                synthetic[pos] = X_cluster[idx] + lam * (X_cluster[nn_idx] - X_cluster[idx])
                pos += 1

        X_resampled = np.vstack([X, synthetic])
        y_resampled = np.hstack([y, np.full(n_synthetic, min_cls)])

        return X_resampled, y_resampled
