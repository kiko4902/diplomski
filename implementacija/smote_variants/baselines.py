import numpy as np


class NoOversampling:
    def __init__(self, k=5, random_state=None):
        pass

    def fit_resample(self, X, y):
        return np.asarray(X, dtype=np.float64).copy(), np.asarray(y).copy()


class RandomOversampling:
    def __init__(self, k=5, random_state=None):
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
        minority_count = counts.min()
        majority_count = counts.max()

        n_to_generate = majority_count - minority_count

        X_min = X[y == minority_class]

        indices = self.random_state.randint(0, minority_count, n_to_generate)
        X_synth = X_min[indices]
        y_synth = np.full(n_to_generate, minority_class)

        X_resampled = np.vstack([X, X_synth])
        y_resampled = np.hstack([y, y_synth])

        return X_resampled, y_resampled


class RandomUndersampling:
    def __init__(self, k=5, random_state=None):
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
        minority_count = counts.min()

        X_maj = X[y == majority_class]
        X_min = X[y == minority_class]
        y_maj = y[y == majority_class]
        y_min = y[y == minority_class]

        indices = self.random_state.choice(len(X_maj), minority_count, replace=False)
        X_maj_under = X_maj[indices]
        y_maj_under = y_maj[indices]

        X_resampled = np.vstack([X_min, X_maj_under])
        y_resampled = np.hstack([y_min, y_maj_under])

        return X_resampled, y_resampled
