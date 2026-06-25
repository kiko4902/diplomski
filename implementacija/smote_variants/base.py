from abc import ABC, abstractmethod
import numpy as np


class BaseSMOTE(ABC):
    def __init__(self, k=5, random_state=None):
        self.k = k
        if random_state is not None:
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = np.random.RandomState()

    @abstractmethod
    def fit_resample(self, X, y):
        pass

    def _validate_input(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got {y.ndim}D")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("X contains NaN or Inf")

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError(f"Binary classification required, got {len(classes)} classes")

        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]
        minority_count = counts.min()
        majority_count = counts.max()

        if minority_count < self.k:
            raise ValueError(
                f"Minority class has {minority_count} samples, need at least k={self.k}"
            )

        return X, y, minority_class, majority_class, minority_count, majority_count
