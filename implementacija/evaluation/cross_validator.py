import numpy as np
from sklearn.model_selection import StratifiedKFold

from config import CV_FOLDS, CV_REPEATS, CV_SEED_START


def stratified_repeated_cv(X, y, n_splits=CV_FOLDS, n_repeats=CV_REPEATS, seed_start=CV_SEED_START):
    seeds = range(seed_start, seed_start + n_repeats)
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(X, y):
            yield X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def get_iteration_count():
    return CV_FOLDS * CV_REPEATS
