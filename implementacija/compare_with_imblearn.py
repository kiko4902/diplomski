"""Uspoređuje naše SMOTE implementacije s imbalanced-learn referentnim."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

from smote_variants.smote import SMOTE as OurSMOTE
from smote_variants.borderline import BorderlineSMOTE as OurBorderline
from smote_variants.adasyn import ADASYN as OurADASYN
from smote_variants.svm_smote import SVMSMOTE as OurSVMSMOTE
from smote_variants.kmeans_smote import KMeansSMOTE as OurKMeans
from smote_variants.smote_enn import SMOTEENN as OurENN, SMOTETomek as OurTomek

from imblearn.over_sampling import (
    SMOTE as ImbSMOTE,
    BorderlineSMOTE as ImbBorderline,
    ADASYN as ImbADASYN,
    SVMSMOTE as ImbSVMSMOTE,
    KMeansSMOTE as ImbKMeans,
)
from imblearn.combine import SMOTEENN as ImbENN, SMOTETomek as ImbTomek
from imblearn.under_sampling import EditedNearestNeighbours


def make_data(n_samples=100, ir=0.2, random_state=42):
    X, y = make_classification(
        n_samples=n_samples, n_features=5, n_redundant=0,
        n_clusters_per_class=1, weights=[ir, 1 - ir], random_state=random_state,
    )
    return X, y


def compare(name, our_cls, imb_cls, our_kwargs=None, imb_kwargs=None, imb_neighbors_key="k_neighbors"):
    our_kwargs = our_kwargs or {}
    imb_kwargs = imb_kwargs or {}

    X, y = make_data()
    our = our_cls(k=5, random_state=42, **our_kwargs)
    X_our, y_our = our.fit_resample(X, y)

    imb = imb_cls(**{imb_neighbors_key: 5}, random_state=42, **imb_kwargs)
    X_imb, y_imb = imb.fit_resample(X, y)

    _, c_our = np.unique(y_our, return_counts=True)
    _, c_imb = np.unique(y_imb, return_counts=True)
    shape_ok = X_our.shape[1] == X_imb.shape[1]
    bal_our = int(c_our[0] == c_our[1])
    bal_imb = int(c_imb[0] == c_imb[1])

    extra = len(X_our) - len(X)
    e_imb = len(X_imb) - len(X)
    syn_diff = abs(extra - e_imb)

    syn_our = X_our[len(X):]
    syn_imb = X_imb[len(X):]
    min_len = min(len(syn_our), len(syn_imb))
    if min_len > 0:
        nn = NearestNeighbors(n_neighbors=1).fit(syn_our[:min_len])
        dist = nn.kneighbors(syn_imb[:min_len])[0].mean()
    else:
        dist = float("nan")

    print(f"{name:20s} | Shape: {shape_ok} | Bal(our/imb): {bal_our}/{bal_imb} | SynDiff: {syn_diff:4d} | AvgDist: {dist:.4f}")


if __name__ == "__main__":
    print(f"{'Algorithm':20s} | Shape | Bal     | Syn# | AvgDist")
    print("-" * 72)

    compare("1. SMOTE", OurSMOTE, ImbSMOTE)
    compare("2. Borderline BS1", OurBorderline, ImbBorderline,
            our_kwargs={"kind": "BS1"}, imb_kwargs={"kind": "borderline-1"})
    compare("3. Borderline BS2", OurBorderline, ImbBorderline,
            our_kwargs={"kind": "BS2"}, imb_kwargs={"kind": "borderline-2"})
    compare("4. ADASYN", OurADASYN, ImbADASYN,
            imb_neighbors_key="n_neighbors", imb_kwargs={"sampling_strategy": "auto"})
    compare("5. SVM-SMOTE", OurSVMSMOTE, ImbSVMSMOTE)
    compare("6. KMeans-SMOTE", OurKMeans, ImbKMeans,
            our_kwargs={"n_clusters": 2},
            imb_kwargs={
                "kmeans_estimator": KMeans(n_clusters=2, n_init=10, random_state=42),
                "cluster_balance_threshold": 0.0,
            })

    X, y = make_data()
    imb_smote = ImbSMOTE(k_neighbors=5, random_state=42)
    our7 = OurENN(k=5, k_enn=3, random_state=42)
    imb7 = ImbENN(smote=ImbSMOTE(k_neighbors=5, random_state=42),
                  enn=EditedNearestNeighbours(n_neighbors=3))
    X7o, y7o = our7.fit_resample(X, y)
    X7i, y7i = imb7.fit_resample(X, y)
    _, c7o = np.unique(y7o, return_counts=True)
    _, c7i = np.unique(y7i, return_counts=True)
    print("7. SMOTE-ENN         | Shape: True | Bal(our/imb): %d/%d | SynDiff: %4d | AvgDist: %.4f" % (
        int(c7o[0]==c7o[1]), int(c7i[0]==c7i[1]),
        abs((len(y7o)-len(X)) - (len(y7i)-len(X))),
        (lambda a,b: NearestNeighbors(n_neighbors=1).fit(a).kneighbors(b)[0].mean()
         if min(len(a),len(b))>0 else 0.0)(X7o[len(X):], X7i[len(X):])
    ))

    X, y = make_data()
    our8 = OurTomek(k=5, random_state=42)
    imb8 = ImbTomek(smote=ImbSMOTE(k_neighbors=5, random_state=42))
    X8o, y8o = our8.fit_resample(X, y)
    X8i, y8i = imb8.fit_resample(X, y)
    _, c8o = np.unique(y8o, return_counts=True)
    _, c8i = np.unique(y8i, return_counts=True)
    print("8. SMOTE-Tomek       | Shape: True | Bal(our/imb): %d/%d | SynDiff: %4d | AvgDist: %.4f" % (
        int(c8o[0]==c8o[1]), int(c8i[0]==c8i[1]),
        abs((len(y8o)-len(X)) - (len(y8i)-len(X))),
        (lambda a,b: NearestNeighbors(n_neighbors=1).fit(a).kneighbors(b)[0].mean()
         if min(len(a),len(b))>0 else 0.0)(X8o[len(X):], X8i[len(X):])
    ))

    print("-" * 72)
    print("Naše varijante bez imblearn ekvivalenta:")
    print("  - SafeLevelSMOTE")
    print("  - GeometricSMOTE (G-SMOTE)")
    print("  - RandomSMOTE")
    print("  - PolynomFitSMOTE")
