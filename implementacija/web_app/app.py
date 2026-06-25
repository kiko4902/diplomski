"""FastAPI backend for SMOTE visualization."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, Response
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go

from smote_variants.smote import SMOTE
from smote_variants.borderline import BorderlineSMOTE
from smote_variants.adasyn import ADASYN
from smote_variants.safe_level import SafeLevelSMOTE
from smote_variants.kmeans_smote import KMeansSMOTE
from smote_variants.svm_smote import SVMSMOTE
from smote_variants.g_smote import GeometricSMOTE
from smote_variants.random_smote import RandomSMOTE
from smote_variants.polynom_fit import PolynomFitSMOTE
from smote_variants.smote_enn import SMOTEENN, SMOTETomek
from smote_variants.baselines import (
    NoOversampling,
    RandomOversampling,
    RandomUndersampling,
)
from smote_variants.undersampling import (
    NearMiss1,
    NearMiss2,
    NearMiss3,
    TomekLinks,
    ENN,
)

app = FastAPI()

SMOTE_INFO = {
    "SMOTE": "Osnovni algoritam — linearna interpolacija između manjinskih primjera.",
    "Borderline-SMOTE (BS1)": "Fokusira se na granične primjere. BS1 koristi samo manjinske susjede.",
    "Borderline-SMOTE (BS2)": "Kao BS1, ali dopušta i većinske susjede uz ograničenje.",
    "ADASYN": "Adaptivno generira više uzoraka u gušćim područjima većinske klase.",
    "SafeLevel-SMOTE": "Generira samo u 'sigurnim' područjima s dovoljno manjinskih susjeda.",
    "KMeans-SMOTE": "Grupira manjinske primjere klasterima, više uzoraka u rijetkim klasterima.",
    "SVM-SMOTE": "Generira uzorke samo oko potpornih vektora na granici odluke.",
    "SMOTE-ENN": "SMOTE + čišćenje Edited Nearest Neighbors — uklanja šum.",
    "SMOTE-Tomek": "SMOTE + uklanjanje Tomek Link parova na granici klasa.",
    "G-SMOTE": "Geometrijsko proširenje — generira unutar sektora, ne samo na liniji.",
    "Random-SMOTE": "Nasumični smjer i udaljenost za maksimalnu raznolikost.",
    "PolynomFit-SMOTE": "Polinomna interpolacija kroz više susjeda za nelinearne distribucije.",
    "NoOversampling": "Bez preuzorkovanja — originalni podaci bez ikakve izmjene (baseline).",
    "RandomOversampling": "Nasumično dupliciranje manjinskih primjera do balansiranja.",
    "RandomUndersampling": "Nasumično uklanjanje većinskih primjera do balansiranja.",
    "NearMiss-1": "Bira većinske primjere najbliže manjinskim (prosječna udaljenost).",
    "NearMiss-2": "Bira većinske primjere s najmanjom prosječnom udaljenošću do najdaljih manjinskih.",
    "NearMiss-3": "Za svaki manjinski primjer zadržava k najbližih većinskih susjeda.",
    "TomekLinks": "Uklanja Tomek Link parove — većinski primjer kojemu je najbliži susjed manjinski i obratno.",
    "ENN": "Edited Nearest Neighbors — uklanja primjere krivo klasificirane od strane većine susjeda.",
    "WGAN": "Wasserstein GAN s gradijentnom penalizacijom — generativno preuzorkovanje manjinske klase.",
}

SMOTE_VARIANTS = {
    "SMOTE": SMOTE,
    "Borderline-SMOTE (BS1)": lambda **kw: BorderlineSMOTE(kind="BS1", **kw),
    "Borderline-SMOTE (BS2)": lambda **kw: BorderlineSMOTE(kind="BS2", **kw),
    "ADASYN": ADASYN,
    "SafeLevel-SMOTE": SafeLevelSMOTE,
    "KMeans-SMOTE": KMeansSMOTE,
    "SVM-SMOTE": SVMSMOTE,
    "SMOTE-ENN": SMOTEENN,
    "SMOTE-Tomek": SMOTETomek,
    "G-SMOTE": GeometricSMOTE,
    "Random-SMOTE": RandomSMOTE,
    "PolynomFit-SMOTE": PolynomFitSMOTE,
    "NoOversampling": NoOversampling,
    "RandomOversampling": RandomOversampling,
    "RandomUndersampling": RandomUndersampling,
    "NearMiss-1": NearMiss1,
    "NearMiss-2": NearMiss2,
    "NearMiss-3": NearMiss3,
    "TomekLinks": TomekLinks,
    "ENN": ENN,
}

# Lazy import WGAN to avoid torch DLL issues at import time
def _lazy_wgan_factory(k=None, random_state=None):
    from smote_variants.gan import WGAN
    return WGAN(k=k or 5, random_state=random_state)

SMOTE_VARIANTS["WGAN"] = _lazy_wgan_factory
SMOTE_INFO["WGAN"] = "Wasserstein GAN s gradijentnom penalizacijom — generativno preuzorkovanje manjinske klase."

# --- Static files ---
with open(os.path.join(os.path.dirname(__file__), "index.html"), "r", encoding="utf-8") as f:
    HTML_PAGE = f.read()


@app.get("/api/info")
def api_info():
    return SMOTE_INFO


@app.get("/api/variants")
def api_variants():
    return list(SMOTE_VARIANTS.keys())


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


@app.get("/api/smote")
def api_smote(
    smote_name: str = Query("SMOTE"),
    k: int = Query(5, ge=1, le=15),
    n_samples: int = Query(200, ge=50, le=800),
    ir: float = Query(5.0, ge=1.5, le=30.0),
    noise: float = Query(0.05, ge=0.0, le=0.3),
    dim_method: str = Query("PCA"),
    smote_name2: str = Query(""),
    dataset_type: str = Query("random"),
    seed: int = Query(42, ge=1, le=9999),
):
    try:
        X, y, data_meta = _make_dataset(n_samples, ir, noise, seed, dataset_type)
        n_orig = len(X)
        names = [smote_name]
        if smote_name2 and smote_name2 in SMOTE_VARIANTS:
            names.append(smote_name2)

        results = _run_compare(X, y, n_orig, names, k, seed, dim_method)
        return {"results": results, "ir": float(data_meta["ir"]), "dim": dim_method,
                "dataset": dataset_type, "n_orig": n_orig, "noise": noise}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/smote_batch")
def api_smote_batch(
    k: int = Query(5, ge=1, le=15),
    n_samples: int = Query(200, ge=50, le=800),
    ir: float = Query(5.0, ge=1.5, le=30.0),
    noise: float = Query(0.05, ge=0.0, le=0.3),
    dim_method: str = Query("PCA"),
    dataset_type: str = Query("random"),
    seed: int = Query(42, ge=1, le=9999),
):
    try:
        X, y, data_meta = _make_dataset(n_samples, ir, noise, seed, dataset_type)
        n_orig = len(X)

        # Fit reducer ONCE on original data
        if dim_method == "tSNE" and n_orig > 3:
            perp = min(30, max(5, n_orig - 1))
            reducer = TSNE(n_components=2, random_state=seed, perplexity=perp)
        else:
            reducer = PCA(n_components=2)

        # Combine original + all synthetic for a unified fit, then split back
        all_X = [X]
        n_synths = []
        for name in SMOTE_VARIANTS:
            smote = SMOTE_VARIANTS[name](k=k, random_state=seed)
            X_res, _ = smote.fit_resample(X, y)
            synth = X_res[n_orig:]
            all_X.append(synth)
            n_synths.append(len(synth))

        all_stacked = np.vstack(all_X)
        all_2d = reducer.fit_transform(all_stacked)

        X_orig_2d = all_2d[:n_orig]
        offset = n_orig
        results = []
        for i, name in enumerate(SMOTE_VARIANTS):
            n_s = n_synths[i]
            X_synth_2d = all_2d[offset:offset + n_s]

            fig = go.Figure()
            classes = np.unique(y)
            for ci, cls in enumerate(classes):
                mask = y == cls
                fig.add_trace(go.Scatter(
                    x=X_orig_2d[mask, 0], y=X_orig_2d[mask, 1],
                    mode="markers", marker=dict(size=5, color=["#636efa","#ef553b"][ci], opacity=0.6, line=dict(width=0.3, color="white")),
                    name=f"Class {cls}", showlegend=False,
                ))
            if n_s > 0:
                fig.add_trace(go.Scatter(
                    x=X_synth_2d[:, 0], y=X_synth_2d[:, 1],
                    mode="markers", marker=dict(size=3, color="#00cc96", opacity=0.4, symbol="x"),
                    name="Synthetic", showlegend=False,
                ))
            fig.update_layout(
                title=dict(text=f"{name}<br><sub>{n_s} synth</sub>", font=dict(size=12)),
                height=260, margin=dict(l=10, r=10, t=35, b=10),
                plot_bgcolor="#fafafa", paper_bgcolor="#fafafa",
            )
            results.append({"name": name, "n_synth": n_s, "total": n_orig + n_s, "plot_json": fig.to_json()})
            offset += n_s

        return {"results": results, "ir": float(data_meta["ir"]), "dim": dim_method,
                "dataset": dataset_type, "n_orig": n_orig, "noise": noise, "k": k}
    except Exception as e:
        return {"error": str(e)}


def _run_compare(X, y, n_orig, names, k, seed, dim_method):
    if dim_method == "tSNE" and n_orig > 3:
        perp = min(30, max(5, n_orig - 1))
        reducer = TSNE(n_components=2, random_state=seed, perplexity=perp)
    else:
        reducer = PCA(n_components=2)

    all_X = [X]
    n_synths = []
    for name in names:
        smote = SMOTE_VARIANTS[name](k=k, random_state=seed)
        X_res, _ = smote.fit_resample(X, y)
        n_synths.append(len(X_res) - n_orig)
        all_X.append(X_res[n_orig:])

    all_stacked = np.vstack(all_X)
    all_2d = reducer.fit_transform(all_stacked)
    X_orig_2d = all_2d[:n_orig]

    results = []
    offset = n_orig
    for i, name in enumerate(names):
        n_s = n_synths[i]
        X_synth_2d = all_2d[offset:offset + n_s]

        fig = go.Figure()
        classes = np.unique(y)
        for ci, cls in enumerate(classes):
            mask = y == cls
            fig.add_trace(go.Scatter(
                x=X_orig_2d[mask, 0], y=X_orig_2d[mask, 1],
                mode="markers", marker=dict(size=7, color=["#636efa","#ef553b"][ci], opacity=0.75, line=dict(width=0.5, color="white")),
                name=f"Class {cls}",
            ))
        if n_s > 0:
            fig.add_trace(go.Scatter(
                x=X_synth_2d[:, 0], y=X_synth_2d[:, 1],
                mode="markers", marker=dict(size=4, color="#00cc96", opacity=0.5, symbol="x"),
                name="Synthetic",
            ))
        fig.update_layout(
            title=dict(text=f"{name} (k={k}) — {n_s} synthesized", font=dict(size=14)),
            height=520, margin=dict(l=30, r=20, t=50, b=30),
            plot_bgcolor="#fafafa", paper_bgcolor="#fafafa",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        )
        results.append({"name": name, "n_synth": n_s, "total": n_orig + n_s, "plot_json": fig.to_json()})
        offset += n_s
    return results


def _make_dataset(n_samples, ir, noise, seed, dataset_type):
    rng = np.random.RandomState(seed)
    n_min = max(4, int(n_samples / (ir + 1)))
    n_maj = n_samples - n_min

    if dataset_type == "normal":
        X_min = rng.randn(n_min, 5) * 1.5 + np.array([2, 2, 0, 0, 0])
        X_maj = rng.randn(n_maj, 5) * 1.0
        X, y = np.vstack([X_min, X_maj]), np.hstack([np.ones(n_min), np.zeros(n_maj)])
    elif dataset_type == "exponential":
        X_min = rng.exponential(1.0, (n_min, 5)) + 1.0
        X_maj = rng.exponential(0.5, (n_maj, 5))
        X, y = np.vstack([X_min, X_maj]), np.hstack([np.ones(n_min), np.zeros(n_maj)])
    elif dataset_type == "multimodal":
        n_min1, n_min2 = n_min // 2, n_min - n_min // 2
        X_min1 = rng.randn(n_min1, 5) * 0.8 + np.array([3, 3, 0, 0, 0])
        X_min2 = rng.randn(n_min2, 5) * 0.8 + np.array([-3, -3, 0, 0, 0])
        X_maj = rng.randn(n_maj, 5) * 1.2
        X = np.vstack([X_min1, X_min2, X_maj])
        y = np.hstack([np.ones(n_min), np.zeros(n_maj)])
    elif dataset_type == "circles":
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=seed)
        n_keep = max(4, int(n_samples / (ir + 1)))
        min_idx = np.where(y == 0)[0][:n_keep]
        keep = np.ones(len(X), dtype=bool)
        keep[np.setdiff1d(np.where(y == 0)[0], min_idx)] = False
        X, y = X[keep], y[keep]
    elif dataset_type == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X, y = data.data.astype(np.float64), data.target.astype(int)
    elif dataset_type == "wine":
        from sklearn.datasets import load_wine
        data = load_wine()
        X, y = data.data.astype(np.float64), np.where(data.target == 0, 1, 0)
    elif dataset_type == "iris":
        from sklearn.datasets import load_iris
        data = load_iris()
        X, y = data.data.astype(np.float64), np.where(data.target == 0, 1, 0)
    else:
        X, y = make_classification(
            n_samples=n_samples, n_features=5, n_redundant=0,
            n_clusters_per_class=1, weights=[1/(ir+1), ir/(ir+1)],
            flip_y=noise, random_state=seed,
        )
    classes, counts = np.unique(y, return_counts=True)
    actual_ir = max(counts) / max(min(counts), 1e-9)
    return X.astype(np.float64), y.astype(int), {"ir": actual_ir, "type": dataset_type}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
