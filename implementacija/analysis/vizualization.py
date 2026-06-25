import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import FIGURES_DIR
from analysis.statistical import load_all_results, compute_avg_rankings, compute_rank_matrix


os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# 1. BOXPLOT
# ============================================================


def plot_boxplot(df, metric="f1", save=True):
    metric_df = df[df["metric"] == metric].copy()
    plt.figure(figsize=(12, 6))
    order = metric_df.groupby("smote")["mean"].median().sort_values().index.tolist()
    sns.boxplot(data=metric_df, x="smote", y="mean", order=order, palette="Set2")
    plt.title(f"Distribucija {metric.upper()} po SMOTE varijantama")
    plt.xlabel("SMOTE varijanta")
    plt.ylabel(metric.upper())
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, f"boxplot_{metric}.pdf")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved boxplot to {path}")


# ============================================================
# 2. VIOLIN PLOT
# ============================================================
def plot_violin(df, metric="f1", save=True, top_n=None):
    metric_df = df[df["metric"] == metric].copy()
    order = metric_df.groupby("smote")["mean"].median().sort_values(ascending=False).index.tolist()
    if top_n:
        order = order[:top_n]
        metric_df = metric_df[metric_df["smote"].isin(order)]

    plt.figure(figsize=(14, 6))
    sns.violinplot(data=metric_df, x="smote", y="mean", order=order,
                   palette="Set2", inner="quartile", cut=0)
    plt.title(f"Distribucija {metric.upper()} po SMOTE varijantama (violin)")
    plt.xlabel("SMOTE varijanta")
    plt.ylabel(metric.upper())
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, f"violin_{metric}.pdf")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved violin plot to {path}")


# ============================================================
# 3. CD DIJAGRAM
# ============================================================
def plot_cd_diagram(df, metric="f1", save=True):
    """Kritični dijagram — statističke grupe."""
    try:
        rank_matrix = compute_rank_matrix(df, metric)
        avg_ranks = rank_matrix.mean().sort_values()
        n_datasets = rank_matrix.shape[0]
        n_algos = rank_matrix.shape[1]

        from scipy.stats import studentized_range
        q_alpha = studentized_range.ppf(0.95, n_algos, (n_datasets - 1) * (n_algos - 1))
        cd = q_alpha * np.sqrt(n_algos * (n_algos + 1) / (6 * n_datasets))

        fig, ax = plt.subplots(figsize=(10, 4))
        y_positions = np.arange(len(avg_ranks))
        ax.scatter(avg_ranks.values, y_positions, s=60, zorder=5)

        for i, (algo, rank) in enumerate(avg_ranks.items()):
            ax.text(rank + 0.02, i, algo, va="center", fontsize=10)

        # Nemenyi CD grupe: pairwise usporedba svih algoritama
        n = len(avg_ranks)
        groups = []
        remaining = set(range(n))
        while remaining:
            seed = min(remaining)
            group = {seed}
            for j in range(n):
                if j != seed and abs(avg_ranks.iloc[j] - avg_ranks.iloc[seed]) <= cd:
                    group.add(j)
            groups.append(sorted(group))
            remaining -= group

        for group in groups:
            if len(group) > 1:
                y_center = np.mean(group)
                ax.plot(
                    [avg_ranks.iloc[group[0]], avg_ranks.iloc[group[-1]]],
                    [y_center, y_center],
                    "k-", linewidth=2,
                )

        ax.set_title(f"CD dijagram — {metric.upper()}")
        ax.set_xlabel("Prosječan rang")
        ax.set_ylim(-1, len(avg_ranks))
        ax.invert_yaxis()
        plt.tight_layout()
        if save:
            path = os.path.join(FIGURES_DIR, f"cd_diagram_{metric}.pdf")
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  Saved CD diagram to {path}")
    except Exception as e:
        print(f"  CD diagram failed: {e}")


def plot_heatmap(df, metric="f1", save=True):
    pivot = df[df["metric"] == metric].pivot_table(
        index="dataset", columns="smote", values="mean", aggfunc="median",
    )
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", cbar_kws={"label": metric.upper()})
    plt.title(f"Heatmap — {metric.upper()} (Data × SMOTE)")
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, f"heatmap_{metric}.pdf")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved heatmap to {path}")


# ============================================================
# 5. PER-DATASET BAR CHART: Baseline vs najbolji SMOTE
# ============================================================
def plot_per_dataset_bars(df, metric="f1", save=True):
    metric_df = df[df["metric"] == metric].copy()
    datasets = sorted(metric_df["dataset"].unique())

    baseline_vals = []
    best_smote_vals = []
    best_smote_names = []

    for ds in datasets:
        ds_df = metric_df[metric_df["dataset"] == ds]
        baseline = ds_df[ds_df["smote"] == "NoOversampling"]["mean"].mean()
        smote_only = ds_df[ds_df["smote"] != "NoOversampling"]
        if len(smote_only) > 0:
            best = smote_only.groupby("smote")["mean"].mean().sort_values(ascending=False)
            best_smote_vals.append(best.iloc[0])
            best_smote_names.append(best.index[0])
        else:
            best_smote_vals.append(baseline)
            best_smote_names.append("N/A")
        baseline_vals.append(baseline)

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, baseline_vals, width, label="NoOversampling (baseline)", color="#d62728")
    bars2 = ax.bar(x + width/2, best_smote_vals, width, label="Najbolji SMOTE", color="#2ca02c")

    for i, (b, s) in enumerate(zip(baseline_vals, best_smote_vals)):
        delta = s - b
        color = "green" if delta > 0.01 else "gray"
        ax.annotate(f"{delta:+.3f}", (x[i] + width/2, s + 0.01),
                    ha="center", fontsize=8, color=color, fontweight="bold")

    ax.set_ylabel(metric.upper())
    ax.set_title(f"Baseline vs najbolji SMOTE po datasetu ({metric.upper()})")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, f"per_dataset_bars_{metric}.pdf")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved per-dataset bars to {path}")


# ============================================================
# 6. PCA SCATTER sintetičkih primjera
# ============================================================
def plot_smote_scatter(dataset_name, X_orig, y_orig, smote_methods, metric="f1", save=True):
    """Za odabrane SMOTE metode generira PCA scatter original + synthetic."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_orig)

    n_methods = len(smote_methods)
    cols = min(3, n_methods)
    rows = (n_methods + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, (name, smote) in zip(axes, smote_methods.items()):
        try:
            X_res, y_res = smote.fit_resample(X_orig, y_orig)
            X_all = np.vstack([X_orig, X_res])
            y_all = np.hstack([y_orig, y_res])
            is_synthetic = np.hstack([np.zeros(len(y_orig)), np.ones(len(y_res))])

            X_all_pca = pca.transform(X_all)

            # Original minority
            mask_min = (y_orig == 1)
            ax.scatter(X_pca[mask_min, 0], X_pca[mask_min, 1],
                      c="blue", s=20, alpha=0.6, label="manjina (orig)", edgecolors="none")
            # Original majority
            mask_maj = (y_orig == 0)
            ax.scatter(X_pca[mask_maj, 0], X_pca[mask_maj, 1],
                      c="red", s=20, alpha=0.3, label="vecina (orig)", edgecolors="none")
            # Synthetic
            mask_syn = is_synthetic == 1
            ax.scatter(X_all_pca[mask_syn, 0], X_all_pca[mask_syn, 1],
                      c="green", s=15, alpha=0.7, marker="^", label="sinteticki", edgecolors="none")

            ax.set_title(f"{name}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        except Exception as e:
            ax.text(0.5, 0.5, f"Greska:\n{e}", ha="center", va="center", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    # Sakrij visak osi
    for j in range(n_methods, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"SMOTE vizualizacija — {dataset_name} (PCA 2D)", fontsize=12, y=1.02)
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, f"scatter_smote_{dataset_name}.pdf")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved SMOTE scatter to {path}")


def generate_all_plots(metric="f1"):
    df = load_all_results()
    print(f"\n  Generating plots for {metric.upper()}...")
    plot_boxplot(df, metric)
    plot_violin(df, metric)
    plot_cd_diagram(df, metric)
    plot_heatmap(df, metric)
    plot_per_dataset_bars(df, metric)
