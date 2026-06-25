import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from scikit_posthocs import posthoc_nemenyi_friedman
import os
from config import RAW_DIR, TABLES_DIR, FIGURES_DIR


def load_all_results():
    """Učitava sve CSV datoteke iz raw direktorija."""
    dfs = []
    for fname in os.listdir(RAW_DIR):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(RAW_DIR, fname))
            dfs.append(df)
    if not dfs:
        raise ValueError("No result files found in " + RAW_DIR)
    return pd.concat(dfs, ignore_index=True)


def compute_avg_rankings(df, metric="f1"):
    """Računa prosječni rang svakog SMOTE algoritma kroz sve skupove i klasifikatore."""
    metric_df = df[df["metric"] == metric].copy()
    best_k = metric_df.groupby(["dataset", "classifier", "smote"])["mean"].max().reset_index()
    groupings = best_k.groupby(["dataset", "classifier"])
    rankings = []

    for _, group in groupings:
        group = group.sort_values("mean", ascending=False)
        group["rank"] = range(1, len(group) + 1)
        rankings.append(group)

    rank_df = pd.concat(rankings)
    avg_ranks = rank_df.groupby("smote")["rank"].mean().sort_values()
    return avg_ranks


def compute_rank_matrix(df, metric="f1"):
    """Vraća matricu rangova: redovi = (dataset, classifier), stupci = SMOTE varijante.
    Za svaku metodu uzima se NAJBOLJI k (maksimalni mean)."""
    metric_df = df[df["metric"] == metric].copy()
    # Agregiraj po najboljem k za svaku kombinaciju
    best_k = metric_df.groupby(["dataset", "classifier", "smote"])["mean"].max().reset_index()
    pivot = best_k.pivot_table(
        index=["dataset", "classifier"],
        columns="smote",
        values="mean",
        aggfunc="first",
    )
    rank_matrix = pivot.rank(axis=1, ascending=False)
    return rank_matrix


def friedman_test(df, metric="f1"):
    """Friedmanov test — postoje li značajne razlike među algoritmima?"""
    rank_matrix = compute_rank_matrix(df, metric)
    rank_matrix = rank_matrix.dropna(axis=1, how="any").dropna(axis=0, how="any")
    smote_variants = rank_matrix.columns.tolist()
    if len(smote_variants) < 2:
        return None, None, None

    rankings_by_algo = [rank_matrix[algo].values for algo in smote_variants]
    stat, p_value = friedmanchisquare(*rankings_by_algo)
    return stat, p_value, smote_variants


def nemenyi_posthoc(df, metric="f1"):
    """Nemenyi post-hoc test."""
    rank_matrix = compute_rank_matrix(df, metric)
    if rank_matrix.shape[1] < 2:
        return None

    long_form = rank_matrix.reset_index().melt(
        id_vars=["dataset", "classifier"],
        var_name="smote",
        value_name="rank",
    )

    pivot_ranks = long_form.pivot_table(
        index=["dataset", "classifier"],
        columns="smote",
        values="rank",
    )

    try:
        posthoc = posthoc_nemenyi_friedman(pivot_ranks.values)
        posthoc.columns = pivot_ranks.columns
        posthoc.index = pivot_ranks.columns
        return posthoc
    except Exception:
        return None


def wilcoxon_vs_baseline(df, metric="f1", baseline="SMOTE"):
    """Wilcoxon signed-rank test uspoređuje svaku izvedenicu s baselineom."""
    metric_df = df[df["metric"] == metric].copy()
    best_k = metric_df.groupby(["dataset", "classifier", "smote"])["mean"].max().reset_index()
    pivot = best_k.pivot_table(
        index=["dataset", "classifier"],
        columns="smote",
        values="mean",
        aggfunc="first",
    )

    if baseline not in pivot.columns:
        return {}

    results = {}
    baseline_vals_all = pivot[baseline]
    for algo in pivot.columns:
        if algo == baseline:
            continue
        try:
            mask = baseline_vals_all.notna() & pivot[algo].notna()
            if mask.sum() < 3:
                results[algo] = {"statistic": np.nan, "p_value": np.nan}
                continue
            stat, p_val = wilcoxon(baseline_vals_all[mask], pivot[algo][mask])
            results[algo] = {"statistic": stat, "p_value": p_val}
        except Exception:
            results[algo] = {"statistic": np.nan, "p_value": np.nan}
    return results


def run_all_statistical_tests(metric="f1", baseline="SMOTE"):
    """Pokreće sva tri statistička testa za jednu metriku i sprema tablice."""
    df = load_all_results()
    print(f"\n{'='*60}")
    print(f"  Statistical Analysis — {metric.upper()}")
    print(f"{'='*60}")

    stat, p_val, variants = friedman_test(df, metric)
    if stat is not None and not np.isnan(stat):
        print(f"\nFriedman test:")
        print(f"  Statistic = {stat:.4f}")
        print(f"  p-value   = {p_val:.6f}")
        print(f"  {'Significant' if p_val < 0.05 else 'Not significant'} (alpha=0.05)")
    else:
        print(f"\nFriedman test: skipped (need 2+ SMOTE variants)")

    posthoc = nemenyi_posthoc(df, metric)
    if posthoc is not None:
        print(f"\nNemenyi post-hoc (p-values):")
        print(posthoc.to_string(float_format=lambda x: f"{x:.4f}"))

    wilcox = wilcoxon_vs_baseline(df, metric, baseline)
    if wilcox:
        print(f"\nWilcoxon vs {baseline}:")
        for algo, res in sorted(wilcox.items()):
            sig = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else "*" if res["p_value"] < 0.05 else ""
            print(f"  {algo:25s}  p={res['p_value']:.4f} {sig}")

    wilcox_base = wilcoxon_vs_baseline(df, metric, "NoOversampling")
    if wilcox_base:
        print(f"\nWilcoxon vs NoOversampling:")
        for algo, res in sorted(wilcox_base.items()):
            sig = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else "*" if res["p_value"] < 0.05 else ""
            print(f"  {algo:25s}  p={res['p_value']:.4f} {sig}")

    # Save all tables
    save_ranking_table(df, metric)
    save_pvalue_table(posthoc)
    save_wilcoxon_table(wilcox, baseline)
    save_wilcoxon_table(wilcox_base, "NoOversampling", filename="wilcoxon_vs_nooversampling.tex")

    return stat, p_val, posthoc, wilcox


def save_ranking_table(df, metric="f1", filename="avg_rankings.tex"):
    avg_ranks = compute_avg_rankings(df, metric)
    os.makedirs(TABLES_DIR, exist_ok=True)
    filepath = os.path.join(TABLES_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lr}\n")
        f.write("\\toprule\n")
        f.write("Algoritam & Prosječan rang \\\\\n")
        f.write("\\midrule\n")
        for algo, rank in avg_ranks.items():
            f.write(f"{algo} & {rank:.2f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"  Saved ranking table to {filepath}")


def save_pvalue_table(posthoc, alpha=0.05, filename="nemenyi_pvalues.tex"):
    """Sprema Nemenyi post-hoc p-vrijednosti kao LaTeX tablicu."""
    if posthoc is None:
        print("  No posthoc results to save.")
        return

    os.makedirs(TABLES_DIR, exist_ok=True)
    filepath = os.path.join(TABLES_DIR, filename)

    algos = posthoc.columns.tolist()
    n = len(algos)
    max_name = max(len(a) for a in algos) if algos else 10

    with open(filepath, "w", encoding="utf-8") as f:
        col_spec = "l" + "c" * n
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")
        f.write(" & " + " & ".join(algos) + " \\\\\n")
        f.write("\\midrule\n")
        for i, a1 in enumerate(algos):
            row = [a1]
            for j, a2 in enumerate(algos):
                if i == j:
                    row.append("---")
                else:
                    p = posthoc.iloc[i, j]
                    if p < 0.001:
                        row.append("\\textbf{<0.001}")
                    elif p < 0.01:
                        row.append(f"\\textbf{{{p:.3f}}}")
                    elif p < alpha:
                        row.append(f"\\textit{{{p:.3f}}}")
                    else:
                        row.append(f"{p:.3f}")
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"  Saved p-value table to {filepath}")


def save_wilcoxon_table(wilcox_results, baseline="SMOTE", filename="wilcoxon_vs_baseline.tex"):
    """Sprema Wilcoxon vs baseline rezultate kao LaTeX tablicu."""
    if not wilcox_results:
        print("  No Wilcoxon results to save.")
        return

    os.makedirs(TABLES_DIR, exist_ok=True)
    filepath = os.path.join(TABLES_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrl}\n")
        f.write("\\toprule\n")
        f.write(f"Algoritam (vs {baseline}) & Statistika & p-vrijednost & Značajnost \\\\\n")
        f.write("\\midrule\n")
        for algo, res in sorted(wilcox_results.items()):
            stat = res["statistic"]
            p = res["p_value"]
            if np.isnan(p):
                sig = ""
            elif p < 0.001:
                sig = "***"
            elif p < 0.01:
                sig = "**"
            elif p < 0.05:
                sig = "*"
            else:
                sig = "n.s."
            f.write(f"{algo} & {stat:.1f} & {p:.4f} & {sig} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"  Saved Wilcoxon table to {filepath}")
