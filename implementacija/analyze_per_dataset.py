"""Per-dataset i per-classifier analiza — gdje varijante pobjeđuju SMOTE?"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from config import RAW_DIR

dfs = [pd.read_csv(os.path.join(RAW_DIR, f)) for f in os.listdir(RAW_DIR) if f.endswith(".csv") and "_k7" not in f]
df = pd.concat(dfs, ignore_index=True)
f1 = df[df["metric"] == "f1"]
best = f1.groupby(["dataset", "classifier", "smote"])["mean"].max().reset_index()

print("=" * 80)
print("  1. PO DATASETU — gdje varijante pobjeđuju SMOTE? (F1)")
print("=" * 80)

for ds in sorted(best["dataset"].unique()):
    ds_data = best[best["dataset"] == ds]
    smote_val = float(ds_data[ds_data["smote"] == "SMOTE"]["mean"].mean())
    baseline_val = float(ds_data[ds_data["smote"] == "NoOversampling"]["mean"].mean())

    others = ds_data[~ds_data["smote"].isin(["SMOTE", "NoOversampling"])]
    avg_others = others.groupby("smote")["mean"].mean()

    better = [(s, v) for s, v in avg_others.items() if v > smote_val + 0.005]
    worse = [(s, v) for s, v in avg_others.items() if v < smote_val - 0.005]

    ir = round(float(1.0), 1)  # placeholder, actual IR is in meta
    print(f"\n{ds:25s}  SMOTE={smote_val:.4f}  Baseline={baseline_val:.4f}")

    if better:
        best3 = sorted(better, key=lambda x: -x[1])[:3]
        print(f"  BOLJI:   {', '.join(f'{s} ({v:.4f})' for s, v in best3)}")
    if worse:
        worst3 = sorted(worse, key=lambda x: x[1])[:3]
        print(f"  LOSIJI:  {', '.join(f'{s} ({v:.4f})' for s, v in worst3)}")
    if not better and not worse:
        print(f"  (svi priblizno jednaki SMOTE-u)")

print("\n" + "=" * 80)
print("  2. PO KLASIFIKATORU — gdje varijante pobjeđuju SMOTE? (F1)")
print("=" * 80)

for clf in sorted(best["classifier"].unique()):
    clf_data = best[best["classifier"] == clf]
    smote_val = float(clf_data[clf_data["smote"] == "SMOTE"]["mean"].mean())
    baseline_val = float(clf_data[clf_data["smote"] == "NoOversampling"]["mean"].mean())

    others = clf_data[~clf_data["smote"].isin(["SMOTE", "NoOversampling"])]
    avg_others = others.groupby("smote")["mean"].mean()

    better = [(s, v) for s, v in avg_others.items() if v > smote_val + 0.005]
    worse = [(s, v) for s, v in avg_others.items() if v < smote_val - 0.005]

    print(f"\n{clf:8s}  SMOTE={smote_val:.4f}  Baseline={baseline_val:.4f}")

    if better:
        best3 = sorted(better, key=lambda x: -x[1])[:3]
        print(f"  BOLJI:   {', '.join(f'{s} ({v:.4f})' for s, v in best3)}")
    if worse:
        worst3 = sorted(worse, key=lambda x: x[1])[:3]
        print(f"  LOSIJI:  {', '.join(f'{s} ({v:.4f})' for s, v in worst3)}")

print("\n" + "=" * 80)
print("  3. GLOBALNI RANG (F1, average preko svega)")
print("=" * 80)

ranking = best.groupby("smote")["mean"].mean().sort_values(ascending=False)
for i, (s, v) in enumerate(ranking.items()):
    bar = "#" * int(v * 40)
    marker = ""
    if s == "SMOTE":
        marker = " <-- SMOTE"
    elif s == "NoOversampling":
        marker = " <-- Baseline"
    print(f"  {i+1:2d}. {s:25s} {v:.4f} {bar}{marker}")
