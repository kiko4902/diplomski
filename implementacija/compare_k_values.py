"""Usporedba k=[3,5] vs k=[7,10] — vidi ima li razlike."""
import pandas as pd
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from config import RAW_DIR

datasets = ["ecoli", "yeast_me2", "optical_digits", "synth_high_ir", "synth_clean"]
METRICS = ["f1", "g_mean", "auc_roc"]

for ds in datasets:
    f_lo = os.path.join(RAW_DIR, f"results_{ds}.csv")
    f_hi = os.path.join(RAW_DIR, f"results_{ds}_k7_k10.csv")

    if not os.path.exists(f_lo):
        print(f"\n{ds}: No k=[3,5] results yet (run still in progress)")
        continue

    lo = pd.read_csv(f_lo)
    hi = pd.read_csv(f_hi)

    print(f"\n{'='*70}")
    print(f"  {ds}")
    print(f"{'='*70}")
    print(f"{'Metric':10s} {'SMOTE':22s} {'Clf':6s} {'k=3,5':>8s} {'k=7,10':>8s} {'Delta':>8s} {'Verdict':>10s}")
    print("-" * 70)

    for metric in METRICS:
        lo_m = lo[lo["metric"] == metric]
        hi_m = hi[hi["metric"] == metric]

        for smote in ["SMOTE", "ADASYN", "Borderline-SMOTE1"]:
            for clf in ["rf", "lr"]:
                lo_val = lo_m[(lo_m["smote"] == smote) & (lo_m["classifier"] == clf)]["mean"].mean()
                hi_val = hi_m[(hi_m["smote"] == smote) & (hi_m["classifier"] == clf)]["mean"].mean()

                if pd.isna(lo_val) or pd.isna(hi_val):
                    continue

                delta = hi_val - lo_val
                if abs(delta) < 0.003:
                    verdict = "SAME"
                elif abs(delta) < 0.01:
                    verdict = f"{delta:+.4f} ~"
                else:
                    verdict = f"{delta:+.4f} !!!"

                print(f"{metric:10s} {smote:22s} {clf:6s} {lo_val:8.4f} {hi_val:8.4f} {hi_val-lo_val:+8.4f} {verdict}")
