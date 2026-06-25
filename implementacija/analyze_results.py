import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

from config import RAW_DIR
RAW = RAW_DIR

datasets = {}
for fname in sorted(os.listdir(RAW)):
    if fname.startswith("results_") and fname.endswith(".csv"):
        ds_name = fname[len("results_"):-4]
        path = os.path.join(RAW, fname)
        df = pd.read_csv(path)
        datasets[ds_name] = df
        print(f"Loaded {ds_name}")
print()

METRICS = ['f1', 'g_mean', 'auc_roc', 'auc_pr', 'balanced_accuracy', 'mcc', 'f2']
all_dfs = pd.concat(datasets.values())

# ============================================================
# 1. PER DATASET: Baseline vs best SMOTE
# ============================================================
print('=' * 70)
print('1. BASELINE (NoOversampling) vs NAJBOLJI SMOTE PO F1')
print('=' * 70)
for name, df in datasets.items():
    f1 = df[df['metric'] == 'f1']
    baseline = f1[(f1['smote'] == 'NoOversampling')].groupby('classifier')['mean'].mean().mean()
    # Best SMOTE per classifier, then average
    best_by_clf = f1[f1['smote'] != 'NoOversampling'].groupby(['classifier', 'smote'])['mean'].mean().groupby('classifier').max()
    best_avg = best_by_clf.mean()
    print(f'{name:20s}: Baseline F1={baseline:.4f}, Best SMOTE F1={best_avg:.4f}, Delta={best_avg - baseline:+.4f}')

# ============================================================
# 2. GLOBAL RANKING: Prosjecni F1 po metodi (preko sva 4 dataset-a)
# ============================================================
print('\n' + '=' * 70)
print('2. GLOBALNI RANG SMOTE METODA (F1 prosjek preko 4 dataset-a)')
print('=' * 70)
f1_all = all_dfs[all_dfs['metric'] == 'f1']
# Average by smote across all datasets and classifiers
ranking = f1_all.groupby('smote')['mean'].mean().sort_values(ascending=False)
for i, (smote, val) in enumerate(ranking.items()):
    bar = '#' * int(val * 40)
    print(f'  {i+1:2d}. {smote:22s}: {val:.4f} {bar}')

# ============================================================
# 3. WHICH SMOTE BEATS BASELINE ON ECOLI (hard dataset)?
# ============================================================
print('\n' + '=' * 70)
print('3. ECOLI (IR=8.6) - koje metode poboljsavaju F1 vs baseline?')
print('=' * 70)
f1_ecoli = datasets['ecoli'][datasets['ecoli']['metric'] == 'f1']
base_f1 = f1_ecoli[f1_ecoli['smote'] == 'NoOversampling']['mean'].mean()
print(f'  Baseline NoOversampling F1: {base_f1:.4f}')
smote_avg = f1_ecoli[f1_ecoli['smote'] != 'NoOversampling'].groupby('smote')['mean'].mean().sort_values(ascending=False)
for smote, val in smote_avg.items():
    delta = val - base_f1
    marker = '++' if delta > 0.01 else ('+' if delta > 0 else '')
    print(f'  {smote:22s}: F1={val:.4f}  Delta={delta:+.4f} {marker}')

# ============================================================
# 4. KLASIFIKATORI - rangiranje
# ============================================================
print('\n' + '=' * 70)
print('4. PROSJECAN F1 PO KLASIFIKATORU (preko svega)')
print('=' * 70)
clf_avg = f1_all.groupby('classifier')['mean'].mean().sort_values(ascending=False)
for clf, val in clf_avg.items():
    print(f'  {clf:8s}: {val:.4f}')

print('\n  PO DATASETU:')
clf_by_ds = f1_all.groupby(['dataset', 'classifier'])['mean'].mean().reset_index()
for name in datasets:
    sub = clf_by_ds[clf_by_ds['dataset'] == name].sort_values('mean', ascending=False)
    top = sub.iloc[0]
    bot = sub.iloc[-1]
    print(f'  {name:20s}: best={top["classifier"]:8s} ({top["mean"]:.4f})  worst={bot["classifier"]:8s} ({bot["mean"]:.4f})')

# ============================================================
# 5. K-VRIJEDNOSTI - utjecaj
# ============================================================
print('\n' + '=' * 70)
print('5. UTJECAJ K NA F1 (originalni SMOTE)')
print('=' * 70)
smote_f1 = f1_all[f1_all['smote'] == 'SMOTE'].groupby(['dataset', 'k'])['mean'].mean()
for name in datasets:
    vals = {int(k): f'{v:.4f}' for k, v in smote_f1[name].items()}
    print(f'  {name:20s}: {vals}')

# ============================================================
# 6. METRICS - koreliraju li F1, G-Mean, AUC-ROC?
# ============================================================
print('\n' + '=' * 70)
print('6. KORELACIJA METRIKA (na ecoli, gdje ima razlike)')
print('=' * 70)
pivot = all_dfs[all_dfs['dataset'] == 'ecoli'].pivot_table(
    values='mean', index=['smote', 'k', 'classifier'], columns='metric'
).dropna()
corr = pivot.corr()
for m1 in METRICS:
    for m2 in METRICS:
        if m1 < m2:
            print(f'  {m1:18s} vs {m2:18s}: r={corr.loc[m1, m2]:.3f}')

# ============================================================
# 7. SMOTE families comparison
# ============================================================
print('\n' + '=' * 70)
print('7. USPOREDBA GRUPA METODA (prosjecni F1)')
print('=' * 70)
groups = {
    'Regular SMOTE': ['SMOTE'],
    'Borderline': ['Borderline-SMOTE1', 'Borderline-SMOTE2'],
    'Adaptive': ['ADASYN', 'SafeLevel-SMOTE'],
    'Cluster/SVM': ['KMeans-SMOTE', 'SVM-SMOTE'],
    'Cleaning': ['SMOTE-ENN', 'SMOTE-Tomek'],
    'Geometric': ['G-SMOTE', 'Random-SMOTE', 'PolynomFit-SMOTE'],
    'Undersampling': ['NearMiss-1', 'NearMiss-2', 'NearMiss-3', 'TomekLinks', 'ENN'],
    'Baseline': ['NoOversampling', 'RandomOversampling', 'RandomUndersampling'],
}
for gname, methods in groups.items():
    vals = f1_all[f1_all['smote'].isin(methods)]['mean']
    print(f'  {gname:20s}: F1={vals.mean():.4f} (n={len(vals)})')

print('\nDONE.')
