#!/usr/bin/env python3
"""Sweep CEBRA hyperparameters and save embedding plots."""
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cebra import CEBRA
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch

# === Config ===
NEURAL_DIR = "/global/scratch/users/ahmostafa/CEBRA_modeling_local/1000_ICARE_patient_10s_94f_with_spike"
LABELS_PATH = "/global/scratch/users/ahmostafa/CEBRA_modeling_local/labels/ICARE_clinical.csv"
OUT_DIR = "visualizations/sweep"
SUBSAMPLE = 2_000_000
PLOT_POINTS = 50_000
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GRID = {
    'max_iterations': [5000, 10000],
    'temperature': [0.5, 1.0, 1.12],
    'learning_rate': [1e-4, 3e-4, 1e-3],
    'output_dimension': [3, 8],
    'batch_size': [512, 1024],
}

# === Stream CSVs one at a time ===
print("Loading labels...")
labels_df = pd.read_csv(LABELS_PATH)
patient_to_cpc = dict(zip(labels_df['pat_ICARE'], labels_df['cpc_bin']))
csv_files = sorted(Path(NEURAL_DIR).glob("*.csv"))
print(f"Found {len(csv_files)} CSVs")

# Detect feature columns from first file
sample = pd.read_csv(csv_files[0], encoding='utf-8-sig', nrows=5)
sample = sample.merge(labels_df, left_on='patient', right_on='pat_ICARE', how='inner')
meta_cols = {'patient', 'seg_no', 'file', 'row_in_seg', 'rel_sec',
             'pat_ICARE', 'cpc', 'cpc_bin', 'ROSC(minutes)', 'age',
             'sex', 'vfib', 'time_to_CA(seconds)'}
feat_cols = [c for c in sample.columns if c not in meta_cols]
n_feats = len(feat_cols)
print(f"Features: {n_feats}")

# Split patients first
valid_patients = [p for p in labels_df['pat_ICARE'] if pd.notna(patient_to_cpc.get(p))]
pat_cpc = [patient_to_cpc[p] for p in valid_patients]
train_pats, test_pats = train_test_split(
    valid_patients, test_size=0.2, stratify=pat_cpc, random_state=SEED)
train_set, test_set = set(train_pats), set(test_pats)
print(f"Patients — train: {len(train_pats)}, test: {len(test_pats)}")

# Stream CSVs, extract features per patient
train_arrays, test_arrays = [], []
train_meta, test_meta = {'cpc': [], 'rel': []}, {'cpc': [], 'rel': []}

for fi, f in enumerate(csv_files):
    df = pd.read_csv(f, encoding='utf-8-sig')
    pid = df['patient'].iloc[0]
    if pid not in train_set and pid not in test_set:
        continue
    df = df.merge(labels_df, left_on='patient', right_on='pat_ICARE', how='inner')
    df = df.dropna(subset=['cpc_bin'])
    if len(df) == 0:
        continue

    feats = df[feat_cols].values.astype(np.float32)
    cpc = (df['cpc_bin'] == 'poor').astype(np.float32).values
    rel = df['rel_sec'].values.astype(np.float32) if 'rel_sec' in df.columns else np.zeros(len(df), dtype=np.float32)

    if pid in train_set:
        train_arrays.append(feats); train_meta['cpc'].append(cpc); train_meta['rel'].append(rel)
    else:
        test_arrays.append(feats); test_meta['cpc'].append(cpc); test_meta['rel'].append(rel)

    del df, feats
    if fi % 100 == 0:
        print(f"  {fi}/{len(csv_files)}")

train_X = np.concatenate(train_arrays); del train_arrays
test_X = np.concatenate(test_arrays); del test_arrays
train_cpc = np.concatenate(train_meta['cpc']); train_rel = np.concatenate(train_meta['rel'])
test_cpc = np.concatenate(test_meta['cpc']); test_rel = np.concatenate(test_meta['rel'])
del train_meta, test_meta; gc.collect()

# Clean inf/nan with mean
def clean(x):
    x = np.where(np.isinf(x), np.nan, x)
    means = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(means, inds[1])
    return x

train_X = clean(train_X); test_X = clean(test_X)
print(f"Train: {train_X.shape}, Test: {test_X.shape}, Device: {DEVICE}")

# === Subsample for CEBRA fit ===
rng = np.random.RandomState(SEED)
if len(train_X) > SUBSAMPLE:
    idx = np.sort(rng.choice(len(train_X), SUBSAMPLE, replace=False))
    sub_X, sub_cpc, sub_rel = train_X[idx], train_cpc[idx], train_rel[idx]
else:
    sub_X, sub_cpc, sub_rel = train_X, train_cpc, train_rel

LABEL_MODES = {
    'cpc_bin': sub_cpc,
    'cpc_bin+rel_sec': np.column_stack([sub_cpc, sub_rel]),
}

def transform_batched(model, X, bs=500_000):
    if len(X) <= bs:
        return model.transform(X)
    return np.concatenate([model.transform(X[i:i+bs]) for i in range(0, len(X), bs)])

# === Sweep ===
from itertools import product
keys = list(GRID.keys())
combos = list(product(*GRID.values()))
print(f"\n{len(combos)} configs x {len(LABEL_MODES)} label modes = {len(combos)*len(LABEL_MODES)} runs\n")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

for i, vals in enumerate(combos):
    params = dict(zip(keys, vals))
    tag = "_".join(f"{k[:3]}{v}" for k, v in params.items())

    for label_name, labels in LABEL_MODES.items():
        run_tag = f"{tag}__{label_name.replace('+','_')}"
        out_path = f"{OUT_DIR}/{run_tag}.png"

        if Path(out_path).exists():
            print(f"[{i+1}/{len(combos)}] SKIP {run_tag}")
            continue

        print(f"[{i+1}/{len(combos)}] {run_tag}")
        try:
            np.random.seed(SEED); torch.manual_seed(SEED)
            model = CEBRA(model_architecture='offset10-model', time_offsets=10,
                          device=DEVICE, verbose=False, **params)
            model.fit(sub_X, labels)

            train_emb = transform_batched(model, train_X)
            test_emb = transform_batched(model, test_X)

            title = f"{label_name} | " + " ".join(f"{k}={v}" for k, v in params.items())
            proj = {'projection': '3d'} if params['output_dimension'] >= 3 else {}
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), subplot_kw=proj)

            for ax, emb, lbl, name in [(ax1, train_emb, train_cpc, 'Train'), (ax2, test_emb, test_cpc, 'Test')]:
                n = min(len(emb), PLOT_POINTS)
                si = rng.choice(len(emb), n, replace=False)
                e, l = emb[si], lbl[si]
                if params['output_dimension'] >= 3:
                    ax.scatter(e[:,0], e[:,1], e[:,2], c=l, cmap='RdYlGn', s=1, alpha=0.5)
                else:
                    ax.scatter(e[:,0], e[:,1], c=l, cmap='RdYlGn', s=1, alpha=0.5)
                ax.set_title(name)

            fig.suptitle(title, fontsize=9)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"  -> {out_path}")
        except Exception as ex:
            print(f"  FAILED: {ex}")

        del model; gc.collect(); torch.cuda.empty_cache()

print("\nDone! Check", OUT_DIR)
