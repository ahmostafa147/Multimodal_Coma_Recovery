#!/usr/bin/env python3
"""Sweep CEBRA hyperparameters and save embedding plots."""
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cebra import CEBRA
from pathlib import Path
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
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

# === Load data ===
print("Loading labels...")
labels_df = pd.read_csv(LABELS_PATH)
csv_files = sorted(Path(NEURAL_DIR).glob("*.csv"))
print(f"Found {len(csv_files)} CSVs")

meta_cols = {'patient', 'seg_no', 'file', 'row_in_seg', 'rel_sec',
             'pat_ICARE', 'cpc', 'cpc_bin', 'ROSC(minutes)', 'age',
             'sex', 'vfib', 'time_to_CA(seconds)'}

def read_csv(f):
    df = pd.read_csv(f, encoding='utf-8-sig')
    return df.merge(labels_df, left_on='patient', right_on='pat_ICARE', how='inner')

print("Reading and merging CSVs...")
with ThreadPoolExecutor(max_workers=4) as ex:
    dfs = list(ex.map(read_csv, csv_files))
merged = pd.concat([d for d in dfs if len(d) > 0], ignore_index=True)
del dfs; gc.collect()
print(f"Merged: {len(merged)} samples")

merged = merged.dropna(subset=['cpc_bin']).reset_index(drop=True)
print(f"After dropping NaN labels: {len(merged)} samples")

feat_cols = [c for c in merged.columns if c not in meta_cols]
neural = merged[feat_cols].values.astype(np.float32)
neural = np.where(np.isinf(neural), np.nan, neural)
col_means = np.nanmean(neural, axis=0)
inds = np.where(np.isnan(neural))
neural[inds] = np.take(col_means, inds[1])

cpc_bin = (merged['cpc_bin'] == 'poor').astype(np.float32).values
rel_sec = merged['rel_sec'].values.astype(np.float32)
patients = merged['patient'].values

# === Patient-level split ===
pat_labels = merged[['patient', 'cpc_bin']].drop_duplicates()
train_pats, test_pats = train_test_split(
    pat_labels['patient'].values, test_size=0.2,
    stratify=pat_labels['cpc_bin'].values, random_state=SEED)

train_mask = np.isin(patients, train_pats)
test_mask = np.isin(patients, test_pats)
train_X, test_X = neural[train_mask], neural[test_mask]
train_cpc, test_cpc = cpc_bin[train_mask], cpc_bin[test_mask]
train_rel, test_rel = rel_sec[train_mask], rel_sec[test_mask]
del merged, neural; gc.collect()

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
