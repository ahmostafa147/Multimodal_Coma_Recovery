#!/usr/bin/env python3
"""Prepare data with memory-efficient streaming CSV processing"""
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor


def _read_single_csv(csv_path):
    """Read a single CSV and return (patient_id, dataframe)"""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    patient_id = df['patient'].iloc[0] if 'patient' in df.columns else csv_path.stem
    return patient_id, df


def _detect_feature_cols(sample_df):
    """Detect feature columns from a sample dataframe"""
    meta_cols = {'patient', 'seg_no', 'file', 'row_in_seg', 'rel_sec',
                 'pat_ICARE', 'cpc', 'cpc_bin', 'ROSC(minutes)', 'age',
                 'sex', 'vfib', 'time_to_CA(seconds)'}
    return [col for col in sample_df.columns if col not in meta_cols]


def _handle_nans(neural, nan_strategy):
    """Apply NaN/Inf handling strategy to neural array"""
    # Replace inf with NaN first so all strategies handle them
    inf_count = np.isinf(neural).sum()
    if inf_count > 0:
        print(f"  Inf values: {inf_count} ({inf_count / neural.size * 100:.2f}%) — replaced with NaN")
        neural = np.where(np.isinf(neural), np.nan, neural)

    nan_count = np.isnan(neural).sum()
    if nan_count == 0:
        return neural

    print(f"  NaN values: {nan_count} ({nan_count / neural.size * 100:.2f}%)")
    if nan_strategy == 'drop':
        mask = ~np.isnan(neural).any(axis=1)
        neural = neural[mask]
        print(f"  Dropped {(~mask).sum()} samples")
    elif nan_strategy == 'mean':
        col_means = np.nanmean(neural, axis=0)
        inds = np.where(np.isnan(neural))
        neural[inds] = np.take(col_means, inds[1])
    elif nan_strategy == 'median':
        col_medians = np.nanmedian(neural, axis=0)
        inds = np.where(np.isnan(neural))
        neural[inds] = np.take(col_medians, inds[1])
    elif nan_strategy == 'zero':
        neural = np.nan_to_num(neural, nan=0.0)
    elif nan_strategy == 'ffill':
        neural = pd.DataFrame(neural).ffill().fillna(0).values.astype(np.float32)
    return neural


def load_and_merge_data(neural_dir, labels_path):
    """
    Load neural data and merge with clinical labels.
    Legacy function — loads all CSVs into memory at once.
    Use prepare_data_streaming() for large datasets.
    """
    data_path = Path(neural_dir)
    csv_files = sorted(data_path.glob("*.csv"))

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        dfs.append(df)

    neural_df = pd.concat(dfs, ignore_index=True)
    labels_df = pd.read_csv(labels_path)
    merged = neural_df.merge(labels_df, left_on='patient', right_on='pat_ICARE', how='inner')

    print(f"Neural data: {len(neural_df)} samples")
    print(f"Labels: {len(labels_df)} patients")
    print(f"Merged: {len(merged)} samples")

    return merged


def prepare_data_streaming(neural_dir, labels_path, test_size=0.2, seed=42,
                           nan_strategy='mean', n_workers=4):
    """
    Memory-efficient data preparation: stream CSVs one at a time.

    Phase 1: Read labels, discover patients, stratified split
    Phase 2: Stream each CSV, extract features, assign to train or test

    Peak memory: ~4-5GB instead of 60-120GB for 1000 patients.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x

    data_path = Path(neural_dir)
    csv_files = sorted(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {neural_dir}")

    # --- Phase 1: Build patient→label mapping and split ---
    print(f"Found {len(csv_files)} CSV files")

    labels_df = pd.read_csv(labels_path)
    patient_to_cpc = dict(zip(labels_df['pat_ICARE'], labels_df['cpc']))
    patient_to_cpc_bin = dict(zip(labels_df['pat_ICARE'], labels_df['cpc_bin']))

    # Discover patients from filenames by reading first row of each CSV
    # (faster than reading whole file)
    patient_ids = []
    patient_files = {}
    for csv_file in tqdm(csv_files, desc="Scanning patients"):
        df_head = pd.read_csv(csv_file, encoding='utf-8-sig', nrows=1)
        pid = df_head['patient'].iloc[0]
        if pid in patient_to_cpc_bin:
            patient_ids.append(pid)
            patient_files.setdefault(pid, []).append(csv_file)

    unique_patients = sorted(set(patient_ids))
    patient_cpc_bins = [patient_to_cpc_bin[p] for p in unique_patients]

    print(f"Matched {len(unique_patients)} patients with labels")

    # Adjust test_size for small datasets
    n_patients = len(unique_patients)
    min_test_patients = 2
    if n_patients * test_size < min_test_patients:
        test_size = min_test_patients / n_patients
        print(f"Adjusted test_size to {test_size:.2f} ({min_test_patients} patients)")

    # Stratified patient split
    train_patients, test_patients = train_test_split(
        unique_patients,
        test_size=test_size,
        stratify=patient_cpc_bins,
        random_state=seed
    )
    train_set = set(train_patients)
    test_set = set(test_patients)

    print(f"Train patients: {len(train_patients)}, Test patients: {len(test_patients)}")

    # --- Phase 2: Stream CSVs and extract features ---
    # Read one sample to detect feature columns
    sample_df = pd.read_csv(csv_files[0], encoding='utf-8-sig', nrows=5)
    sample_df = sample_df.merge(labels_df, left_on='patient', right_on='pat_ICARE', how='inner')
    feature_cols = _detect_feature_cols(sample_df)
    print(f"Features: {len(feature_cols)} columns")

    train_arrays = []
    test_arrays = []
    train_meta = {'cpc': [], 'cpc_bin': [], 'patient_names': [], 'rel_sec': []}
    test_meta = {'cpc': [], 'cpc_bin': [], 'patient_names': [], 'rel_sec': []}

    def process_csv(csv_file):
        """Read CSV, merge with labels, return (patient_id, features, meta)"""
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        pid = df['patient'].iloc[0]
        if pid not in patient_to_cpc_bin:
            return None

        df_merged = df.merge(labels_df, left_on='patient', right_on='pat_ICARE', how='inner')
        if len(df_merged) == 0:
            return None

        feats = df_merged[feature_cols].values.astype(np.float32)
        cpc_bin = (df_merged['cpc_bin'] == 'poor').astype(np.int32).values
        cpc = df_merged['cpc'].values.astype(np.float32)
        names = df_merged['patient'].values
        rel_sec = df_merged['rel_sec'].values.astype(np.float32) if 'rel_sec' in df_merged.columns else np.zeros(len(df_merged), dtype=np.float32)

        return pid, feats, cpc, cpc_bin, names, rel_sec

    # Process CSVs with thread pool (I/O bound)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(
            executor.map(process_csv, csv_files),
            total=len(csv_files),
            desc="Processing CSVs"
        ))

    for result in results:
        if result is None:
            continue
        pid, feats, cpc, cpc_bin, names, rel_sec = result

        if pid in train_set:
            train_arrays.append(feats)
            train_meta['cpc'].append(cpc)
            train_meta['cpc_bin'].append(cpc_bin)
            train_meta['patient_names'].append(names)
            train_meta['rel_sec'].append(rel_sec)
        elif pid in test_set:
            test_arrays.append(feats)
            test_meta['cpc'].append(cpc)
            test_meta['cpc_bin'].append(cpc_bin)
            test_meta['patient_names'].append(names)
            test_meta['rel_sec'].append(rel_sec)

    del results
    gc.collect()

    # Concatenate
    train_neural = np.concatenate(train_arrays, axis=0)
    test_neural = np.concatenate(test_arrays, axis=0)
    del train_arrays, test_arrays
    gc.collect()

    # Handle NaNs
    print("Handling NaN values (train)...")
    train_neural = _handle_nans(train_neural, nan_strategy)
    print("Handling NaN values (test)...")
    test_neural = _handle_nans(test_neural, nan_strategy)

    # Build patient ID maps
    train_names = np.concatenate(train_meta['patient_names'])
    test_names = np.concatenate(test_meta['patient_names'])

    def make_patient_ids(names):
        unique = sorted(set(names))
        pmap = {p: i for i, p in enumerate(unique)}
        return np.array([pmap[n] for n in names], dtype=np.int32)

    train_data = {
        'neural': train_neural,
        'cpc': np.concatenate(train_meta['cpc']),
        'cpc_bin': np.concatenate(train_meta['cpc_bin']),
        'patient_ids': make_patient_ids(train_names),
        'patient_names': train_names,
        'rel_sec': np.concatenate(train_meta['rel_sec']),
        'feature_names': feature_cols
    }
    test_data = {
        'neural': test_neural,
        'cpc': np.concatenate(test_meta['cpc']),
        'cpc_bin': np.concatenate(test_meta['cpc_bin']),
        'patient_ids': make_patient_ids(test_names),
        'patient_names': test_names,
        'rel_sec': np.concatenate(test_meta['rel_sec']),
        'feature_names': feature_cols
    }

    del train_meta, test_meta
    gc.collect()

    print(f"\nTrain: {len(train_neural)} samples, {len(train_patients)} patients")
    print(f"  cpc_bin distribution: {np.bincount(train_data['cpc_bin'])}")
    print(f"Test: {len(test_neural)} samples, {len(test_patients)} patients")
    print(f"  cpc_bin distribution: {np.bincount(test_data['cpc_bin'])}")

    # Verify no patient overlap
    assert len(set(train_names) & set(test_names)) == 0, "Patient overlap detected!"

    return train_data, test_data


def prepare_train_test_split(merged_df, test_size=0.2, seed=42, nan_strategy='drop'):
    """
    Split data at patient level, stratified by cpc_bin.
    Legacy function for small datasets or pre-loaded DataFrames.
    """
    patient_labels = merged_df[['patient', 'cpc_bin']].drop_duplicates()
    n_patients = len(patient_labels)

    min_test_patients = 2
    if n_patients * test_size < min_test_patients:
        test_size = min_test_patients / n_patients
        print(f"Adjusted test_size to {test_size:.2f} ({min_test_patients} patients)")

    train_patients, test_patients = train_test_split(
        patient_labels['patient'].values,
        test_size=test_size,
        stratify=patient_labels['cpc_bin'].values,
        random_state=seed
    )

    train_df = merged_df[merged_df['patient'].isin(train_patients)]
    test_df = merged_df[merged_df['patient'].isin(test_patients)]

    feature_cols = _detect_feature_cols(merged_df)

    def extract_data(df):
        neural = df[feature_cols].values.astype(np.float32)
        neural = _handle_nans(neural, nan_strategy)

        cpc_bin = (df['cpc_bin'] == 'poor').astype(np.int32).values
        cpc = df['cpc'].values.astype(np.float32)

        unique_patients = sorted(df['patient'].unique())
        patient_map = {pid: idx for idx, pid in enumerate(unique_patients)}
        patient_ids = df['patient'].map(patient_map).values.astype(np.int32)

        return {
            'neural': neural,
            'cpc': cpc,
            'cpc_bin': cpc_bin,
            'patient_ids': patient_ids,
            'patient_names': df['patient'].values,
            'rel_sec': df['rel_sec'].values.astype(np.float32),
            'feature_names': feature_cols
        }

    train_data = extract_data(train_df)
    test_data = extract_data(test_df)

    print(f"\nTrain: {len(train_df)} samples, {len(np.unique(train_data['patient_ids']))} patients")
    print(f"  cpc_bin distribution: {np.bincount(train_data['cpc_bin'])}")
    print(f"Test: {len(test_df)} samples, {len(np.unique(test_data['patient_ids']))} patients")
    print(f"  cpc_bin distribution: {np.bincount(test_data['cpc_bin'])}")

    train_patients_set = set(train_data['patient_names'])
    test_patients_set = set(test_data['patient_names'])
    assert len(train_patients_set & test_patients_set) == 0, "Patient overlap detected!"

    return train_data, test_data


def save_splits(train_data, test_data, output_dir='data'):
    """Save train/test splits as compressed NPZ"""
    Path(output_dir).mkdir(exist_ok=True)

    np.savez_compressed(
        f'{output_dir}/train.npz',
        neural=train_data['neural'],
        cpc=train_data['cpc'],
        cpc_bin=train_data['cpc_bin'],
        patient_ids=train_data['patient_ids'],
        patient_names=train_data['patient_names'],
        rel_sec=train_data['rel_sec'],
        feature_names=train_data['feature_names']
    )

    np.savez_compressed(
        f'{output_dir}/test.npz',
        neural=test_data['neural'],
        cpc=test_data['cpc'],
        cpc_bin=test_data['cpc_bin'],
        patient_ids=test_data['patient_ids'],
        patient_names=test_data['patient_names'],
        rel_sec=test_data['rel_sec'],
        feature_names=test_data['feature_names']
    )

    print(f"\n✓ Saved: {output_dir}/train.npz")
    print(f"✓ Saved: {output_dir}/test.npz")


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--nan-strategy', choices=['drop', 'mean', 'median', 'zero', 'ffill'])
    parser.add_argument('--streaming', action='store_true', default=True,
                        help='Use memory-efficient streaming (default: True)')
    parser.add_argument('--no-streaming', dest='streaming', action='store_false')
    args = parser.parse_args()

    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f)['data']

    nan_strategy = args.nan_strategy or config.get('nan_strategy', 'mean')
    neural_dir = config.get('neural_dir', '1000_ICARE_patient_10s_94f_with_spike')
    labels_path = config.get('labels_path', 'labels/ICARE_clinical.csv')
    test_size = config.get('test_size', 0.2)
    seed = config.get('seed', 42)
    n_workers = config.get('n_workers', 4)

    if args.streaming:
        train_data, test_data = prepare_data_streaming(
            neural_dir, labels_path, test_size, seed, nan_strategy, n_workers
        )
    else:
        merged = load_and_merge_data(neural_dir, labels_path)
        train_data, test_data = prepare_train_test_split(merged, test_size, seed, nan_strategy)

    save_splits(train_data, test_data)
