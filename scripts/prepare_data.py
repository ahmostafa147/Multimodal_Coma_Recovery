#!/usr/bin/env python3
"""Prepare data with labels and train/test split"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_and_merge_data(neural_dir, labels_path):
    """Load neural data and merge with clinical labels"""
    # Load neural data
    data_path = Path(neural_dir)
    csv_files = sorted(data_path.glob("*.csv"))

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        dfs.append(df)

    neural_df = pd.concat(dfs, ignore_index=True)

    # Load labels
    labels_df = pd.read_csv(labels_path)

    # Merge on patient ID
    merged = neural_df.merge(labels_df, left_on='patient', right_on='pat_ICARE', how='inner')

    print(f"Neural data: {len(neural_df)} samples")
    print(f"Labels: {len(labels_df)} patients")
    print(f"Merged: {len(merged)} samples")

    return merged

def prepare_train_test_split(merged_df, test_size=0.2, seed=42, nan_strategy='drop'):
    """
    Split data at patient level, stratified by cpc_bin

    Args:
        merged_df: Merged dataframe with neural data and labels
        test_size: Fraction for test set
        seed: Random seed
        nan_strategy: 'drop', 'mean', 'median', 'zero', or 'ffill'

    Returns:
        train_data, test_data dictionaries
    """
    # Get unique patients with their cpc_bin
    patient_labels = merged_df[['patient', 'cpc_bin']].drop_duplicates()
    n_patients = len(patient_labels)

    # Adjust test_size for small datasets
    min_test_patients = 2  # Minimum for stratification
    if n_patients * test_size < min_test_patients:
        test_size = min_test_patients / n_patients
        print(f"Adjusted test_size to {test_size:.2f} ({min_test_patients} patients)")

    # Split patients stratified by cpc_bin
    train_patients, test_patients = train_test_split(
        patient_labels['patient'].values,
        test_size=test_size,
        stratify=patient_labels['cpc_bin'].values,
        random_state=seed
    )

    # Split data based on patient assignment
    train_df = merged_df[merged_df['patient'].isin(train_patients)]
    test_df = merged_df[merged_df['patient'].isin(test_patients)]

    # Extract features and labels
    meta_cols = ['patient', 'seg_no', 'file', 'row_in_seg', 'rel_sec',
                 'pat_ICARE', 'cpc', 'cpc_bin', 'ROSC(minutes)', 'age', 'sex', 'vfib', 'time_to_CA(seconds)']
    feature_cols = [col for col in merged_df.columns if col not in meta_cols]

    def extract_data(df):
        neural = df[feature_cols].values.astype(np.float32)

        # Handle NaN values
        nan_count = np.isnan(neural).sum()
        if nan_count > 0:
            print(f"  NaN values: {nan_count} ({nan_count/neural.size*100:.2f}%)")
            if nan_strategy == 'drop':
                mask = ~np.isnan(neural).any(axis=1)
                df = df[mask].reset_index(drop=True)
                neural = df[feature_cols].values.astype(np.float32)
                print(f"  Dropped {(~mask).sum()} samples")
            elif nan_strategy == 'mean':
                neural = np.where(np.isnan(neural), np.nanmean(neural, axis=0), neural)
            elif nan_strategy == 'median':
                neural = np.where(np.isnan(neural), np.nanmedian(neural, axis=0), neural)
            elif nan_strategy == 'zero':
                neural = np.nan_to_num(neural, nan=0.0)
            elif nan_strategy == 'ffill':
                neural = pd.DataFrame(neural).fillna(method='ffill').fillna(0).values.astype(np.float32)

        # Convert cpc_bin to numeric
        cpc_bin = (df['cpc_bin'] == 'poor').astype(np.int32).values
        cpc = df['cpc'].values.astype(np.float32)

        # Map patients to numeric IDs
        unique_patients = sorted(df['patient'].unique())
        patient_map = {pid: idx for idx, pid in enumerate(unique_patients)}
        patient_ids = df['patient'].map(patient_map).values.astype(np.int32)

        return {
            'neural': neural,
            'cpc': cpc,
            'cpc_bin': cpc_bin,
            'patient_ids': patient_ids,
            'patient_names': df['patient'].values,
            'feature_names': feature_cols
        }

    train_data = extract_data(train_df)
    test_data = extract_data(test_df)

    print(f"\nTrain: {len(train_df)} samples, {len(np.unique(train_data['patient_ids']))} patients")
    print(f"  cpc_bin distribution: {np.bincount(train_data['cpc_bin'])}")
    print(f"Test: {len(test_df)} samples, {len(np.unique(test_data['patient_ids']))} patients")
    print(f"  cpc_bin distribution: {np.bincount(test_data['cpc_bin'])}")

    # Verify no patient overlap
    train_patients_set = set(train_data['patient_names'])
    test_patients_set = set(test_data['patient_names'])
    assert len(train_patients_set & test_patients_set) == 0, "Patient overlap detected!"

    return train_data, test_data

def save_splits(train_data, test_data, output_dir='data'):
    """Save train/test splits"""
    Path(output_dir).mkdir(exist_ok=True)

    np.savez_compressed(
        f'{output_dir}/train.npz',
        neural=train_data['neural'],
        cpc=train_data['cpc'],
        cpc_bin=train_data['cpc_bin'],
        patient_ids=train_data['patient_ids'],
        patient_names=train_data['patient_names'],
        feature_names=train_data['feature_names']
    )

    np.savez_compressed(
        f'{output_dir}/test.npz',
        neural=test_data['neural'],
        cpc=test_data['cpc'],
        cpc_bin=test_data['cpc_bin'],
        patient_ids=test_data['patient_ids'],
        patient_names=test_data['patient_names'],
        feature_names=test_data['feature_names']
    )

    print(f"\n✓ Saved: {output_dir}/train.npz")
    print(f"✓ Saved: {output_dir}/test.npz")

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--nan-strategy', choices=['drop', 'mean', 'median', 'zero', 'ffill'])
    args = parser.parse_args()

    # Load config
    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f)['data']

    nan_strategy = args.nan_strategy or config.get('nan_strategy', 'mean')
    neural_dir = config.get('neural_dir', '1000_ICARE_patient_10s_94f_with_spike')
    labels_path = config.get('labels_path', 'labels/ICARE_clinical.csv')
    test_size = config.get('test_size', 0.2)
    seed = config.get('seed', 42)

    merged = load_and_merge_data(neural_dir, labels_path)
    train_data, test_data = prepare_train_test_split(merged, test_size, seed, nan_strategy)
    save_splits(train_data, test_data)
