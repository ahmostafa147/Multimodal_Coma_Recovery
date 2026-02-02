#!/usr/bin/env python3
"""Test pipeline"""
import sys
import numpy as np
from pathlib import Path

def test_data_split():
    """Test train/test split"""
    assert Path('data/train.npz').exists(), "Missing train.npz"
    assert Path('data/test.npz').exists(), "Missing test.npz"

    train = np.load('data/train.npz', allow_pickle=True)
    test = np.load('data/test.npz', allow_pickle=True)

    # Check keys
    required_keys = ['neural', 'cpc', 'cpc_bin', 'patient_ids', 'patient_names', 'feature_names']
    for key in required_keys:
        assert key in train, f"Missing {key} in train"
        assert key in test, f"Missing {key} in test"

    # Check shapes
    assert train['neural'].ndim == 2, "Train neural should be 2D"
    assert test['neural'].ndim == 2, "Test neural should be 2D"
    assert train['neural'].shape[1] == test['neural'].shape[1], "Feature mismatch"

    # Check no patient overlap
    train_patients = set(train['patient_names'])
    test_patients = set(test['patient_names'])
    assert len(train_patients & test_patients) == 0, "Patient overlap detected"

    # Check stratification
    train_cpc_dist = np.bincount(train['cpc_bin'])
    test_cpc_dist = np.bincount(test['cpc_bin'])

    print(f"✓ Data split:")
    print(f"  Train: {train['neural'].shape}, {len(np.unique(train['patient_ids']))} patients")
    print(f"  Test: {test['neural'].shape}, {len(np.unique(test['patient_ids']))} patients")
    print(f"  No patient overlap: {len(train_patients & test_patients) == 0}")

if __name__ == "__main__":
    try:
        test_data_split()
        print("\n✓ All tests passed")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
