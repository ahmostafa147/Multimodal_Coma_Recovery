#!/usr/bin/env python3
"""Test all pipeline functionality"""
import sys
import numpy as np
from pathlib import Path

def test_prepare():
    from scripts.prepare_data import load_and_merge_data, prepare_train_test_split, save_splits

    merged = load_and_merge_data("test_data", "test_data/labels/test_clinical.csv")
    train, test = prepare_train_test_split(merged, test_size=0.5, seed=42, nan_strategy='mean')

    # Save to separate test location
    np.savez_compressed('data/test_train.npz', **train)
    np.savez_compressed('data/test_test.npz', **test)
    print("✓ Data preparation")

def test_train():
    from scripts.train import train_cebra

    data = np.load('data/test_train.npz', allow_pickle=True)
    train_cebra(data['neural'], data['cpc_bin'], 'models/test_model.pt',
                max_iter=30, output_dim=2, seed=42)
    print("✓ Training")

def test_evaluate():
    from scripts.evaluate import evaluate_model

    train = np.load('data/test_train.npz', allow_pickle=True)
    test = np.load('data/test_test.npz', allow_pickle=True)
    results = evaluate_model('models/test_model.pt', train, test, 'cpc_bin')
    print(f"✓ Evaluation (acc: {results['test_accuracy']:.2f})")

def test_visualize():
    from scripts.visualize import plot_train_test
    from cebra import CEBRA

    train = np.load('data/test_train.npz', allow_pickle=True)
    test = np.load('data/test_test.npz', allow_pickle=True)
    model = CEBRA.load('models/test_model.pt')

    train_emb = model.transform(train['neural'])
    test_emb = model.transform(test['neural'])
    plot_train_test(train_emb, test_emb, train['cpc_bin'], test['cpc_bin'],
                    'visualizations/test.png')
    print("✓ Visualization")

def test_tune():
    from scripts.tune import tune_hyperparameters

    data = np.load('data/test_train.npz', allow_pickle=True)
    param_grid = {'learning_rate': [0.001], 'temperature': [1.0],
                  'output_dimension': [2], 'max_iterations': [20]}
    tune_hyperparameters(data['neural'], data['cpc_bin'], param_grid,
                        'tuning/test_results.json', n_runs=1)
    print("✓ Tuning")

if __name__ == "__main__":
    try:
        test_prepare()
        test_train()
        test_evaluate()
        test_visualize()
        test_tune()
        print("\n✓ All tests passed")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
