#!/usr/bin/env python3
"""Test all pipeline functionality on dummy data"""
import sys
import numpy as np
from pathlib import Path


def test_prepare():
    from scripts.prepare_data import load_and_merge_data, prepare_train_test_split, save_splits

    merged = load_and_merge_data("test_data", "test_data/labels/test_clinical.csv")
    train, test = prepare_train_test_split(merged, test_size=0.5, seed=42, nan_strategy='mean')

    np.savez_compressed('data/test_train.npz', **train)
    np.savez_compressed('data/test_test.npz', **test)
    print("✓ Data preparation")


def test_train():
    from scripts.train import train_cebra

    data = np.load('data/test_train.npz', allow_pickle=True)
    train_cebra(data['neural'], data['cpc_bin'], 'models/test_model.pt',
                max_iter=30, output_dim=3, seed=42)
    print("✓ Training")


def test_predict():
    from scripts.predict import predict_model

    train = np.load('data/test_train.npz', allow_pickle=True)
    test = np.load('data/test_test.npz', allow_pickle=True)

    results, clf = predict_model(
        'models/test_model.pt', train, test, 'cpc_bin',
        catboost_config={'task_type': 'CPU', 'iterations': 50, 'verbose': 0},
        output_dir='evaluation'
    )
    print(f"✓ Prediction (acc: {results['test_accuracy']:.2f}, auc: {results['test_auc']:.2f})")
    return clf


def test_visualize():
    from scripts.visualize import plot_train_test, plot_embedding
    from cebra import CEBRA

    train = np.load('data/test_train.npz', allow_pickle=True)
    test = np.load('data/test_test.npz', allow_pickle=True)
    model = CEBRA.load('models/test_model.pt')

    train_emb = model.transform(train['neural'])
    test_emb = model.transform(test['neural'])

    plot_embedding(train_emb, train['cpc_bin'], 'Test Train', 'visualizations/test_train.png')
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


def test_animate():
    from scripts.animate import create_animation
    from cebra import CEBRA

    data = np.load('data/test_train.npz', allow_pickle=True)
    model = CEBRA.load('models/test_model.pt')
    embedding = model.transform(data['neural'])

    patients = np.unique(data['patient_names'])
    highlight = patients[0]

    create_animation(
        embedding=embedding,
        labels=data['cpc_bin'],
        patient_ids=data['patient_names'],
        rel_sec=data['rel_sec'],
        highlight_patient=highlight,
        output_path='visualizations/test_trajectory.mp4',
        duration=5,
        fps=10
    )
    assert Path('visualizations/test_trajectory.mp4').exists(), "Animation not saved"
    print("✓ Animation")


if __name__ == "__main__":
    try:
        test_prepare()
        test_train()
        test_predict()
        test_visualize()
        test_tune()
        test_animate()
        print("\n✓ All tests passed")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
