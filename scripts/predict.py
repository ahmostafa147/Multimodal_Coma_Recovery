#!/usr/bin/env python3
"""Predict CPC outcomes using CatBoost on CEBRA embeddings"""
import argparse
import numpy as np
import json
from pathlib import Path
from cebra import CEBRA
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix


def predict_model(model_path, train_data, test_data, label_key='cpc_bin',
                  catboost_config=None, output_dir='evaluation'):
    """
    Train CatBoost classifier on CEBRA embeddings and evaluate.

    Args:
        model_path: path to trained CEBRA model
        train_data: dict or NpzFile with 'neural' and label_key
        test_data: dict or NpzFile with 'neural' and label_key
        label_key: label key for classification
        catboost_config: dict with CatBoost params
        output_dir: directory to save CatBoost model and results

    Returns:
        results dict with metrics, trained CatBoost model
    """
    from scripts.train import transform_batched
    model = CEBRA.load(model_path)
    train_emb = transform_batched(model, train_data['neural'])
    test_emb = transform_batched(model, test_data['neural'])

    train_labels = np.array(train_data[label_key])
    test_labels = np.array(test_data[label_key])

    # CatBoost params
    params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'verbose': 100,
        'random_seed': 42,
        'eval_metric': 'Logloss',
        'task_type': 'GPU',
    }
    if catboost_config:
        params.update(catboost_config)

    # Fall back to CPU if GPU not available
    try:
        clf = CatBoostClassifier(**params)
        clf.fit(train_emb, train_labels, eval_set=(test_emb, test_labels), use_best_model=True)
    except Exception:
        print("GPU not available, falling back to CPU...")
        params['task_type'] = 'CPU'
        clf = CatBoostClassifier(**params)
        clf.fit(train_emb, train_labels, eval_set=(test_emb, test_labels), use_best_model=True)

    train_pred = clf.predict(train_emb).flatten()
    test_pred = clf.predict(test_emb).flatten()
    train_proba = clf.predict_proba(train_emb)[:, 1]
    test_proba = clf.predict_proba(test_emb)[:, 1]

    cm = confusion_matrix(test_labels, test_pred)

    results = {
        'task': 'classification',
        'classifier': 'CatBoost',
        'train_accuracy': float(accuracy_score(train_labels, train_pred)),
        'test_accuracy': float(accuracy_score(test_labels, test_pred)),
        'train_auc': float(roc_auc_score(train_labels, train_proba)),
        'test_auc': float(roc_auc_score(test_labels, test_proba)),
        'test_f1': float(f1_score(test_labels, test_pred)),
        'confusion_matrix': cm.tolist(),
        'model_path': model_path,
        'train_shape': list(train_emb.shape),
        'test_shape': list(test_emb.shape),
        'catboost_params': {k: v for k, v in params.items() if k != 'verbose'}
    }

    # Save CatBoost model
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    catboost_path = f'{output_dir}/catboost_model.cbm'
    clf.save_model(catboost_path)
    results['catboost_model_path'] = catboost_path

    return results, clf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='CEBRA model .pt file')
    parser.add_argument('--train-data', default='data/train.npz')
    parser.add_argument('--test-data', default='data/test.npz')
    parser.add_argument('--label', default='cpc_bin', choices=['cpc', 'cpc_bin'])
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--output', default='evaluation/results.json')
    args = parser.parse_args()

    # Load CatBoost config
    catboost_config = None
    if Path(args.config).exists():
        with open(args.config) as f:
            catboost_config = json.load(f).get('prediction', {})

    train_data = np.load(args.train_data, allow_pickle=True)
    test_data = np.load(args.test_data, allow_pickle=True)

    output_dir = str(Path(args.output).parent)
    results, clf = predict_model(
        args.model, train_data, test_data, args.label,
        catboost_config=catboost_config, output_dir=output_dir
    )

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults:")
    print(json.dumps(results, indent=2))
    print(f"\n✓ Results: {args.output}")
    print(f"✓ CatBoost model: {results['catboost_model_path']}")
