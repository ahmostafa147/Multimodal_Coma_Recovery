#!/usr/bin/env python3
"""Evaluate CEBRA models"""
import argparse
import numpy as np
from cebra import CEBRA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import json
from pathlib import Path

def evaluate_model(model_path, train_data, test_data, label_key='cpc_bin'):
    """Evaluate trained model on train and test sets"""
    model = CEBRA.load(model_path)

    train_emb = model.transform(train_data['neural'])
    test_emb = model.transform(test_data['neural'])

    train_labels = train_data[label_key]
    test_labels = test_data[label_key]

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(train_emb, train_labels)

    train_pred = clf.predict(train_emb)
    test_pred = clf.predict(test_emb)

    results = {
        'task': 'classification',
        'train_accuracy': float(accuracy_score(train_labels, train_pred)),
        'test_accuracy': float(accuracy_score(test_labels, test_pred)),
        'model_path': model_path,
        'train_shape': train_emb.shape,
        'test_shape': test_emb.shape
    }

    # Add AUC for binary classification
    if len(np.unique(train_labels)) == 2:
        train_proba = clf.predict_proba(train_emb)[:, 1]
        test_proba = clf.predict_proba(test_emb)[:, 1]
        results['train_auc'] = float(roc_auc_score(train_labels, train_proba))
        results['test_auc'] = float(roc_auc_score(test_labels, test_proba))

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--train-data', default='data/train.npz')
    parser.add_argument('--test-data', default='data/test.npz')
    parser.add_argument('--label', default='cpc_bin', choices=['cpc', 'cpc_bin'])
    parser.add_argument('--output', default='evaluation/results.json')
    args = parser.parse_args()

    train_data = np.load(args.train_data, allow_pickle=True)
    test_data = np.load(args.test_data, allow_pickle=True)

    results = evaluate_model(args.model, train_data, test_data, args.label)

    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\n✓ Results: {args.output}")
