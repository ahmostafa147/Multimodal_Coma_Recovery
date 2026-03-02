#!/usr/bin/env python3
"""Train CEBRA models"""
import argparse
import numpy as np
import torch
from cebra import CEBRA
from pathlib import Path


def train_cebra(neural_data, labels, output_path, config=None,
                max_iter=5000, output_dim=8, seed=42):
    """
    Train CEBRA model.

    Args:
        neural_data: (n_samples, n_features) array
        labels: 1D or 2D label array
        output_path: where to save model
        config: dict with full training params (overrides individual args)
        max_iter: training iterations
        output_dim: embedding dimensions
        seed: random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build params from config or defaults
    params = {
        'model_architecture': 'offset10-model',
        'batch_size': 512,
        'learning_rate': 3e-4,
        'temperature': 1.12,
        'max_iterations': max_iter,
        'time_offsets': 10,
        'output_dimension': output_dim,
        'device': device,
        'verbose': True
    }

    if config:
        params['model_architecture'] = config.get('model_architecture', params['model_architecture'])
        params['batch_size'] = config.get('batch_size', params['batch_size'])
        params['learning_rate'] = config.get('learning_rate', params['learning_rate'])
        params['temperature'] = config.get('temperature', params['temperature'])
        params['max_iterations'] = config.get('max_iterations', params['max_iterations'])
        params['time_offsets'] = config.get('time_offsets', params['time_offsets'])
        params['output_dimension'] = config.get('output_dimension', params['output_dimension'])

    model = CEBRA(**params)
    model.fit(neural_data, labels)

    Path(output_path).parent.mkdir(exist_ok=True)
    model.save(output_path)

    embedding = model.transform(neural_data)
    print(f"\n✓ Model: {output_path}")
    print(f"  Embedding: {embedding.shape}, Device: {device}")

    return model, embedding


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--train-data', default='data/train.npz')
    parser.add_argument('--labels')
    parser.add_argument('--output', default='models/cebra_model.pt')
    parser.add_argument('--output-dim', type=int)
    parser.add_argument('--max-iter', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f).get('training', {})

    # CLI args override config
    if args.output_dim:
        config['output_dimension'] = args.output_dim
    if args.max_iter:
        config['max_iterations'] = args.max_iter
    if args.seed:
        config['seed'] = args.seed

    labels_key = args.labels or config.get('labels', 'cpc_bin')
    seed = args.seed or config.get('seed', 42)

    data = np.load(args.train_data, allow_pickle=True)
    neural = data['neural']

    # Support multiple labels
    label_keys = labels_key.split(',')
    if len(label_keys) == 1:
        label_data = data[label_keys[0]]
    else:
        label_data = np.column_stack([data[k] for k in label_keys])

    print(f"Training CEBRA with labels: {labels_key}")
    print(f"Data: {neural.shape}")
    print(f"Labels: {label_data.shape}")

    train_cebra(neural, label_data, args.output, config=config, seed=seed)
