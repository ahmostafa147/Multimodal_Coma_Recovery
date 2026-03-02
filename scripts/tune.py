#!/usr/bin/env python3
"""Hyperparameter tuning for CEBRA models"""
import argparse
import numpy as np
import torch
from cebra import CEBRA
from pathlib import Path
from itertools import product
import json


def tune_hyperparameters(neural_data, labels, param_grid, output_path, n_runs=3,
                         base_config=None):
    """
    Grid search over CEBRA hyperparameters.

    Args:
        neural_data: (n_samples, n_features) array
        labels: label array
        param_grid: dict of param_name -> list of values
        output_path: where to save results JSON
        n_runs: number of runs per config
        base_config: dict of fixed training params
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fixed params from config
    fixed = {
        'model_architecture': 'offset10-model',
        'batch_size': 512,
        'time_offsets': 10,
    }
    if base_config:
        fixed['model_architecture'] = base_config.get('model_architecture', fixed['model_architecture'])
        fixed['batch_size'] = base_config.get('batch_size', fixed['batch_size'])
        fixed['time_offsets'] = base_config.get('time_offsets', fixed['time_offsets'])

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    configs = [dict(zip(keys, v)) for v in product(*values)]

    results = []
    print(f"Testing {len(configs)} configs x {n_runs} runs on {device}")

    for i, config in enumerate(configs):
        print(f"\n[{i + 1}/{len(configs)}] {config}")

        losses = []
        for run in range(n_runs):
            seed = 42 + run
            np.random.seed(seed)
            torch.manual_seed(seed)

            model = CEBRA(
                **fixed,
                device=device,
                verbose=False,
                **config
            )

            model.fit(neural_data, labels)
            loss_history = model.state_dict_.get('loss', [])
            if len(loss_history) > 0:
                losses.append(float(loss_history[-1]))

        avg_loss = np.mean(losses)
        std_loss = np.std(losses)

        results.append({
            'config': config,
            'avg_loss': float(avg_loss),
            'std_loss': float(std_loss)
        })

        print(f"  Loss: {avg_loss:.4f} +/- {std_loss:.4f}")

    results.sort(key=lambda x: x['avg_loss'])

    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results: {output_path}")
    print(f"Best config: {results[0]['config']}")
    print(f"Best loss: {results[0]['avg_loss']:.4f} +/- {results[0]['std_loss']:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--train-data', default='data/train.npz')
    parser.add_argument('--label', choices=['cpc', 'cpc_bin'])
    parser.add_argument('--output', default='tuning/results.json')
    parser.add_argument('--n-runs', type=int)
    args = parser.parse_args()

    config = {}
    training_config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            full_config = json.load(f)
            config = full_config.get('tuning', {})
            training_config = full_config.get('training', {})

    label = args.label or 'cpc_bin'
    n_runs = args.n_runs or config.get('n_runs', 3)
    param_grid = config.get('param_grid', {
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'temperature': [1.0, 1.12, 1.5],
        'output_dimension': [3, 8, 16],
        'max_iterations': [3000, 5000, 10000]
    })

    data = np.load(args.train_data, allow_pickle=True)
    tune_hyperparameters(data['neural'], data[label], param_grid, args.output,
                         n_runs, base_config=training_config)
