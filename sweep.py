#!/usr/bin/env python3
"""Grid search over CEBRA hyperparameters. Plots embeddings for each config."""
import argparse
import gc
import json
import numpy as np
import torch
from itertools import product
from datetime import datetime
from pathlib import Path
from cebra import CEBRA

from scripts.data import load_patient_ids, load_or_cache
from scripts.model import transform_batched
from scripts.plot import plot_embeddings, save


def main():
    parser = argparse.ArgumentParser(description='CEBRA Hyperparameter Sweep')
    parser.add_argument('--config', default='config.json')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    dc = cfg['data']
    cc = cfg['cebra']
    sc = cfg['sweep']
    grid = sc['param_grid']
    seed = dc.get('seed', 42)
    nan = dc.get('nan_strategy', 'mean')
    nw = dc.get('n_workers', 4)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load data ---
    print("=== Loading Data ===")
    train_ids = load_patient_ids(dc.get('train_ids', 'train_ids.csv'))
    test_ids = load_patient_ids(dc.get('test_ids', 'test_ids.csv'))

    train = load_or_cache(f'data/train_{nan}.npz', dc['neural_dir'], dc['labels_path'],
                          train_ids, nan, nw)
    test = load_or_cache(f'data/test_{nan}.npz', dc['neural_dir'], dc['labels_path'],
                         test_ids, nan, nw)

    train_X, train_y = train['neural'], train['cpc_bin']
    test_X, test_y = test['neural'], test['cpc_bin']
    print(f"Train: {train_X.shape}, Test: {test_X.shape}")

    # Subsample training data for fitting
    max_fit = sc.get('max_train_samples', 2_000_000)
    if len(train_X) > max_fit:
        idx = np.sort(np.random.RandomState(seed).choice(len(train_X), max_fit, replace=False))
        fit_X, fit_y = train_X[idx], train_y[idx]
        print(f"Fit subsample: {fit_X.shape}")
    else:
        fit_X, fit_y = train_X, train_y

    # --- Build grid ---
    keys = list(grid.keys())
    combos = list(product(*grid.values()))
    out = Path('output') / f'sweep_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    out.mkdir(parents=True, exist_ok=True)

    max_plot = sc.get('max_plot_points', 50000)
    print(f"\n{len(combos)} configs, device={device}\n")

    results = []
    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        tag = "_".join(f"{k[:3]}{v}" for k, v in params.items())
        print(f"[{i+1}/{len(combos)}] {tag}")

        try:
            np.random.seed(seed)
            torch.manual_seed(seed)

            model = CEBRA(
                model_architecture=cc.get('model_architecture', 'offset10-model'),
                time_offsets=cc.get('time_offsets', 10),
                device=device, verbose=False, **params)
            model.fit(fit_X, fit_y)

            train_emb = transform_batched(model, train_X)
            test_emb = transform_batched(model, test_X)

            for name, emb, lab in [('train', train_emb, train_y), ('test', test_emb, test_y)]:
                fig = plot_embeddings(emb, lab, f'{name} | {tag}', max_plot, seed)
                save(fig, str(out / f'{tag}_{name}.html'))

            loss = model.state_dict_.get('loss', [])
            final_loss = float(loss[-1]) if loss else None
            results.append({'params': params, 'loss': final_loss})
            print(f"  loss={final_loss}")

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({'params': params, 'loss': None, 'error': str(e)})

        del model; gc.collect(); torch.cuda.empty_cache()

    # Save results
    results.sort(key=lambda r: r.get('loss') or float('inf'))
    with open(out / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    if results and results[0].get('loss') is not None:
        print(f"\nBest: {results[0]['params']} (loss={results[0]['loss']:.4f})")
    print(f"Output: {out}")


if __name__ == '__main__':
    main()
