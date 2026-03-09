#!/usr/bin/env python3
"""Full CEBRA pipeline: load -> preprocess -> train -> transform -> plot."""
import argparse
import gc
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from scripts.data import load_patient_ids, load_or_cache
from scripts.model import train_cebra, transform_batched
from scripts.plot import plot_embeddings, plot_trajectory, save


def main():
    parser = argparse.ArgumentParser(description='CEBRA Pipeline')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--run-name', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    run_name = args.run_name or datetime.now().strftime('run_%Y%m%d_%H%M%S')
    out = Path('output') / run_name
    out.mkdir(parents=True, exist_ok=True)

    # Snapshot config
    with open(out / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    dc = cfg['data']
    cc = cfg['cebra']
    seed = dc.get('seed', 42)
    nan = dc.get('nan_strategy', 'mean')
    nw = dc.get('n_workers', 4)

    # --- 1. Data ---
    print("=== 1. Data ===")
    train_ids = load_patient_ids(dc.get('train_ids', 'train_ids.csv'))
    test_ids = load_patient_ids(dc.get('test_ids', 'test_ids.csv'))
    print(f"Patients - train: {len(train_ids)}, test: {len(test_ids)}")

    assert len(set(train_ids) & set(test_ids)) == 0, "Train/test patient overlap!"

    train = load_or_cache(f'data/train_{nan}.npz', dc['neural_dir'], dc['labels_path'],
                          train_ids, nan, nw)
    test = load_or_cache(f'data/test_{nan}.npz', dc['neural_dir'], dc['labels_path'],
                         test_ids, nan, nw)
    print(f"Train: {train['neural'].shape}, Test: {test['neural'].shape}")

    # --- 2. Train CEBRA ---
    print("\n=== 2. Train CEBRA ===")
    model = train_cebra(train['neural'], train['cpc_bin'], cc, seed,
                        patient_ids=train['patient_ids'])
    model_path = str(out / 'model.pt')
    model.save(model_path)
    print(f"Model: {model_path}")

    # --- 3. Transform ---
    print("\n=== 3. Transform ===")
    train_emb = transform_batched(model, train['neural'])
    test_emb = transform_batched(model, test['neural'])
    print(f"Embeddings - train: {train_emb.shape}, test: {test_emb.shape}")

    # --- 4. Plot training embeddings ---
    print("\n=== 4. Plots ===")
    fig = plot_embeddings(train_emb, train['cpc_bin'], f'{run_name} - Training Embeddings')
    save(fig, str(out / 'train_embedding.html'))

    # --- 5. Test patient trajectory ---
    np.random.seed(seed)
    test_patients = np.unique(test['patient_names'])
    patient = np.random.choice(test_patients)
    print(f"Test patient: {patient}")

    mask = test['patient_names'] == patient
    p_emb = test_emb[mask]
    p_time = test['rel_sec'][mask]
    order = np.argsort(p_time)
    p_emb = p_emb[order]

    fig = plot_trajectory(train_emb, train['cpc_bin'], p_emb, patient)
    save(fig, str(out / f'trajectory_{patient}.html'))

    del model, train_emb, test_emb, train, test; gc.collect()
    print(f"\n=== Done: {out} ===")


if __name__ == '__main__':
    main()
