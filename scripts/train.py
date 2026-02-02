#!/usr/bin/env python3
"""Train CEBRA models"""
import argparse
import numpy as np
import torch
from cebra import CEBRA
from pathlib import Path

def train_cebra(neural_data, labels, output_path, max_iter=5000, output_dim=8, seed=42):
    """
    Train CEBRA model

    Args:
        labels: Can be 1D array or 2D array for multiple labels
                - 1D: (n_samples,) single label
                - 2D: (n_samples, n_labels) multiple labels
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CEBRA(
        model_architecture='offset10-model',
        batch_size=512,
        learning_rate=3e-4,
        temperature=1.12,
        max_iterations=max_iter,
        time_offsets=10,
        output_dimension=output_dim,
        device=device,
        verbose=True
    )

    model.fit(neural_data, labels)

    Path(output_path).parent.mkdir(exist_ok=True)
    model.save(output_path)

    embedding = model.transform(neural_data)
    print(f"\n✓ Model: {output_path}")
    print(f"  Embedding: {embedding.shape}, Device: {device}")

    return model, embedding

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', default='data/train.npz')
    parser.add_argument('--labels', default='cpc_bin', help='Comma-separated: cpc,cpc_bin or single: cpc_bin')
    parser.add_argument('--output', default='models/cebra_model.pt')
    parser.add_argument('--output-dim', type=int, default=8)
    parser.add_argument('--max-iter', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data = np.load(args.train_data, allow_pickle=True)
    neural = data['neural']

    # Support multiple labels
    label_keys = args.labels.split(',')
    if len(label_keys) == 1:
        labels = data[label_keys[0]]
    else:
        labels = np.column_stack([data[k] for k in label_keys])

    print(f"Training CEBRA with labels: {args.labels}")
    print(f"Data: {neural.shape}")
    print(f"Labels: {labels.shape}")

    train_cebra(neural, labels, args.output, args.max_iter, args.output_dim, args.seed)
