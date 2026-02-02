#!/usr/bin/env python3
"""Visualize CEBRA embeddings"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from cebra import CEBRA
from pathlib import Path

def plot_embedding(embedding, labels, title, output_path, label_names=None):
    """Plot 2D or 3D embedding"""
    n_dims = embedding.shape[1]

    if n_dims == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                           c=labels, cmap='RdYlGn', s=1, alpha=0.6)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        plt.colorbar(scatter, ax=ax, label='CPC')
    else:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                           c=labels, cmap='RdYlGn', s=1, alpha=0.6)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        plt.colorbar(scatter, ax=ax, label='CPC', shrink=0.5)

    ax.set_title(title)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_train_test(train_emb, test_emb, train_labels, test_labels, output_path):
    """Plot train and test embeddings side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.scatter(train_emb[:, 0], train_emb[:, 1],
               c=train_labels, cmap='RdYlGn', s=1, alpha=0.6)
    ax1.set_title('Train')
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')

    scatter = ax2.scatter(test_emb[:, 0], test_emb[:, 1],
                         c=test_labels, cmap='RdYlGn', s=1, alpha=0.6)
    ax2.set_title('Test')
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')

    plt.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(scatter, cax=cbar_ax, label='CPC')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--train-data', default='data/train.npz')
    parser.add_argument('--test-data', default='data/test.npz')
    parser.add_argument('--label', default='cpc_bin', choices=['cpc', 'cpc_bin'])
    parser.add_argument('--output-dir', default='visualizations')
    args = parser.parse_args()

    train_data = np.load(args.train_data, allow_pickle=True)
    test_data = np.load(args.test_data, allow_pickle=True)

    model = CEBRA.load(args.model)
    train_emb = model.transform(train_data['neural'])
    test_emb = model.transform(test_data['neural'])

    model_name = Path(args.model).stem

    if train_emb.shape[1] >= 2 and test_emb.shape[1] >= 2:
        plot_train_test(
            train_emb, test_emb,
            train_data[args.label], test_data[args.label],
            f"{args.output_dir}/{model_name}_train_test.png"
        )
    else:
        print("Need at least 2 dimensions for visualization")
