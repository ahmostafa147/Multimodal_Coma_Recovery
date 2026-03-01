#!/usr/bin/env python3
"""Visualize CEBRA embeddings in 2D and 3D"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from cebra import CEBRA
from pathlib import Path


def plot_embedding(embedding, labels, title, output_path, label_names=None):
    """Plot 2D or 3D embedding colored by labels"""
    n_dims = embedding.shape[1]

    if n_dims >= 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                             c=labels, cmap='RdYlGn', s=1, alpha=0.6)
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.set_zlabel('Dim 3')
        plt.colorbar(scatter, ax=ax, label='CPC', shrink=0.5, pad=0.1)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                             c=labels, cmap='RdYlGn', s=1, alpha=0.6)
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        plt.colorbar(scatter, ax=ax, label='CPC')

    ax.set_title(title)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_train_test(train_emb, test_emb, train_labels, test_labels, output_path):
    """Plot train and test embeddings side by side (2D or 3D)"""
    n_dims = min(train_emb.shape[1], test_emb.shape[1])
    use_3d = n_dims >= 3

    if use_3d:
        fig = plt.figure(figsize=(18, 8))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(train_emb[:, 0], train_emb[:, 1], train_emb[:, 2],
                    c=train_labels, cmap='RdYlGn', s=1, alpha=0.6)
        ax1.set_title('Train Embedding')
        ax1.set_xlabel('Dim 1')
        ax1.set_ylabel('Dim 2')
        ax1.set_zlabel('Dim 3')

        ax2 = fig.add_subplot(122, projection='3d')
        scatter = ax2.scatter(test_emb[:, 0], test_emb[:, 1], test_emb[:, 2],
                              c=test_labels, cmap='RdYlGn', s=1, alpha=0.6)
        ax2.set_title('Test Embedding')
        ax2.set_xlabel('Dim 1')
        ax2.set_ylabel('Dim 2')
        ax2.set_zlabel('Dim 3')

        plt.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        fig.colorbar(scatter, cax=cbar_ax, label='CPC')
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.scatter(train_emb[:, 0], train_emb[:, 1],
                    c=train_labels, cmap='RdYlGn', s=1, alpha=0.6)
        ax1.set_title('Train')
        ax1.set_xlabel('Dim 1')
        ax1.set_ylabel('Dim 2')

        scatter = ax2.scatter(test_emb[:, 0], test_emb[:, 1],
                              c=test_labels, cmap='RdYlGn', s=1, alpha=0.6)
        ax2.set_title('Test')
        ax2.set_xlabel('Dim 1')
        ax2.set_ylabel('Dim 2')

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
    output_dir = args.output_dir

    # Individual plots
    plot_embedding(train_emb, train_data[args.label],
                   'Train Embedding', f"{output_dir}/{model_name}_train.png")
    plot_embedding(test_emb, test_data[args.label],
                   'Test Embedding', f"{output_dir}/{model_name}_test.png")

    # Side-by-side
    if train_emb.shape[1] >= 2:
        plot_train_test(train_emb, test_emb,
                        train_data[args.label], test_data[args.label],
                        f"{output_dir}/{model_name}_train_test.png")
