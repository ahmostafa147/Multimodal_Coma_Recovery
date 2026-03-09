"""CEBRA model training and batched transform."""
import numpy as np
import torch
from cebra import CEBRA
from pathlib import Path


def _patient_chunks(patient_ids, n_chunks):
    """Split data into chunks where each chunk has the same fraction of every patient."""
    indices = []
    for pid in np.unique(patient_ids):
        pid_idx = np.where(patient_ids == pid)[0]
        splits = np.array_split(pid_idx, n_chunks)
        for c, s in enumerate(splits):
            while len(indices) <= c:
                indices.append([])
            indices[c].append(s)
    return [np.concatenate(idx) for idx in indices]


def train_cebra(neural, labels, config, seed=42, patient_ids=None):
    """Train CEBRA model. Uses partial_fit for large datasets."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    params = {
        'model_architecture': config.get('model_architecture', 'offset10-model'),
        'batch_size': config.get('batch_size', 512),
        'learning_rate': config.get('learning_rate', 3e-4),
        'temperature': config.get('temperature', 1.0),
        'time_offsets': config.get('time_offsets', 10),
        'output_dimension': config.get('output_dimension', 3),
        'device': device,
        'verbose': True,
    }

    n = len(neural)
    max_iter = config.get('max_iterations', 5000)
    threshold = config.get('partial_fit_threshold', 2_000_000)

    if n > threshold and patient_ids is not None:
        n_chunks = (n + threshold - 1) // threshold
        chunks = _patient_chunks(patient_ids, n_chunks)
        print(f"Partial fit: {len(chunks)} chunks")
        model = CEBRA(**params, max_iterations=max_iter)
        for i, idx in enumerate(chunks):
            print(f"  Chunk {i+1}/{len(chunks)}: {len(idx)} samples")
            model.partial_fit(neural[idx], labels[idx])
    else:
        model = CEBRA(**params, max_iterations=max_iter)
        model.fit(neural, labels)

    print(f"Trained on {device}, {n} samples")
    return model


def transform_batched(model, data, batch_size=500_000):
    """Transform in batches to avoid OOM."""
    if len(data) <= batch_size:
        return model.transform(data)
    parts = [model.transform(data[i:i+batch_size])
             for i in range(0, len(data), batch_size)]
    return np.concatenate(parts)
