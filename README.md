# Multimodal Coma Recovery — CEBRA EEG Embedding Pipeline

CEBRA contrastive embedding pipeline for EEG state classification using PPNet prototype activations from the ICARE cohort.

## Classes

| ID | Label |
|----|-------|
| 0 | Seizure |
| 1 | LPD |
| 2 | GPD |
| 3 | LRDA |
| 4 | GRDA |
| 5 | Burst Suppression |
| 6 | Continuous |
| 7 | Discontinuous |

## Pipeline

Run scripts in order:

| Script | Description |
|--------|-------------|
| `cebra/01_data_preprocessing.py` | Resample raw PPNet features to 5-min bins, PCA (1275 → 50), StandardScaler. Train/test split at patient level. |
| `cebra/02_cebra_hybrid_training.py` | Train hybrid CEBRA model (discrete label + bidirectional 6-hour time offset). Computes train + test embeddings. |
| `cebra/03_embedding_visualization.py` | Interactive 3D Plotly scatter of train vs test embeddings. |
| `cebra/04_classification_evaluation.py` | Held-out InfoNCE, kNN, logistic regression, confusion matrices, centroid distances, silhouette score. |
| `cebra/05_temporal_analysis.py` | Per-patient centroid similarity over time, prediction entropy, transition matrix, sojourn times, first-passage time, path length. |

## Training Config

| Parameter | Value |
|-----------|-------|
| Embedding dim | 3 |
| Time offset | 72 bins (6 hours, bidirectional) |
| Batch size | 1024 |
| Iterations | 50,000 |
| Temperature | 1.0 |
| Architecture | offset10-model (Conv1d, 32 units) |
| Learning rate | 3e-4 + cosine annealing |

## Data

Expected input: `PPNet_data_train.npz` and `PPNet_data_test.npz` containing:
- `features` (N, 1275) — PPNet prototype activations
- `predictions` (N,) — labels 1-8
- `patient_ids` (N,) — patient identifiers
- `chunks` (N,) — chunk identifiers within patients
- `resolutions` (N,) — time resolution per row (seconds)
- `times` (N,) — timestamps
- `cpc_scores` (N,) — CPC outcome scores
- `activations` (N, D) — raw activations

## Requirements

```
numpy
torch
cebra
scikit-learn
scipy
matplotlib
seaborn
plotly
tqdm
```
