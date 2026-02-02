<<<<<<< HEAD
# ICARE CEBRA Pipeline

## Setup

**Docker:**
```bash
make build
```

**Local (without Docker):**
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

**Notebook:**
```bash
# Docker
make notebook

# Local
jupyter notebook pipeline.ipynb
```

**Command Line:**
```bash
make prepare    # Prepare train/test splits
make train      # Train CEBRA model
make evaluate   # Evaluate on test set
make visualize  # Plot embeddings
```

## Pipeline

**1. Data Preparation**
- Merges neural data with clinical labels (CPC)
- Stratified train/test split by cpc_bin
- Ensures patients only in train OR test

**2. Training**
- CEBRA-Behavior with CPC labels
- Default: 5000 iterations, 8 dimensions

**3. Evaluation**
- KNN decoding performance
- Train/test accuracy and AUC

**4. Visualization**
- Train/test embedding plots
- Colored by CPC labels

## Custom Usage

```bash
# Train with continuous CPC
docker run --rm -v "$(pwd)":/app icare-cebra \
  python scripts/train.py --label cpc --output models/cebra_cpc.pt

# Tune hyperparameters
docker run --rm -v "$(pwd)":/app icare-cebra \
  python scripts/tune.py --label cpc_bin --n-runs 5
```
=======
# Multimodal_Coma_Recovery
>>>>>>> 98bb3078f0a4f4dabbeefcaa6cbab4c172aef723
