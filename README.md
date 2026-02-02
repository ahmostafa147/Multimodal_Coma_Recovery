# ICARE CEBRA Pipeline
CPC prediction from neural data using CEBRA embeddings

## Data Setup
```
10_w_spikes/
├── 1000_ICARE_patient_10s_94f_with_spike/
│   ├── ICARE_0001_rel10s_with_spike.csv
│   ├── ICARE_0002_rel10s_with_spike.csv
│   ├── ICARE_0003_rel10s_with_spike.csv
│   └── ICARE_0004_rel10s_with_spike.csv
└── labels/
    └── ICARE_clinical.csv
```

## Setup

**Docker:**
```bash
make build
```

**Local:**
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

**Notebook:**
```bash
make notebook          # Docker
jupyter notebook       # Local
```

**CLI:**
```bash
make prepare    # Merge data, split train/test (stratified by cpc_bin, patient-level)
make train      # Train CEBRA (5000 iter, 8 dims, cpc_bin labels)
make evaluate   # KNN classification (accuracy, AUC)
make visualize  # Plot embeddings
```

## Scripts

- `prepare_data.py` - Merge neural + labels, handle NaN (mean imputation), split train/test
- `train.py` - Train CEBRA model (supports single or multiple labels)
- `evaluate.py` - KNN classification performance
- `visualize.py` - 2D/3D embedding plots
- `tune.py` - Hyperparameter grid search

## Custom Training

**Multiple labels:**
```bash
python scripts/train.py --labels cpc,cpc_bin --output models/multi.pt
```

**Different NaN strategy:**
```bash
python scripts/prepare_data.py --nan-strategy median  # drop, mean, median, zero, ffill
```

**Custom hyperparameters:**
```bash
python scripts/train.py --output-dim 3 --max-iter 10000 --labels cpc
```
