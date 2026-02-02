# ICARE CEBRA Pipeline
CPC prediction from neural data using CEBRA embeddings

## Quick Start

**Test pipeline (30 sec):**
```bash
make test
```

**Run on test data:**
```bash
python scripts/prepare_data.py --config test_config.json
python scripts/train.py --config test_config.json
python scripts/evaluate.py --model models/cebra_model.pt
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

## Data Structure
```
├── 1000_ICARE_patient_10s_94f_with_spike/  # Your data (not in repo)
│   └── ICARE_*.csv (95 features + metadata)
├── labels/
│   └── ICARE_clinical.csv (CPC labels)
└── test_data/  # Dummy data (included, 4 patients, 400 samples)
    ├── TEST_*.csv
    └── labels/test_clinical.csv
```

## Pipeline

**1. Prepare:** Merge neural + labels, handle NaN, stratified train/test split
**2. Train:** CEBRA model with CPC labels (single or multi-label)
**3. Evaluate:** KNN classification (accuracy, AUC)
**4. Visualize:** 2D/3D embedding plots
**5. Tune:** Hyperparameter grid search

## Run

**Interactive:**
```bash
make notebook  # Opens Jupyter
```

**CLI:**
```bash
make prepare    # Split data (patient-level, stratified by cpc_bin)
make train      # Train CEBRA (5000 iter, 8 dims)
make evaluate   # Test accuracy
make visualize  # Plot embeddings
```

## Configuration

**Use config.json:**
```bash
# Edit config.json, then:
python scripts/prepare_data.py
python scripts/train.py
```

**CLI overrides:**
```bash
python scripts/train.py --labels cpc,cpc_bin --max-iter 10000
python scripts/prepare_data.py --nan-strategy median
```

**Multiple labels:**
```json
// config.json
"training": {
  "labels": "cpc,cpc_bin"  // Train with both
}
```
