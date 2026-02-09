#!/bin/bash
#SBATCH --job-name=cebra_pipeline
#SBATCH --account=ic_engin296f25
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --reservation=maint
#SBATCH --output=pipeline_%j.out
#SBATCH --error=pipeline_%j.err

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "GPU info:"
nvidia-smi

# Paths
DATA_DIR="/global/scratch/users/ahmostafa/CEBRA_modeling_local/1000_ICARE_patient_10s_94f_with_spike"
LABELS_DIR="/global/scratch/users/ahmostafa/CEBRA_modeling_local/labels"
WORK_DIR="/global/scratch/users/ahmostafa/CEBRA_modeling_local/github_model"

cd $WORK_DIR

# Load conda
module load anaconda3/2024.02-1-11.4
source activate $HOME/envs/cebra

echo "=== 1. Data Preparation ==="
python scripts/prepare_data.py --config savio_config.json

echo "=== 2. Training CEBRA ==="
python scripts/train.py --config savio_config.json

echo "=== 3. Evaluation ==="
python scripts/evaluate.py \
  --model models/cebra_model.pt \
  --train-data data/train.npz \
  --test-data data/test.npz \
  --label cpc_bin \
  --output evaluation/results.json

echo "=== 4. Visualization ==="
python scripts/visualize.py \
  --model models/cebra_model.pt \
  --train-data data/train.npz \
  --test-data data/test.npz \
  --label cpc_bin \
  --output-dir visualizations

echo "=== 5. Animation ==="
# Get random patient
PATIENT=$(python -c "import numpy as np; data=np.load('data/train.npz', allow_pickle=True); print(np.random.choice(np.unique(data['patient_names'])))")
echo "Creating animation for patient: $PATIENT"

python scripts/animate.py \
  --model models/cebra_model.pt \
  --data data/train.npz \
  --patient "$PATIENT" \
  --duration 120 \
  --output visualizations/trajectory_${PATIENT}.mp4

echo "=== Pipeline Complete ==="
echo "Outputs:"
ls -lh models/cebra_model.pt
ls -lh data/*.npz
ls -lh evaluation/results.json
ls -lh visualizations/

echo "Job finished at $(date)"
