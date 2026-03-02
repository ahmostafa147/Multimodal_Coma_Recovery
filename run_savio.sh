#!/bin/bash
#SBATCH --job-name=cebra_pipeline
#SBATCH --account=ic_engin296f25
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:A40:1
#SBATCH --time=04:00:00
#SBATCH --output=pipeline_%j.out
#SBATCH --error=pipeline_%j.err

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "GPU info:"
nvidia-smi

# Paths
WORK_DIR="/global/scratch/users/ahmostafa/CEBRA_modeling_local/Multimodal_Coma_Recovery"
cd $WORK_DIR

# Load conda environment
module load anaconda3/2024.02-1-11.4
source activate $HOME/envs/cebra

echo "Python: $(which python)"
echo "PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run pipeline — pass any extra args (e.g. --stages train predict)
# Usage:
#   sbatch run_savio.sh                              # run all stages
#   sbatch run_savio.sh --stages train predict       # run specific stages
#   sbatch run_savio.sh --stages visualize animate   # just viz
echo "=== Running Pipeline ==="
python run_pipeline.py --config config.json $@

echo "=== Pipeline Complete ==="
echo "Outputs:"
ls -lh models/cebra_model.pt 2>/dev/null
ls -lh data/*.npz 2>/dev/null
ls -lh evaluation/ 2>/dev/null
ls -lh visualizations/ 2>/dev/null

echo "Job finished at $(date)"
