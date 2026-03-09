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
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err

WORK_DIR="/global/scratch/users/ahmostafa/CEBRA_modeling_local/Multimodal_Coma_Recovery"
cd "$WORK_DIR"
mkdir -p logs

module load anaconda3/2024.02-1-11.4
source activate "$HOME/envs/cebra"

echo "Started: $(date) | Host: $(hostname)"
nvidia-smi

python run_pipeline.py --config config.json "$@"

echo "Finished: $(date)"
