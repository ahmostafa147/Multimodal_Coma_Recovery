#!/bin/bash
#SBATCH --job-name=cebra_sweep
#SBATCH --account=ic_engin296f25
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --time=12:00:00
#SBATCH --output=sweep_%j.out
#SBATCH --error=sweep_%j.err

cd /global/scratch/users/ahmostafa/CEBRA_modeling_local/Multimodal_Coma_Recovery
module load anaconda3/2024.02-1-11.4
source activate $HOME/envs/cebra

echo "Started: $(date)"
nvidia-smi
python sweep.py
echo "Finished: $(date)"
