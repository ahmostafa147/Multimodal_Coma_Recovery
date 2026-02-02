# Running on Savio HPC

## Setup (One-time)

**1. Login:**
```bash
ssh YOUR_USERNAME@hpc.brc.berkeley.edu
```

**2. Clone repo:**
```bash
cd $SCRATCH
git clone https://github.com/ahmostafa147/Multimodal_Coma_Recovery.git
cd Multimodal_Coma_Recovery
```

**3. Create conda environment:**
```bash
module load anaconda3/2024.02-1-11.4
conda create -p $HOME/envs/cebra python=3.10 -y
source activate $HOME/envs/cebra
pip install -r requirements.txt
```

**4. Upload data:**
```bash
# On your local machine:
scp -r 1000_ICARE_patient_10s_94f_with_spike YOUR_USERNAME@dtn.brc.berkeley.edu:$SCRATCH/Multimodal_Coma_Recovery/
scp -r labels YOUR_USERNAME@dtn.brc.berkeley.edu:$SCRATCH/Multimodal_Coma_Recovery/
```

## Quick Test

**Interactive session:**
```bash
srun -A YOUR_ACCOUNT -p savio2 --time=00:10:00 --pty bash
module load anaconda3/2024.02-1-11.4
source activate $HOME/envs/cebra
python test.py
```

## Run Pipeline

**1. Create job script** (`run_pipeline.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=cebra_pipeline
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=savio2_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=pipeline_%j.out
#SBATCH --error=pipeline_%j.err

module load anaconda3/2024.02-1-11.4
source activate $HOME/envs/cebra

# Run pipeline
python scripts/prepare_data.py
python scripts/train.py
python scripts/evaluate.py --model models/cebra_model.pt
```

**2. Submit job:**
```bash
sbatch run_pipeline.sh
```

**3. Check status:**
```bash
squeue -u $USER
```

**4. View output:**
```bash
cat pipeline_JOBID.out
```

## GPU Training

**For GPU (faster training):**
- Use partition: `savio2_1080ti` or `savio2_gpu`
- Add: `#SBATCH --gres=gpu:1`
- PyTorch will auto-detect CUDA

**CPU-only:**
- Use partition: `savio2` or `savio3`
- Remove `--gres=gpu:1` line

## Jupyter Notebook

**1. Start Jupyter on Savio:**
- Go to https://ood.brc.berkeley.edu/
- Apps → Jupyter
- Select partition, hours, GPUs
- Launch

**2. Open `pipeline.ipynb`**

## Resources

- Partitions: `savio2` (24 cores/node), `savio2_gpu` (4 GPUs/node), `savio3` (32 cores/node)
- Storage: `$HOME` (10GB), `$SCRATCH` (unlimited, purged after 30 days)
- Check limits: `check_usage.sh`

## Troubleshooting

**Out of memory:** Reduce batch_size in config.json
**Module not found:** Check `pip list` in activated environment
**CUDA error:** Use CPU partition or check GPU availability with `nvidia-smi`

## Sources
- [Savio Documentation](https://docs-research-it.berkeley.edu/services/high-performance-computing/)
- [Python on Savio](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/software/using-software/using-python-savio/)
- [SLURM Examples](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/scheduler-examples/)
