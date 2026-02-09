# Running Pipeline on Savio

## Steps

**1. SSH to Savio:**
```bash
ssh YOUR_USERNAME@hpc.brc.berkeley.edu
```

**2. Navigate to scratch:**
```bash
cd /global/scratch/users/ahmostafa/CEBRA_modeling_local
```

**3. Clone repo:**
```bash
git clone https://github.com/ahmostafa147/Multimodal_Coma_Recovery.git github_model
cd github_model
```

**4. Setup environment (one-time):**
```bash
module load anaconda3/2024.02-1-11.4
conda create -p $HOME/envs/cebra python=3.10 -y
source activate $HOME/envs/cebra
pip install -r requirements.txt
```

**5. Make sure data is in place:**
```bash
ls /global/scratch/users/ahmostafa/CEBRA_modeling_local/1000_ICARE_patient_10s_94f_with_spike
ls /global/scratch/users/ahmostafa/CEBRA_modeling_local/labels/ICARE_clinical.csv
```

**6. Submit job:**
```bash
sbatch run_savio.sh
```

**7. Check status:**
```bash
squeue -u $USER
```

**8. Monitor output:**
```bash
tail -f pipeline_JOBID.out
```

## Outputs

After ~2 hours:
```
models/cebra_model.pt                    # Trained model
data/train.npz                           # Train split
data/test.npz                            # Test split
evaluation/results.json                  # Test accuracy & AUC
visualizations/train_test_comparison.png # Embedding plots
visualizations/trajectory_PATIENT.mp4    # 2-min animation
```

## Download Results

**From Savio:**
```bash
# On your local machine:
scp -r YOUR_USERNAME@dtn.brc.berkeley.edu:/global/scratch/users/ahmostafa/CEBRA_modeling_local/github_model/models .
scp -r YOUR_USERNAME@dtn.brc.berkeley.edu:/global/scratch/users/ahmostafa/CEBRA_modeling_local/github_model/visualizations .
```

## Troubleshooting

**Job fails:** Check `pipeline_JOBID.err`
**Out of memory:** Reduce batch_size in savio_config.json
**GPU not detected:** Check `nvidia-smi` output in .out file
