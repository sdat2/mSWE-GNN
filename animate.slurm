#!/bin/bash

# --- SLURM Directives (GPU Job for ORCHID) ---
#SBATCH --job-name=animate
#SBATCH --account=orchid             # Use ORCHID account
#SBATCH --partition=orchid           # Use ORCHID partition
#SBATCH --qos=orchid                 # Use ORCHID QoS
#SBATCH --gres=gpu:1                 # Request 1 GPU (A100)
#SBATCH --exclude=gpuhost007         # Exclude problematic host (it seems to have some cuda issue)
#SBATCH -o log/%j.out                # Standard output log
#SBATCH -e log/%j.err                # Standard error log
#SBATCH --time=01:00:00              # Set to 1 hour max walltime
#SBATCH --ntasks-per-node=8          # Number of CPU tasks per node
#SBATCH --mem=40000                  # Memory request in MB (for data loaders)


# --- Your job commands ---
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# --- 1. Environment Setup ---
echo "Loading conda environment..."
source ~/.bashrc
micromamba activate mswegnn-gpu # Activate the new GPU environment

# Add cluster's CUDA to path, as per JASMIN docs
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}

echo "Checking for GPU..."
nvidia-smi # Print GPU status to the log

# --- 2. Data Setup ---
# Define permanent paths

# --- 3. Run Executable ---
echo "Animating..."
# ---
# --- THIS COMMAND IS UPDATED FOR THE NEW HYDRA CONFIG ---
# ---
python -m mswegnn.utils.adforce_predict_animate \
     -c /home/users/sithom/mSWE-GNN/best_config.yaml \
     -ckpt /home/users/sithom/my_results/checkpoints/GNN-epoch=99-val_loss=0.3644.ckpt \
     -nc /home/users/sithom/swegnn_5sec/152_KATRINA_2005.nc \
     -r -1

echo "Job completed."

