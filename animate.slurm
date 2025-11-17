#!/bin/bash

# --- SLURM Directives (GPU Job for ORCHID) ---
#SBATCH --job-name=GCN-GNN-P1-128-bias
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

# --- 2. Data Setup (NEW) ---
# Create a unique output directory for this job's cache, frames, and video
# This now points to your home directory
OUTPUT_DIR="/home/users/sithom/mSWE-GNN/animation_runs/$SLURM_JOB_NAME"
echo "Creating output directory: $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR


# --- 3. Run Executable (UPDATED) ---
echo "Animating..."

# CHECKPOINT_NAME="GNN-epoch=96-val_loss=0.5524.ckpt"
CHECKPOINT_NAME="GNN-epoch=65-val_loss=0.5557.ckpt"

# Use the config/checkpoint for K=3
# Added the new required -o argument
python -m mswegnn.utils.adforce_predict_animate \
     -c /home/users/sithom/my_results/${SLURM_JOB_NAME}/checkpoints/config.yaml \
     -ckpt /home/users/sithom/my_results/${SLURM_JOB_NAME}/checkpoints/${CHECKPOINT_NAME} \
     -nc /home/users/sithom/swegnn_5sec/152_KATRINA_2005.nc \
     -s /home/users/sithom/my_results/${SLURM_JOB_NAME}/processed/scaling_stats.yaml \
     -r -1 \
     -o $OUTPUT_DIR

echo "Job completed."
echo "Output saved to: $OUTPUT_DIR"
echo "Done."
