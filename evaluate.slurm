#!/bin/bash

# --- SLURM Directives (GPU Job for ORCHID) ---
#SBATCH --job-name=eval
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
micromamba activate mswegnn-gpu 

# Add cluster's CUDA to path, as per JASMIN docs
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}

echo "Checking for GPU..."
nvidia-smi # Print GPU status to the log

# --- 3. Run Executable (UPDATED) ---
echo "Evaluating."
python -m mswegnn.utils.adforce_evaluate_models \
    --results_dir /home/users/sithom/my_results \
    --data_dir /home/users/sithom/swegnn_5sec \
    --conf_dir /home/users/sithom/mSWE-GNN/conf \
    --output comparison


echo "Job completed."
echo "Output saved to: comparison"
echo "Done."
