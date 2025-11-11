#!/bin/bash

# --- SLURM Directives (GPU Job for ORCHID) ---
#SBATCH --job-name=GNN-GCN
#SBATCH --account=orchid             # Use ORCHID account
#SBATCH --partition=orchid           # Use ORCHID partition
#SBATCH --qos=orchid                 # Use ORCHID QoS
#SBATCH --gres=gpu:1                 # Request 1 GPU (A100)
#SBATCH -o log/%j.out                # Standard output log
#SBATCH -e log/%j.err                # Standard error log
#SBATCH --time=23:00:00              # Set to 23 hours (under the likely 24-hour limit)
#SBATCH --ntasks-per-node=8
#SBATCH --mem=40000                  # Memory request in MB (for data loaders)

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
DATA_SOURCE_PATH="/home/users/sithom/swegnn_5sec"
RESULTS_DEST_PATH="/home/users/sithom/my_results"

# Create a unique job-specific directory on the SHARED SCRATCH filesystem
JOB_SCRATCH_DIR="/work/scratch-pw3/$USER/$SLURM_JOB_ID"
mkdir -p $JOB_SCRATCH_DIR

# Define paths within this new shared scratch directory
DATA_LOCAL_PATH="$JOB_SCRATCH_DIR/swegnn_5sec"
RESULTS_LOCAL_PATH="$JOB_SCRATCH_DIR/my_results"
mkdir -p $RESULTS_LOCAL_PATH

echo "Copying 36GB data from $DATA_SOURCE_PATH to shared scratch ($JOB_SCRATCH_DIR)..."
rsync -aq $DATA_SOURCE_PATH $JOB_SCRATCH_DIR/
echo "Copy complete."

# --- 3. Run Executable ---
echo "Starting python script on GPU, using paths in shared scratch..."
python -m adforce_main \
    machine=jasmin \
    machine.data_dir=$DATA_LOCAL_PATH \
    machine.output_dir=$RESULTS_LOCAL_PATH \
    +lightning_trainer.accelerator=gpu \
    +lightning_trainer.devices=1 \
    model_params.model_type=GNN \
    model_params.type_gnn=GCN \
    model_params.hid_features=124 \
    model_params.K=3 \
    model_params.normalize=True \
    model_params.with_gradient=True \
    data_params.num_workers=8 \
    model_params.edge_mlp=True \
    trainer_options.batch_size=8

echo "Script finished."

# --- 4. Copy Results Back ---
echo "Copying results from $RESULTS_LOCAL_PATH back to $RESULTS_DEST_PATH..."
mkdir -p $RESULTS_DEST_PATH
rsync -aq $RESULTS_LOCAL_PATH/ $RESULTS_DEST_PATH/
echo "Results copy complete."

# --- 5. CRITICAL: Clean Up Shared Scratch ---
echo "Cleaning up shared scratch directory: $JOB_SCRATCH_DIR"
rm -rf $JOB_SCRATCH_DIR
echo "Job complete."
