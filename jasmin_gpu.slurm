#!/bin/bash

# --- SLURM Directives (GPU Job for ORCHID) ---
#SBATCH --job-name=SWEGNN-K3-P3
#SBATCH --account=orchid             # Use ORCHID account
#SBATCH --partition=orchid           # Use ORCHID partition
#SBATCH --qos=orchid                 # Use ORCHID QoS
#SBATCH --gres=gpu:1                 # Request 1 GPU (A100)
#SBATCH --exclude=gpuhost007         # Exclude problematic host (it seems to have some cuda issue)
#SBATCH -o log/%j.out                # Standard output log
#SBATCH -e log/%j.err                # Standard error log
#SBATCH --time=23:30:00              # Set to 23 hours
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
# ---
# --- THIS COMMAND IS UPDATED FOR THE NEW HYDRA CONFIG ---
# ---
python -m adforce_main \
    machine=jasmin \
    paths.raw_dir=$DATA_LOCAL_PATH \
    paths.processed_dir=$RESULTS_LOCAL_PATH/processed \
    paths.checkpoint_dir=$RESULTS_LOCAL_PATH/checkpoints \
    paths.wandb_dir=$RESULTS_LOCAL_PATH/wandb \
    machine.accelerator=gpu \
    machine.devices=1 \
    machine.num_workers=8 \
    model_params.model_type=GNN \
    model_params.previous_t=3 \
    models.type_gnn=SWEGNN \
    models.hid_features=128 \
    models.K=3 \
    models.normalize=True \
    models.with_gradient=True \
    models.edge_mlp=True \
    trainer_options.batch_size=4 \
    wandb.project=surgeflexdelta \
    wandb.name=${SLURM_JOB_NAME}

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