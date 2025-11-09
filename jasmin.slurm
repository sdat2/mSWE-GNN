#!/bin/bash

# --- SLURM Directives (CPU Job) ---
#SBATCH --job-name=SWEGNN_cpu
#SBATCH --account=no-project             
#SBATCH --partition=standard              
#SBATCH --qos=high                    
#SBATCH -o log/%j.out                    # Standard output log
#SBATCH -e log/%j.err                    # Standard error log
#SBATCH --time=48:00:00                  
#SBATCH -n 10                            # Number of cores
#SBATCH --mem=20000                      # Memory request in MB
#SBATCH --ntasks-per-node=10

# --- 1. Environment Setup ---
echo "Loading conda environment..."
source ~/.bashrc
micromamba activate mswegnn2 # Using your CPU environment

# --- 2. Data Setup ---
# Define permanent paths
DATA_SOURCE_PATH="/home/users/sithom/swegnn_5sec"
RESULTS_DEST_PATH="/home/users/sithom/my_results"

# --- THIS IS THE CORRECTED PATH ---
# Create a unique job-specific directory on the SHARED SCRATCH filesystem
# Using /work/scratch-pw3 as per JASMIN docs.
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
echo "Starting python script, using paths in shared scratch..."
python -m adforce_main \
    machine=jasmin \
    machine.data_dir=$DATA_LOCAL_PATH \
    machine.output_dir=$RESULTS_LOCAL_PATH \
    model_params.model_type=GNN \
    model_params.type_gnn=SWEGNN \
    model_params.hid_features=64 \
    model_params.K=3 \
    model_params.normalize=True \
    model_params.with_gradient=True \
    model_params.edge_mlp=True \
    trainer_options.batch_size=2

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