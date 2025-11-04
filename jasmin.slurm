#!/bin/bash

# --- SLURM Directives (Corrected for New Cluster) ---
#SBATCH --job-name=cnn4.128
#SBATCH --partition=highres              # Use the 'highres' partition
#SBATCH --qos=highres                    # Use the 'highres' QoS
#SBATCH -o log/%j.out                    # Standard output log (%j for SLURM)
#SBATCH -e log/%j.err                    # Standard error log
#SBATCH --time=48:00:00                  # 48-hour walltime (max for highres)
#SBATCH -n 10                            # Number of cores
#SBATCH --mem=20000                      # Memory request in MB

# --- 1. Environment Setup ---
echo "Loading conda environment..."
conda activate swegnn

# --- 2. Data Setup ---
# !! EDIT THESE PATHS !!
DATA_SOURCE_PATH="/home/users/sithom/swegnn_5sec"
RESULTS_DEST_PATH="/home/users/sithom/my_results"

# Define paths on the fast local disk
DATA_LOCAL_PATH="$TMPDIR/swegnn_5sec"
RESULTS_LOCAL_PATH="$TMPDIR/results"
mkdir -p $RESULTS_LOCAL_PATH

echo "Copying data from /home to local scratch ($TMPDIR)..."
rsync -aq $DATA_SOURCE_PATH $TMPDIR/
echo "Copy complete."

# --- 3. Run Executable ---
echo "Starting python script..."
python adforce_main.py --data-path $DATA_LOCAL_PATH --output-dir $RESULTS_LOCAL_PATH
echo "Script finished."

# --- 4. Copy Results Back ---
echo "Copying results from $TMPDIR back to $RESULTS_DEST_PATH..."
mkdir -p $RESULTS_DEST_PATH
rsync -aq $RESULTS_LOCAL_PATH/ $RESULTS_DEST_PATH/
echo "Job complete."
