#!/bin/bash

# --- SLURM Directives (Corrected for New Cluster) ---
#SBATCH --job-name=SWEGNN
#SBATCH --account=no-project             # I'm not back on AI4ER yet
#SBATCH --partition=standard              # Use the 'highres' partition
#SBATCH --qos=high                    # Use the 'highres' QoS
#SBATCH -o log/%j.out                    # Standard output log (%j for SLURM)
#SBATCH -e log/%j.err                    # Standard error log
#SBATCH --time=48:00:00                  # 48-hour walltime (max for highres)
#SBATCH -n 10                            # Number of cores
#SBATCH --mem=20000                      # Memory request in MB
#SBATCH --ntasks-per-node=10

# --- 1. Environment Setup ---
echo "Loading conda environment..."
source ~/.bashrc
micromamba activate mswegnn2

# --- 2. Data Setup ---
# DATA_SOURCE_PATH="/home/users/sithom/swegnn_5sec"
# RESULTS_DEST_PATH="/home/users/sithom/my_results"

# Define paths on the fast local disk
# DATA_LOCAL_PATH="$TMPDIR/swegnn_5sec"
# RESULTS_LOCAL_PATH="$TMPDIR/results"
# mkdir -p $RESULTS_LOCAL_PATH

# echo "Copying data from /home to local scratch ($TMPDIR)..."
# rsync -aq $DATA_SOURCE_PATH $TMPDIR/
# echo "Copy complete."

# --- 3. Run Executable ---
echo "Starting python script..."
# python -m adforce_main machine=jasmin 

# python -m adforce_main machine=jasmin model_params.model_type=MonolithicMLP

python -m adforce_main \
    machine=jasmin \
    model_params.model_type=GNN \
    model_params.type_gnn=SWEGNN \
    model_params.hid_features=64 \
    model_params.K=3 \
    model_params.normalize=True \
    model_params.with_gradient=True \
    model_params.edge_mlp=True \
    trainer_options.batch_size=2
# python -m adforce_main machine=jasmin  model_params.model_type=SWEGNN

# python -m adforce_main # .py #--data-path $DATA_LOCAL_PATH --output-dir $RESULTS_LOCAL_PATH
echo "Script finished."

# --- 4. Copy Results Back ---
# echo "Copying results from $TMPDIR back to $RESULTS_DEST_PATH..."
# mkdir -p $RESULTS_DEST_PATH
# rsync -aq $RESULTS_LOCAL_PATH/ $RESULTS_DEST_PATH/
# echo "Job complete."

# data was transferred with:
# rsync -aPv --stats -e "ssh -o ServerAliveInterval=60" sdat2@archer2-n02:/work/n02/n02/sdat2/adcirc-swan/worstsurge/swegnn_5sec/ /home/users/sithom/swegnn_5sec/