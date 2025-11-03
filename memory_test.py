import glob
import os
import sys
import torch
import xarray as xr
import psutil  # You may need to: pip install psutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- Imports from your project ---
# (Assumes this script is in the same folder as adforce_main.py)
from mswegnn.utils.adforce_dataset import AdforceLazyDataset
from mswegnn.utils.adforce_dataset import _load_static_data_from_ds

# --- Configuration (from adforce_main.py & adforce_config.yaml) ---
DATA_DIR = "/Volumes/s/tcpips/swegnn_5sec/"  #
PREVIOUS_T = 1  #
TRAIN_ROOT = "data_processed/train"  #
STATS_PATH = "data_processed/train/scaling_stats.yaml"  #


def get_process_memory_mb():
    """Returns the memory usage of the current process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_test():
    print("--- Starting Memory Test ---")

    # 1. Get file list (same as adforce_main.py)
    print(f"Searching for NetCDF files in {DATA_DIR}...")
    all_nc_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.nc")))
    if not all_nc_files:
        print(f"ERROR: No *.nc files found in {DATA_DIR}.")
        return

    all_nc_files = all_nc_files[:10]  # Use the same 10 files

    train_files, _ = train_test_split(all_nc_files, test_size=0.2, random_state=42)
    print(f"Found {len(train_files)} training files for the test.")

    # 2. Pre-load static data (same as our fix for adforce_main.py)
    print(f"Loading shared static data from: {train_files[0]}...")
    static_data_cpu = {}
    try:
        with xr.open_dataset(train_files[0]) as ds:
            static_data_cpu = _load_static_data_from_ds(ds)
        print("Shared static data loaded to CPU.")
    except Exception as e:
        print(f"Failed to load static data: {e}")
        return

    # 3. Initialize the dataset
    print("Initializing AdforceLazyDataset...")
    try:
        dataset = AdforceLazyDataset(
            root=TRAIN_ROOT,
            nc_files=train_files,
            previous_t=PREVIOUS_T,
            scaling_stats_path=STATS_PATH,
            # preloaded_static_data=static_data_cpu,
        )
    except Exception as e:
        print(f"Failed to init dataset: {e}")
        return

    print(f"Dataset initialized with {len(dataset)} samples.")
    print("-" * 30)

    # 4. Run the memory test loop
    initial_mem = get_process_memory_mb()
    print(f"Initial memory: {initial_mem:.2f} MB")

    # We will store a few samples to simulate a batch
    batch_size = 32
    batch_buffer = []

    for i in tqdm(range(len(dataset)), desc="Iterating dataset"):
        try:
            # This is what the DataLoader does:
            sample = dataset.get(i)
            batch_buffer.append(sample)

            # Simulate clearing the batch
            if i > 0 and i % batch_size == 0:
                del batch_buffer
                batch_buffer = []

            # Report memory usage every 50 samples
            if i > 0 and i % 50 == 0:
                print(
                    f"  [Sample {i}] Current memory: {get_process_memory_mb():.2f} MB"
                )

        except Exception as e:
            print(f"Error at sample {i}: {e}")
            break

    final_mem = get_process_memory_mb()
    print("-" * 30)
    print("Test complete.")
    print(f"Initial memory: {initial_mem:.2f} MB")
    print(f"Final memory:   {final_mem:.2f} MB")
    print(f"Memory change:  {final_mem - initial_mem:.2f} MB")
    print("-" * 30)


if __name__ == "__main__":
    # python memory_test.py
    run_test()
