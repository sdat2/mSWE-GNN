"""
Example of how to use this in your main.py
"""
import glob
import os
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from utils.adforce_dataset import AdforceLazyDataset # Import your new class

# 1. Find all your NetCDF files
data_dir = "/Volumes/s/tcpips/swegnn_5sec/"
all_nc_files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
print(f"Found {len(all_nc_files)} total simulation files.")

# 2. Split the FILE LIST (e.g., 80% train, 20% validation)
# This ensures simulations don't leak between sets
train_files, val_files = train_test_split(
    all_nc_files,
    test_size=0.2,
    random_state=42
)

print(f"Training on {len(train_files)} files, validating on {len(val_files)} files.")

# 3. Create a separate "lazy" dataset for each split
# The .process() method will run ONLY for its own files
try:
    train_dataset = AdforceLazyDataset(
        root="data_processed/train",
        nc_files=train_files
    )

    val_dataset = AdforceLazyDataset(
        root="data_processed/val",
        nc_files=val_files
    )

    # 4. Create DataLoaders
    # The DataLoader will call train_dataset.get(i) for each batch
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    print("Train DataLoader created.", train_loader)

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    print("Validation DataLoader created.", val_loader)

    # 5. ... Your training loop ...
    # for batch in train_loader:
    #     ...

except Exception as e:
    print(f"Error initializing dataset: {e}")
    # Make sure to close file handles if an error occurs
    if 'train_dataset' in locals():
        train_dataset.close()
    if 'val_dataset' in locals():
        val_dataset.close()

finally:
    # 6. Clean up file handles when training is done
    print("Closing file handles...")
    if 'train_dataset' in locals():
        train_dataset.close()
    if 'val_dataset' in locals():
        val_dataset.close()



# import glob
# import os
# from sklearn.model_selection import train_test_split
# from torch_geometric.loader import DataLoader
# # from adforce_dataset import AdforceLazyDataset # Import your new class

# # 1. Find all your PRE-PROCESSED NetCDF files
# data_dir = "data/processed_simulations/" # <-- NEW PATH
# all_nc_files = sorted(glob.glob(os.path.join(data_dir, "*_swegnn.nc")))
# print(f"Found {len(all_nc_files)} total simulation files.")

# # 2. Split the FILE LIST
# train_files, val_files = train_test_split(
#     all_nc_files, test_size=0.2, random_state=42
# )

# # 3. Create datasets (this builds the index_map.pkl)
# train_dataset = AdforceLazyDataset(root="data_processed/train", nc_files=train_files)
# val_dataset = AdforceLazyDataset(root="data_processed/val", nc_files=val_files)

# # 4. Create DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# # 5. ... Training loop ...
# # ...
# # 6. Clean up
# train_dataset.close()
# val_dataset.close()
