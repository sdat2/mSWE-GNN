"""
A function to test the AdforceLazyDataset by creating an animation
of water depth over time using Matplotlib.

"""

import os
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib import animation
from mswegnn.utils.adforce_dataset import AdforceLazyDataset


def run_animation_test(root_dir, nc_file_path, p_t):
    """
    Runs the animation test using the AdforceLazyDataset.
    """
    print("Starting animation test...")

    # 1. Load coordinates once
    try:
        with xr.open_dataset(nc_file_path) as ds:
            x_coords = ds["x"].values
            y_coords = ds["y"].values
    except Exception as e:
        print(f"Failed to read file coordinates: {e}")
        return

    # 2. Initialize Dataset
    try:
        dataset = AdforceLazyDataset(
            root=root_dir, nc_files=[nc_file_path], previous_t=p_t
        )
    except Exception as e:
        print(f"Failed to initialize AdforceLazyDataset: {e}")
        if "Window mismatch" in str(e):
            print(f"Error: {e}")
            print(
                f"Please delete the 'processed' directory inside '{root_dir}' and try again."
            )
            return
        raise

    if len(dataset) == 0:
        print("Dataset is empty. Check time steps and p_t.")
        return

    print(f"Dataset loaded. Total samples: {len(dataset)}")

    # 3. Set up Matplotlib Animation
    fig, ax = plt.subplots(figsize=(10, 7))
    # Use scatter. For a real mesh, 'tripcolor' would be better if you load 'element' data.
    scat = ax.scatter(x_coords, y_coords, c=[], cmap="cividis", s=15, vmin=0, vmax=1)
    cb = fig.colorbar(scat, ax=ax, label="Water Depth (WD)")
    ax.set_title("Frame 0")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_aspect("equal")

    def init():
        scat.set_array(np.array([]))
        return (scat,)

    def update(frame_index):
        try:
            # THIS IS THE CORE: Get data from your loader
            data = dataset.get(frame_index)

            # Water depth is the first column of the target 'y'
            water_depth = data.y[:, 0].cpu().numpy()

            scat.set_array(water_depth)
            ax.set_title(f"Dataset Index: {frame_index} / {len(dataset) - 1}")

            # Set color limits on the first frame
            if frame_index == 1:
                vmin, vmax = np.min(water_depth), np.max(water_depth)
                scat.set_clim(vmin=vmin, vmax=vmax)

            return (scat,)
        except Exception as e:
            print(f"Error updating frame {frame_index}: {e}")
            return (scat,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(dataset),
        init_func=init,
        blit=True,
        interval=50,  # 50ms per frame
        repeat=False,
    )

    # print("Showing animation... You should see a wave move from left to right.")
    # print("Close the plot window to exit.")
    plt.show()


if __name__ == "__main__":
    # python -m mswegnn.utils.adforce_animate
    root_directory = "/Volumes/s/tcpips/swegnn_5sec/"
    netcdf_file = os.path.join(root_directory, "152_KATRINA_2005.nc")
    previous_time_steps = 2  # Adjust based on your data

    run_animation_test(root_directory, netcdf_file, previous_time_steps)
