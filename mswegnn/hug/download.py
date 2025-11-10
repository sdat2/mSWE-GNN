"""
Script to download the 'sdat2/surgenet-train' dataset from the Hugging Face Hub.

This script uses huggingface_hub.snapshot_download to fetch the
entire repository, which is the counterpart to api.upload_large_folder.
"""

import os
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    # python -m mswegnn.hug.download
    # Define the repository you want to download
    REPO_ID = "sdat2/surgenet-train"
    REPO_TYPE = "dataset"
    
    # Define a local path to download the dataset to
    # This will create the directory if it doesn't exist
    LOCAL_DOWNLOAD_PATH = "../swegnn_5sec_downloaded/"

    print(f"Downloading {REPO_ID} to {LOCAL_DOWNLOAD_PATH}...")

    # Use snapshot_download to get the entire repo
    # It will download the contents of the repo into LOCAL_DOWNLOAD_PATH
    downloaded_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=LOCAL_DOWNLOAD_PATH,
        # Use the token if the dataset is private
        token=os.getenv("HF_TOKEN"), 
    )

    print(f"\nDownload complete. Dataset is in: {downloaded_path}")
    print("Downloaded files:")
    # List the top-level files/folders in the downloaded directory
    try:
        for item in os.listdir(downloaded_path):
            print(f"- {item}")
    except OSError as e:
        print(f"Could not list files: {e}")
