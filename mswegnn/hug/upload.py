import os
from huggingface_hub import HfApi

if __name__ == "__main__":
    # ends up at https://huggingface.co/datasets/sdat2/surgenet-train
    # python -m mswegnn.hug.upload
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.upload_large_folder(
        folder_path="../swegnn_5sec/",
        repo_id="sdat2/surgenet-train",
        repo_type="dataset",
    )
