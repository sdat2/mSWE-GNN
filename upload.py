import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_large_folder(
    folder_path="../swegnn_5sec/",
    repo_id="sdat2/surgenet-train",
    repo_type="dataset",
)
