"""Files used to loop for checkpoints and config files for Adforce model runs."""

import os
import glob
import re
import yaml


def find_best_checkpoint(checkpoint_dir: str) -> str:
    """
    Finds the checkpoint file with the lowest validation loss.

    Args:
        checkpoint_dir (str): Path to the directory containing .ckpt files.

    Returns:
        str: Full path to the best checkpoint, or None if not found.
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not ckpt_files:
        return None
    best_ckpt = None
    min_loss = float("inf")
    for ckpt in ckpt_files:
        match = re.search(r"val_loss=([0-9]+\.[0-9]+)", ckpt)
        if match:
            try:
                loss = float(match.group(1))
                if loss < min_loss:
                    min_loss = loss
                    best_ckpt = ckpt
            except ValueError:
                continue
    return best_ckpt


def get_run_paths(run_dir: str) -> dict:
    """
    Retrieves paths for checkpoint, config, and stats for a given run.

    Args:
        run_dir (str): The root directory of the model run.

    Returns:
        dict: Keys 'ckpt', 'config', 'stats' with file paths, or None if invalid.
    """
    paths = {}
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    paths["ckpt"] = find_best_checkpoint(ckpt_dir)

    cfg_path_ckpt = os.path.join(ckpt_dir, "config.yaml")
    cfg_path_root = os.path.join(run_dir, "config.yaml")
    # Check checkpoint dir first, then root
    paths["config"] = (
        cfg_path_ckpt
        if os.path.exists(cfg_path_ckpt)
        else (cfg_path_root if os.path.exists(cfg_path_root) else None)
    )

    stats_path = os.path.join(run_dir, "processed", "scaling_stats.yaml")
    paths["stats"] = stats_path if os.path.exists(stats_path) else None

    return paths if all(paths.values()) else None


def load_file_list(list_path: str, data_root: str) -> list:
    """
    Loads a list of filenames from a YAML file and prepends the data root.

    Args:
        list_path (str): Path to the YAML file.
        data_root (str): Root directory to prepend.

    Returns:
        list: List of full file paths.

    Raises:
        FileNotFoundError: If list_path does not exist.
    """
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"Missing: {list_path}")
    with open(list_path, "r") as f:
        filenames = yaml.safe_load(f)
    return [os.path.join(data_root, fname) for fname in filenames]
