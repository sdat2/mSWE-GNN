"""Miscellaneous utility functions for Adforce models."""
from typing import Optional, Dict
from omegaconf import DictConfig, OmegaConf, ListConfig
import torch.nn as nn
from mswegnn.models.adforce_models import   MonolithicMLPModel, GNNModelAdforce, PointwiseMLPModel
from mswegnn.training.adforce_train import AdforceLightningModule


def feature_count(cfg: DictConfig) -> Dict[str, int]:
    """Calculate the number of input and output features for the model based on the config.

    Args:
        cfg (DictConfig): Configuration object containing feature specifications.

    Returns:
        Dict[str, int]: A dictionary with counts of node features, edge features, and output features.

    Doctest::
        >>> ex = DictConfig({
        ...     "models": {"previous_t": 1},
        ...     "features": {
        ...         "static": ["DEM", "slopex", "slopey", "area"],
        ...         "forcing": ["P", "WX", "WY"],
        ...         "state": ["WD", "VX", "VY"],
        ...         "derived_state": [{"name": "SSH", "op": "add", "args":["WD", "DEM"]},],
        ...         "edge": ["face_distance", "edge_slope"],
        ...         "targets": ["WD", "VX", "VY"],
        ...     }
        ... })
        >>> feature_count(ex)
        {'num_node_features': 12, 'num_edge_features': 2, 'num_output_features': 3, 'num_static_node_features': 5}
        >>> ex.models["previous_t"] = 2
        >>> feature_count(ex)
        {'num_node_features': 15, 'num_edge_features': 2, 'num_output_features': 3, 'num_static_node_features': 5} 
    """
    p_t = cfg.models.previous_t
    num_static_node_features = len(cfg.features.static) + 1 # +1 for node_type feature (not scaled)
    num_forcing_features = len(cfg.features.forcing)
    # The state can include derived features, so we count them all
    num_current_state_features = len(cfg.features.state)
    if cfg.features.get("derived_state"):
        num_current_state_features += len(cfg.features.derived_state)

    num_node_features = (
        num_static_node_features
        + (num_forcing_features * p_t)
        + num_current_state_features
    )
    num_edge_features = len(cfg.features.edge)

    # Model predicts the delta for the state (which does not include the derived features)
    num_output_features = len(cfg.features.targets)

    return {"num_node_features": num_node_features,
            "num_edge_features": num_edge_features,
            "num_output_features": num_output_features,
            "num_static_node_features": num_static_node_features}


def _create_model(model_cfg_dict, 
                 model_type,
                 num_node_features=None,
                 num_edge_features=None,
                 num_output_features=None,
                 num_static_node_features=None,
                 num_nodes_fixed=None) -> nn.Module:
    """Create a pytorch model.

    Args:
        model_cfg_dict (_type_): _description_
        model_type (_type_): _description_
        num_node_features (_type_, optional): _description_. Defaults to None.
        num_edge_features (_type_, optional): _description_. Defaults to None.
        num_output_features (_type_, optional): _description_. Defaults to None.
        num_static_node_features (_type_, optional): _description_. Defaults to None.
        num_nodes_fixed (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        nn.Module: pytorch model.
    """
    if model_type == "GNN":
        model = GNNModelAdforce(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_output_features=num_output_features,
            num_static_features=num_static_node_features,  # Pass the calculated count
            **model_cfg_dict,  # This dict includes previous_t
        )

    elif model_type == "MLP":
        model = PointwiseMLPModel(
            num_node_features=num_node_features,
            num_output_features=num_output_features,
            **model_cfg_dict,
        )
    elif model_type == "MonolithicMLP":
        if num_nodes_fixed is None:
            raise ValueError(
                "num_nodes_fixed must be provided for MonolithicMLPModel."
            )
        print(f"Found fixed n_nodes from dataset: {num_nodes_fixed}")
        model = MonolithicMLPModel(
            n_nodes=num_nodes_fixed,
            num_node_features=num_node_features,
            num_output_features=num_output_features,
            **model_cfg_dict,
        )
    else:
        raise ValueError(
            f"Unknown model_type in config: {model_type}. Must be 'GNN', 'MLP', or 'MonolithicMLP'."
        )
    return model

# get empty model from cfg
# would not work with monolithic MLP as no node_num
def model_from_cfg(cfg: DictConfig) -> nn.Module:
    fcount_d = feature_count(cfg)
    model = _create_model(model_cfg_dict=dict(cfg.model), **fcount_d)
    return model


def model_from_cfg_and_checkpoint(cfg: DictConfig, checkpoint_path: str) -> AdforceLightningModule:
    """Load a model from config and checkpoint.

    Args:
        cfg (DictConfig): Configuration object.
        checkpoint_path (str): Path to the checkpoint file.
    
    Returns:
        AdforceLightningModule: Loaded model.
    """
    model = model_from_cfg(cfg)
    lightning_model = AdforceLightningModule.load_from_checkpoint(
            checkpoint_path,
            # map_location=device,
            model=model,
            lr_info=cfg.lr_info,  # <-- From config
            trainer_options=cfg.trainer_options,  # <-- From config
        )
    #  lightning_model.to(device)
    lightning_model.eval()
    return lightning_model


if __name__ == "__main__":
    import os
    path = "/work/scratch-pw3/sithom/49751608/my_results/checkpoints"
    cfg_path =  os.path.join(path, "config.yaml")
    checkpoint_path = os.path.join(path, "GNN-epoch=78-val_loss=0.3645.ckpt")
    cfg = OmegaConf.load(cfg_path)
    model = model_from_cfg_and_checkpoint(cfg, checkpoint_path)
    print(model)

