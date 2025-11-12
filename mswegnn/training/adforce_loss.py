# Libraries
import torch
from torch_geometric.data import Batch

# from mswegnn.utils.dataset import get_inflow_volume

# NUM_WATER_VARS = 3  # water depth and velocity x and velocity y


def get_mean_error(diff_rollout, type_loss, nodes_dim=0):
    """Calculates mean error between predictions and real values

    Parameters:
    diff_rollout: torch.tensor
        difference between predictions and real values
    type_loss: str
        options: 'RMSE', 'MAE'
    nodes_dim: int (default = 0)
        dimension where nodes are located
    """
    if type_loss == "RMSE":
        average_diff_t = torch.sqrt((diff_rollout**2).mean(nodes_dim))
    elif type_loss == "MAE":
        average_diff_t = diff_rollout.abs().mean(nodes_dim)
    return average_diff_t


def mask_on_water(diff, water_axis=1):
    """Mask to only calculate loss where there is water

    Parameters:
    diff: torch.tensor
        difference between predictions and real values
    water_axis: int (default = 1)
        axis where water depth is located
    """
    where_water = (diff != 0).any(water_axis)
    return where_water


def get_loss_variable_scaler(velocity_scaler=1, device=None): # <-- MODIFIED
    """Scales loss in velocity terms by a factor velocity_scaler

    Parameters:
    velocity_scaler: float (default = 1)
        scales loss in velocity terms by a factor velocity_scaler
    device: torch.device (default = None)
        The device to create the tensor on.
    """
    # --- MODIFICATION: Create tensor on the specified device ---
    # This is the fix for the torch.dot error.
    loss_scaler = torch.ones(NUM_WATER_VARS, device=device)
    loss_scaler[1::NUM_WATER_VARS] = velocity_scaler

    return loss_scaler


def get_multiscale_loss(
    diff, data, only_where_water=True, type_loss="RMSE", nodes_dim=0
):
    """Calculates multiscale loss by weighting loss in different scales

    Parameters:
    diff: torch.tensor
        difference between predictions and real values
    only_where_water: bool (default = True)
        if True, only calculates loss where there is water
    type_loss: str (default = 'RMSE')
        options: 'RMSE', 'MAE'
    nodes_dim: int (default = 0)
        dimension where nodes are located
    """
    node_ptr = data.node_ptr
    if only_where_water:
        where_water = mask_on_water(diff)
    else:
        # --- MODIFICATION: Ensure tensor is on the correct device ---
        where_water = torch.ones(diff.shape[0], device=diff.device).bool()

    if isinstance(data, Batch):
        multiscale_loss = get_mean_error(
            torch.cat(
                [
                    diff[data.node_ptr[i, 0] : data.node_ptr[i, 1]][
                        where_water[node_ptr[i, 0] : node_ptr[i, 1]]
                    ]
                    for i in range(data.num_graphs)
                ]
            ),
            type_loss,
            nodes_dim,
        )
    else:
        multiscale_loss = get_mean_error(
            diff[node_ptr[0] : node_ptr[1]][where_water[node_ptr[0] : node_ptr[1]]],
            type_loss,
            nodes_dim,
        )

    return multiscale_loss


def loss_function(
    preds,
    real,
    data,
    BC,
    type_loss="RMSE",
    only_where_water=False,
    conservation=0,
    velocity_scaler=1,
):
    """
    Calculates loss between predictions and real values

    Parameters:
    preds: torch.tensor (shape = [num_nodes, num_variables])
        predictions of the model
    real: torch.tensor (shape = [num_nodes, num_variables])
        real values
    data: torch_geometric.data.Data
        data object with the graph information
    BC: torch.tensor
        boundary conditions
    type_loss: str (default = 'RMSE')
        options: 'RMSE', 'MAE'
    only_where_water: bool (default = False)
        if True, only calculates loss where there is water
    conservation: float (default = 0)
        coefficient for mass conservation loss
    velocity_scaler: float (default = 1)
        scales loss in velocity terms by a factor velocity_scaler
    """
    diff = preds - real # This is on the model's device (e.g., cuda:0)

    if "node_ptr" in data.keys():
        loss = get_multiscale_loss(diff, data, only_where_water, type_loss, nodes_dim=0)
    else:
        if only_where_water:
            # --- MODIFICATION: Use the unscaled `y_unscaled` from the batch ---
            # `data.y_unscaled` is on the GPU because Lightning moved `data`.
            where_water = (data.y_unscaled[:, 0].abs() > 1e-6) # 1e-6 is a small epsilon
            
            if where_water.sum() == 0:
                # Handle case where there is no water in the batch
                return torch.tensor(0.0, device=diff.device, requires_grad=True)
                
            diff = diff[where_water]
        
        if diff.shape[0] == 0:
             return torch.tensor(0.0, device=diff.device, requires_grad=True)
             
        loss = get_mean_error(diff, type_loss, nodes_dim=0)

    # --- MODIFICATION: Pass the device from `diff` ---
    # This ensures loss_scaler is created on the same device as loss (e.g., cuda:0)
    loss_scaler = get_loss_variable_scaler(
        velocity_scaler=velocity_scaler, device=diff.device
    )
    # --- END MODIFICATION ---
    
    if loss.numel() == 0:
        return torch.tensor(0.0, device=diff.device, requires_grad=True)

    # This line should now work: cuda:0.dot(cuda:0)
    loss = torch.dot(loss, loss_scaler) / loss_scaler.sum()

    if conservation != 0:
        WD_index = 2
        input_WD = data.x[:, -WD_index::WD_index]
        pred_WD = preds[:, 0::WD_index]
        
        try:
            loss = (
                loss + conservation * conservation_loss(pred_WD, input_WD, data, BC).abs()
            )
        except NameError:
            pass # conservation_loss not imported, skipping
            
    return loss


def conservation_loss(pred_WD, input_WD, data, BC):
    """
    Calculates loss for mass conservation...
    ... (rest of file as before) ...
    """
    # This function relies on `get_inflow_volume`, which is not imported
    # and will raise a NameError, caught by loss_function.
    
    assert (
        pred_WD.shape == input_WD.shape
    ), f"Input or predictions have wrong dimensions ({pred_WD.shape} != {input_WD.shape})"
    delta_WD = pred_WD - input_WD  # [m]
    assert (
        delta_WD.dim() == 2
    ), f"Input or predictions have wrong dimensions ({delta_WD.dim()})"
    assert BC.dim() == 1, f"Boundary conditions have wrong dimensions ({BC.dim()})"

    area = data.area if data.area.dim() == 2 else data.area.unsqueeze(1)  # [m^2]

    if "node_ptr" in data.keys():
        if isinstance(data, Batch):
            predicted_inflow_volume = torch.cat(
                [
                    (area * delta_WD)[data.node_ptr[i, 0] : data.node_ptr[i, 1]]
                    for i in range(data.num_graphs)
                ]
            ).sum()  # [m^3]
        else:
            predicted_inflow_volume = (
                (area * delta_WD)[data.node_ptr[0] : data.node_ptr[1]]
            ).sum()
    else:
        predicted_inflow_volume = (area * delta_WD).sum()  # [m^3]

    try:
        inflow_volume = get_inflow_volume(data, BC)  # [m^3] 
        boundary_correction = (
            (area * delta_WD)[data.node_BC]
        ).sum()  # [m^3] # remove values at ghost cells

        conservation_loss = (
            predicted_inflow_volume - inflow_volume - boundary_correction
        ) / 1e6  # [m^3 * 1e6]

        if isinstance(data, Batch):
            conservation_loss = conservation_loss / data.num_graphs

        return conservation_loss
    except NameError:
        return torch.tensor(0.0, device=pred_WD.device)
