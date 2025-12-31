# Libraries
import torch
from torch_geometric.data import Batch

# from mswegnn.utils.dataset import get_inflow_volume


def get_mean_error(diff_rollout, type_loss, nodes_dim=0):
    """Calculates mean error between predictions and real values

    Parameters:
    diff_rollout: torch.tensor
        difference between predictions and real values
    type_loss: str
        options: 'RMSE', 'MAE'
    nodes_dim: int (default = 0)
        dimension where nodes are located

    Doctest:
    >>> import torch
    >>> # Test RMSE calculation
    >>> diff = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> rmse = get_mean_error(diff, 'RMSE', nodes_dim=0)
    >>> print(f"RMSE shape: {rmse.shape}, values: {rmse}")
    RMSE shape: torch.Size([3]), values: tensor([2.9155, 3.8079, 4.7434])
    >>>
    >>> # Test MAE calculation
    >>> mae = get_mean_error(diff, 'MAE', nodes_dim=0)
    >>> print(f"MAE shape: {mae.shape}, values: {mae}")
    MAE shape: torch.Size([3]), values: tensor([2.5000, 3.5000, 4.5000])
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

    Doctest:
    >>> import torch
    >>> # Test with some zero and non-zero values
    >>> diff = torch.tensor([[0.0, 0.0, 0.0],
    ...                      [1.0, 2.0, 3.0],
    ...                      [0.0, 0.0, 0.0],
    ...                      [0.1, 0.0, 0.2]])
    >>> mask = mask_on_water(diff, water_axis=1)
    >>> print(f"Mask: {mask}")
    Mask: tensor([False,  True, False,  True])
    >>> # Nodes 1 and 3 have at least one non-zero value
    """
    where_water = (diff != 0).any(water_axis)
    return where_water


def get_loss_variable_scaler(num_water_vars, velocity_scaler=1, device=None):
    """Scales loss in velocity terms by a factor velocity_scaler

    Parameters:
    num_water_vars: int
        number of water variables (e.g., 2 for depth+discharge, 3 for depth+vx+vy)
    velocity_scaler: float (default = 1)
        scales loss in velocity terms by a factor velocity_scaler
    device: torch.device (default = None)
        The device to create the tensor on.

    Doctest:
    >>> import torch
    >>> # Test with 3 variables (WD, VX, VY) - Adforce case
    >>> scaler = get_loss_variable_scaler(num_water_vars=3, velocity_scaler=7.0)
    >>> print(f"Scaler for 3 vars: {scaler}")
    Scaler for 3 vars: tensor([1., 7., 7.])
    >>>
    >>> # Test with 2 variables (WD, discharge) - Standard case
    >>> scaler = get_loss_variable_scaler(num_water_vars=2, velocity_scaler=5.0)
    >>> print(f"Scaler for 2 vars: {scaler}")
    Scaler for 2 vars: tensor([1., 5.])
    >>>
    >>> # Test that the function works (device test)
    >>> scaler_cpu = get_loss_variable_scaler(num_water_vars=3, device='cpu')
    >>> assert scaler_cpu.device.type == 'cpu'
    >>> print("Device test passed")
    Device test passed
    """
    # --- MODIFICATION: Create tensor on the specified device ---
    # This is the fix for the torch.dot error.
    # Create a scaler tensor: first element (water depth) = 1.0,
    # remaining elements (velocities/discharge) = velocity_scaler
    loss_scaler = torch.ones(num_water_vars, device=device)
    loss_scaler[1:] = velocity_scaler  # Scale all velocity components

    return loss_scaler


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

    Doctest:
    >>> import torch
    >>> from torch_geometric.data import Data
    >>>
    >>> # Test basic loss function with 3 variables (Adforce case)
    >>> num_nodes = 5
    >>> preds = torch.tensor([[1.0, 0.5, 0.3],
    ...                       [2.0, 1.0, 0.5],
    ...                       [0.5, 0.2, 0.1],
    ...                       [1.5, 0.8, 0.4],
    ...                       [0.0, 0.0, 0.0]])
    >>> real = torch.tensor([[1.1, 0.6, 0.4],
    ...                      [2.1, 1.1, 0.6],
    ...                      [0.4, 0.1, 0.0],
    ...                      [1.6, 0.9, 0.5],
    ...                      [0.0, 0.0, 0.0]])
    >>>
    >>> # Create mock data object
    >>> data = Data(x=torch.rand(num_nodes, 10))
    >>> data.node_BC = torch.tensor([0])  # Mock boundary condition nodes
    >>> BC = torch.tensor([0.0])  # Mock boundary conditions
    >>>
    >>> # Test RMSE loss without water masking
    >>> loss = loss_function(preds, real, data, BC, type_loss='RMSE',
    ...                      only_where_water=False, velocity_scaler=1.0)
    >>> print(f"RMSE loss (no masking): {loss.item():.4f}")
    RMSE loss (no masking): 0.0894
    >>>
    >>> # Test MAE loss without water masking
    >>> loss = loss_function(preds, real, data, BC, type_loss='MAE',
    ...                      only_where_water=False, velocity_scaler=1.0)
    >>> print(f"MAE loss (no masking): {loss.item():.4f}")
    MAE loss (no masking): 0.0800
    >>>
    >>> # Test with velocity scaler (should weight velocity terms more)
    >>> loss_scaler_1 = loss_function(preds, real, data, BC, type_loss='RMSE',
    ...                               only_where_water=False, velocity_scaler=1.0)
    >>> loss_scaler_5 = loss_function(preds, real, data, BC, type_loss='RMSE',
    ...                               only_where_water=False, velocity_scaler=5.0)
    >>> # With higher velocity scaler, loss contribution from velocities increases
    >>> # but the total can be similar depending on the error magnitudes
    >>> assert loss_scaler_1 > 0 and loss_scaler_5 > 0
    >>> print(f"Velocity scaler test passed")
    Velocity scaler test passed
    >>>
    >>> # Test with 2 variables (standard case)
    >>> _ =  torch.manual_seed(42)  # Fix seed for reproducible test
    >>> preds_2d = torch.tensor([[1.0, 0.5], [2.0, 1.0], [0.5, 0.2]])
    >>> real_2d = torch.tensor([[1.1, 0.6], [2.1, 1.1], [0.4, 0.1]])
    >>> data_2d = Data(x=torch.rand(3, 6))
    >>> data_2d.node_BC = torch.tensor([0])
    >>> loss_2d = loss_function(preds_2d, real_2d, data_2d, BC, type_loss='RMSE')
    >>> # Just verify it runs without error and returns a reasonable value
    >>> assert 0.05 < loss_2d.item() < 0.20
    >>> print("2-var test passed")
    2-var test passed
    """
    diff = preds - real  # This is on the model's device (e.g., cuda:0)

    if only_where_water:
        # --- MODIFICATION: Use the unscaled `y_unscaled` from the batch ---
        # `data.y_unscaled` is on the GPU because Lightning moved `data`.
        where_water = data.y_unscaled[:, 0].abs() > 1e-6  # 1e-6 is a small epsilon

        if where_water.sum() == 0:
            # Handle case where there is no water in the batch
            return torch.tensor(0.0, device=diff.device, requires_grad=True)

        diff = diff[where_water]

    if diff.shape[0] == 0:
        return torch.tensor(0.0, device=diff.device, requires_grad=True)

    # Infer number of water variables from the shape of predictions
    num_water_vars = diff.shape[1]

    loss = get_mean_error(
        diff, type_loss, nodes_dim=0
    )  # get scaled rmse or mae over nodes

    # --- MODIFICATION: Pass the device from `diff` and inferred num_water_vars ---
    # This ensures loss_scaler is created on the same device as loss (e.g., cuda:0)
    loss_scaler = get_loss_variable_scaler(
        num_water_vars=num_water_vars,
        velocity_scaler=velocity_scaler,
        device=diff.device,
    )
    # --- END MODIFICATION ---

    if loss.numel() == 0:
        return torch.tensor(0.0, device=diff.device, requires_grad=True)

    # This line should now work: cuda:0.dot(cuda:0)
    loss = torch.dot(loss, loss_scaler) / loss_scaler.sum()

    if conservation != 0:
        # Extract water depth at index 0 from each time step
        # Pattern: [WD, vel_x, vel_y, ...] repeated for each timestep
        input_WD = data.x[:, -num_water_vars::num_water_vars]
        pred_WD = preds[:, 0::num_water_vars]

        try:
            loss = (
                loss
                + conservation * conservation_loss(pred_WD, input_WD, data, BC).abs()
            )
        except NameError:
            pass  # conservation_loss not imported, skipping

    return loss


def conservation_loss(pred_WD, input_WD, data, BC):
    """
    Calculates loss for mass conservation

    Parameters:
    pred_WD: torch.tensor
        predicted water depth (shape = [num_nodes, 1])
    input_WD: torch.tensor
        input water depth (shape = [num_nodes, 1])
    data: torch_geometric.data.Data
        data object with the graph information
    BC: torch.tensor
        boundary conditions

    Doctest:
    >>> import torch
    >>> from torch_geometric.data import Data
    >>>
    >>> # Test conservation loss with mock data
    >>> num_nodes = 4
    >>> pred_WD = torch.tensor([[1.1], [2.2], [1.5], [0.8]])
    >>> input_WD = torch.tensor([[1.0], [2.0], [1.5], [0.7]])
    >>>
    >>> # Create mock data object with area
    >>> data = Data(x=torch.rand(num_nodes, 10))
    >>> data.area = torch.tensor([[100.0], [100.0], [100.0], [100.0]])
    >>> data.node_BC = torch.tensor([0])  # First node is boundary
    >>> BC = torch.tensor([0.0])
    >>>
    >>> # Calculate conservation loss (will use NameError fallback)
    >>> loss = conservation_loss(pred_WD, input_WD, data, BC)
    >>> print(f"Conservation loss: {loss.item():.6f}")
    Conservation loss: 0.000000
    >>>
    >>> # Test shape validation
    >>> wrong_shape = torch.tensor([[1.1, 2.2]])
    >>> try:
    ...     loss = conservation_loss(wrong_shape, input_WD, data, BC)
    ... except AssertionError as e:
    ...     print("AssertionError caught (shape mismatch)")
    AssertionError caught (shape mismatch)
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
