# mswegnn/models/adforce_base.py
import torch
from torch import Tensor
import torch.nn as nn


class AdforceBaseModel(nn.Module):
    """
    Base class for Adforce models.

    This class handles residual connections and output masking specifically
    for the Adforce data pipeline, which uses 3 output variables
    (WD, VX, VY).

    Args:
        previous_t (int): Number of previous time steps (for residual init).
        num_output_vars (int): Number of output variables. Must be 3 for Adforce.
        learned_residuals (bool/str/None): Residual connection type.
        seed (int): Random seed.
        device (str): PyTorch device.
    """

    def __init__(
        self,
        previous_t=1,
        num_output_vars=3,  # <-- NEW: Parameterized output
        learned_residuals=None,
        seed=42,
        residuals_base=2,
        residual_init="exp",
        device="cpu",
        **kwargs,  # Catches unused args like 'with_WL'
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.previous_t = previous_t
        self.learned_residuals = learned_residuals
        self.device = device

        # --- FIXED for Adforce ---
        self.num_output_vars = num_output_vars
        self.out_dim = self.num_output_vars

        if self.num_output_vars != 3:
            raise ValueError(
                f"AdforceBaseModel is designed for 3 output variables (WD, VX, VY), but got {num_output_vars}"
            )
        # ---

        # NOTE: The Adforce data structure (static, forcing, state)
        # is only compatible with `learned_residuals=False`.
        # The other modes read state history from `x` which is not there.
        if self.learned_residuals not in [False, None]:
            raise ValueError(
                f"Adforce data pipeline is only compatible with `learned_residuals: False` or `None`. "
                f"Got: {self.learned_residuals}"
            )

    def _add_residual_connection(self, x_input):
        """
        Add residual connection from the *input state* to the *output delta*.

        Args:
            x_input (Tensor): The *original* model input `x`, which has the
                              shape [N, static + forcing + state(3)].
        """
        residual_output = torch.zeros(
            x_input.shape[0],
            self.out_dim,  # device=self.device
        )

        if self.learned_residuals == False:
            # `x_input` is [static, forcing, state]
            # We want the last `self.out_dim` (3) features, which is the input state
            x0_state = x_input[:, -self.out_dim :]
            residual_output = x0_state

        return residual_output

    def _mask_small_WD(self, x_output, epsilon=0.001):
        """
        Mask water depth below a threshold and velocities where water is 0.

        Args:
            x_output (Tensor): The model output *after* the residual
                               connection. Shape [N, 3] (WD, VX, VY).
        """

        # --- FIXED for Adforce (WD, VX, VY) ---
        wd_col = x_output[:, 0:1]  # Shape [N, 1]
        v_cols = x_output[:, 1:3]  # Shape [N, 2]

        # Create mask based on water depth
        # 1.0 where water > epsilon, 0.0 otherwise
        mask = (wd_col.abs() > epsilon).float()

        # Apply mask
        wd_masked = wd_col * mask
        v_masked = v_cols * mask  # Zeros out velocity where wd is zero

        return torch.cat((wd_masked, v_masked), dim=1)