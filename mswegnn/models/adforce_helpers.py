from torch import nn, Sequential

def make_mlp(
    input_size: int,
    output_size: int,
    hidden_size: int = 32,
    n_layers: int = 2,
    bias: bool = False,
    activation: str = "relu",
    dropout: float = 0,
    layer_norm: bool = False,
    device: str ="cpu",
):
    """Builds an MLP. A factory function for nn.Sequential MLPs.

    Args:
        input_size (int): Input feature size
        output_size (int): Output feature size
        hidden_size (int, optional): Hidden layer size. Defaults to 32.
        n_layers (int, optional): Number of layers. Defaults to 2.
        bias (bool, optional): If True, adds a learnable bias to the layers.
            Defaults to False.
        activation (str, optional): Activation function name. Defaults to "relu".
        dropout (float, optional): Dropout rate. Defaults to 0.
        layer_norm (bool, optional): If True, adds LayerNorm after each layer.
            Defaults to False.
        device (str, optional): Device to place tensors on. Defaults to "cpu".

    Returns:
        nn.Sequential: The constructed MLP model.
    """
    layers = []
    if n_layers == 1:
        layers.append(nn.Linear(input_size, output_size, bias=bias, device=device))
        layers = layers + add_norm_dropout_activation(
            output_size,
            layer_norm=layer_norm,
            dropout=dropout,
            activation=activation,
            device=device,
        )
    else:
        layers.append(nn.Linear(input_size, hidden_size, bias=bias, device=device))
        layers = layers + add_norm_dropout_activation(
            hidden_size,
            layer_norm=layer_norm,
            dropout=dropout,
            activation=activation,
            device=device,
        )

        for layer in range(n_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size, bias=bias, device=device))
            layers = layers + add_norm_dropout_activation(
                hidden_size,
                layer_norm=layer_norm,
                dropout=dropout,
                activation=activation,
                device=device,
            )

        layers.append(nn.Linear(hidden_size, output_size, bias=bias, device=device))
        layers = layers + add_norm_dropout_activation(
            output_size,
            layer_norm=layer_norm,
            dropout=dropout,
            activation=activation,
            device=device,
        )

    mlp = Sequential(*layers)
    # mlp.apply(init_weights)

    return mlp


def activation_functions(activation_name, device="cpu"):
    """Returns an activation function given its name

    Args:
        activation_name (str): Name of the activation function.
        device (str, optional): Device to place tensors on. Defaults to "cpu".

    Returns:
        nn.Module: The activation function module.

    Raises:
        AttributeError: If the activation function name is not recognized.
    """
    if activation_name == "relu":
        return nn.ReLU()
    elif activation_name == "prelu":
        return nn.PReLU(device=device)
    elif activation_name == "leakyrelu":
        return nn.LeakyReLU(0.1)
    elif activation_name == "elu":
        return nn.ELU()
    elif activation_name == "swish":
        return nn.SiLU()
    elif activation_name == "sigmoid":
        return nn.Sigmoid()
    elif activation_name == "tanh":
        return nn.Tanh()
    elif activation_name is None:
        return None
    else:
        raise AttributeError(
            "Please choose one of the following options:\n"
            '"relu", "prelu", "leakyrelu", "elu", "gelu", "sigmoid", "tanh"'
        )
