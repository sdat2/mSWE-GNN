# Libraries
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin
from torch.nn import ReLU, PReLU, ELU, SiLU, Sigmoid, Dropout, Tanh, LeakyReLU

class BaseFloodModel(nn.Module):
    '''Base class for modelling flood inundation
    ------
    learned_residuals: bool/str/None (default=False)
        adds weighted residual connection with solution at previous time steps
    residuals_base: int (default=5)
        when using residual_init='exp', this selects the base of the exponent in the weights initialization
    residual_init: str (default='exp')
        type of residual weights initialization. options 'exp' or 'random'
    previous_t: int
        dataset-specific parameter that indicates the number of previous times steps given as input
    seed: int (default=42)
        seed used for replicability
    '''
    def __init__(self, previous_t=1, learned_residuals=None, seed=42, 
                 residuals_base=2, residual_init='exp', with_WL=False, 
                 device='cpu'):
        super().__init__()
        torch.manual_seed(seed)
        self.previous_t = previous_t
        self.with_WL = with_WL
        self.learned_residuals = learned_residuals
        self.device = device
        self.residuals_base = residuals_base
        self.residual_init = residual_init
        assert residual_init == 'exp' or residual_init == 'random', "Argument 'residual_init' can only be either 'exp' or 'random'"
        self.NUM_WATER_VARS = 2
        self.out_dim = self.NUM_WATER_VARS
        
        if learned_residuals == True:
            if residual_init == 'exp':
                self.residual_weights = init_true_residuals_weights(previous_t, residuals_base, device=device)
            else:
                self.residual_weights = nn.Parameter(torch.Tensor(previous_t,1).to(device))
                nn.init.xavier_normal_(self.residual_weights)

        elif learned_residuals == 'all':
            if residual_init == 'exp':
                self.residual_weights = init_true_residuals_weights(previous_t, residuals_base, repeat=self.out_dim, device=device)
            else:
                self.residual_weights = nn.Parameter(torch.Tensor(previous_t,self.out_dim).to(device))
                nn.init.xavier_normal_(self.residual_weights)
        
    def _add_residual_connection(self, x):
        '''Add residual connections from input to output
        
        options:
            'all' : add learned residual connections for each entry of U^{t-p:t}
            True  : add learned residual connections, independently of the output dimension
            False : add unweighted residual connection for last time instant
            None  : no residual connection
        '''
        residual_output = torch.zeros(x.shape[0], self.out_dim, device=self.device)

        if self.learned_residuals==True:
            x0 = x[:,-self.previous_t*self.NUM_WATER_VARS:].reshape(-1, self.previous_t, self.NUM_WATER_VARS)
            residual_output = torch.stack([(x0[:,:,i]@self.residual_weights[:,0])
                                                    for i in range(self.NUM_WATER_VARS)], -1)
            
        elif self.learned_residuals=='all':
            x0 = x[:,-self.previous_t*self.out_dim:].reshape(-1, self.previous_t, self.out_dim)
            residual_output = torch.stack([(x0[:,:,i]@self.residual_weights[:,i]) for i in range(self.out_dim)], -1)
                
        elif self.learned_residuals==False:
            x0 = x[:,-self.out_dim:]
            residual_output = x0

        # else:
        #     raise AttributeError("Please choose one of the following options:\n 'all', True, False")
        
        return residual_output
    
    def _mask_small_WD(self, x, epsilon=0.001):
        '''Mask water depth below a certain threshold epsilon and 
        mask velocities where there is no water'''
        wd_index = slice(0, x.shape[1], self.NUM_WATER_VARS)
        v_index = slice(1, x.shape[1], self.NUM_WATER_VARS)

        wd = x[:,wd_index] * (x[:,wd_index].abs() > epsilon)

        # Mask velocities where there is no water
        v = x[:,v_index] * (x[:,wd_index] != 0)
        x = torch.cat((wd, v), dim=-1)

        return x

def init_true_residuals_weights(previous_t: int, base=2, repeat=1, device='cpu'):
    '''Residual weights initialization for learned_residuals=True
    The assumption is that final time steps have more influence 
    than initial ones'''
    residual_weights = torch.Tensor([base**exp for exp in range(previous_t)]).to(device)
    norm_residual_weights = residual_weights/residual_weights.sum()
    norm_residual_weights = norm_residual_weights.repeat(repeat).reshape(repeat,-1).T
    return nn.Parameter(norm_residual_weights)

def add_norm_dropout_activation(hidden_size, layer_norm=False, dropout=0, activation='relu', 
                                device='cpu'):
    '''Add LayerNorm, Dropout, and activation function'''
    layers = []
    if layer_norm:
        layers.append(nn.LayerNorm(hidden_size, eps=1e-5, device=device))
    if dropout:
        layers.append(Dropout(dropout))
    if activation is not None:
        layers.append(activation_functions(activation, device=device))
    return layers


def init_weights(layer):
    if isinstance(layer, Lin):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.normal_(layer.bias)

def make_mlp(input_size, output_size, hidden_size=32, n_layers=2, bias=False, 
             activation='relu', dropout=0, layer_norm=False, device='cpu'):
    """Builds an MLP"""
    layers = []
    if n_layers==1:
        layers.append(Lin(input_size, output_size, bias=bias, device=device))
        layers = layers + add_norm_dropout_activation(output_size, layer_norm=layer_norm, 
                                                      dropout=dropout, activation=activation, device=device)
    else:
        layers.append(Lin(input_size, hidden_size, bias=bias, device=device))
        layers = layers + add_norm_dropout_activation(hidden_size, layer_norm=layer_norm, dropout=dropout, 
                                                      activation=activation, device=device)
            
        for layer in range(n_layers-2):
            layers.append(Lin(hidden_size, hidden_size, bias=bias, device=device))
            layers = layers + add_norm_dropout_activation(hidden_size, layer_norm=layer_norm, dropout=dropout, 
                                                          activation=activation, device=device)

        layers.append(Lin(hidden_size, output_size, bias=bias, device=device))
        layers = layers + add_norm_dropout_activation(output_size, layer_norm=layer_norm, dropout=dropout, 
                                                      activation=activation, device=device)

    mlp = Seq(*layers)
    # mlp.apply(init_weights)

    return mlp


def activation_functions(activation_name, device='cpu'):
    '''Returns an activation function given its name'''
    if activation_name == 'relu':
        return ReLU()
    elif activation_name == 'prelu':
        return PReLU(device=device)
    elif activation_name == 'leakyrelu':
        return LeakyReLU(0.1)
    elif activation_name == 'elu':
        return ELU()
    elif activation_name == 'swish':
        return SiLU()
    elif activation_name == 'sigmoid':
        return Sigmoid()
    elif activation_name == 'tanh':
        return Tanh()
    elif activation_name is None:
        return None
    else:
        raise AttributeError('Please choose one of the following options:\n'\
            '"relu", "prelu", "leakyrelu", "elu", "gelu", "sigmoid", "tanh"')