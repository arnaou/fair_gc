########################################################################################################################
#                                                                                                                      #
#    MLP model                                                                                          #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
#    Authors: Adem R.N. Aouichaoui                                                                                     #
#    2025/04/23                                                                                                        #
#                                                                                                                      #
########################################################################################################################

##########################################################################################################
# Import packages and modules
##########################################################################################################
from torch.nn.init import xavier_uniform_
from torch.nn.init import zeros_
from typing import Union, Sequence, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F





##########################################################################################################
# Define layers and models
##########################################################################################################
class Dense(nn.Linear):
    r"""Fully connected linear layer with activation function. from: https://schnetpack.readthedocs.io/en/latest/_modules/schnetpack/nn/base.html#Dense

    .. math::
       y = activation(x W^T + b)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None,
        weight_init: Callable = xavier_uniform_,
        bias_init: Callable = zeros_,
        generator: Optional[torch.Generator] = None,  # Add generator
    ):
        """
        Args:
            in_features: number of input feature :math:`x`.
            out_features: number of output features :math:`y`.
            bias: If False, the layer will not adapt bias :math:`b`.
            activation: if None, no activation function is used.
            weight_init: weight initializer from current weight.
            bias_init: bias initializer from current bias.
        """
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.generator = generator
        super(Dense, self).__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        if self.generator is None:
            self.weight_init(self.weight)
            if self.bias is not None:
                self.bias_init(self.bias)
        else:
            self.weight_init(self.weight, generator=self.generator)
            if self.bias is not None:
                self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        y = F.linear(input, self.weight, self.bias)
        y = self.activation(y)
        return y


def build_mlp(
    n_in: int,
    n_out: int,
    n_hidden: Optional[Union[int, Sequence[int]]] = None,
    n_layers: int = 2,
    activation: Callable = F.relu,
    last_bias: bool = True,
    last_zero_init: bool = False,
    final_dropout: float = 0.0,
    seed: int = 42,

) -> nn.Module:
    """
    Build multiple layer fully connected perceptron neural network.

    Args:
        n_in: number of input nodes.
        n_out: number of output nodes.
        n_hidden: number hidden layer nodes.
            If an integer, same number of node is used for all hidden layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_layers: number of layers.
        activation: activation function. All hidden layers would
            the same activation function except the output layer that does not apply
            any activation function.
    """

    generator = torch.manual_seed(seed)

    # get list of number of nodes in input, hidden & output layers
    if n_hidden is None:
        c_neurons = n_in
        n_neurons = []
        for i in range(n_layers):
            n_neurons.append(c_neurons)
            c_neurons = max(n_out, c_neurons // 2)
        n_neurons.append(n_out)
    else:
        # get list of number of nodes hidden layers
        if type(n_hidden) is int:
            n_hidden = [n_hidden] * (n_layers - 1)
        else:
            n_hidden = list(n_hidden)
            n_layers = len(n_hidden) + 1
        n_neurons = [n_in] + n_hidden + [n_out]

    # assign a Dense layer (with activation function) to each hidden layer
    # layers = [
    #     Dense(n_neurons[i], n_neurons[i + 1], activation=activation, generator=generator)
    #     for i in range(n_layers - 1)
    # ]
    # Create a list to hold all layers (including dropout)
    all_layers = []

    # Add hidden layers with activation and dropout
    for i in range(n_layers - 1):
        all_layers.append(Dense(n_neurons[i], n_neurons[i + 1], activation=activation, generator=generator))
        # Add dropout layer after each hidden layer if dropout > 0
        if final_dropout > 0:
            all_layers.append(nn.Dropout(p=final_dropout))
    # assign a Dense layer (without activation function) to the output layer

    if last_zero_init:
        all_layers.append(
            Dense(
                n_neurons[-2],
                n_neurons[-1],
                activation=None,
                weight_init=torch.nn.init.zeros_,
                bias=last_bias,
                generator=generator
            )
        )
    else:
        all_layers.append(
            Dense(n_neurons[-2], n_neurons[-1], activation=None, bias=last_bias, generator=generator)
        )
    # put all layers together to make the network
    out_net = nn.Sequential(*all_layers)
    return out_net