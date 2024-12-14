import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import MessagePassing, global_add_pool
from typing import Dict, List
from torch_geometric.nn import MessagePassing, Set2Set

class MPNNConv(MessagePassing):
    """
    Message Passing layer from the original MPNN paper.
    Implements the message and update functions.
    """

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            hidden_dim: int
    ):
        super().__init__(aggr='add')

        # Message function (edge network)
        self.message_nn = Sequential(
            Linear(2 * node_dim + edge_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, node_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Concatenate node features and edge features
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_nn(msg_input)


class MPNN(torch.nn.Module):
    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            hidden_dim: int,
            out_dim: int = 1,
            num_message_passing: int = 3,
            message_hidden_dim: int = 128,
            set2set_steps: int = 6,
            mlp_hidden_dims: List[int] = None
    ):
        super().__init__()

        self.num_message_passing = num_message_passing

        # Initial node embedding
        self.node_embedding = Linear(node_dim, hidden_dim)

        # Message passing layer
        self.conv = MPNNConv(
            node_dim=hidden_dim,
            edge_dim=edge_dim,
            hidden_dim=message_hidden_dim
        )

        # State update function (GRU)
        self.gru = GRU(hidden_dim, hidden_dim)

        # Set2Set layer
        self.set2set = Set2Set(hidden_dim, processing_steps=set2set_steps)

        # Regressor network with given hidden dimensions
        regressor_layers = []

        # Input is 2 * hidden_dim from Set2Set
        current_dim = 2 * hidden_dim

        # Add layers with specified dimensions
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [hidden_dim]

        for dim in mlp_hidden_dims:
            regressor_layers.append(Linear(current_dim, dim))
            regressor_layers.append(ReLU())
            current_dim = dim

        # Final output layer
        regressor_layers.append(Linear(current_dim, out_dim))

        self.regressor = Sequential(*regressor_layers)

        self.reset_parameters()

    def reset_parameters(self):
        self.node_embedding.reset_parameters()
        self.conv.message_nn[0].reset_parameters()
        self.conv.message_nn[2].reset_parameters()
        self.gru.reset_parameters()
        for layer in self.regressor:
            if isinstance(layer, Linear):
                layer.reset_parameters()

    def forward(self, data):
        x = self.node_embedding(data.x)
        edge_index, edge_attr = data.edge_index, data.edge_attr
        batch = data.batch

        h = x.unsqueeze(0)

        for t in range(self.num_message_passing):
            m = self.conv(x, edge_index, edge_attr)
            m = m.unsqueeze(0)
            h_new, h = self.gru(m, h)
            x = h_new.squeeze(0)

        # Use Set2Set pooling
        out = self.set2set(x, batch)

        # Apply regressor network
        out = self.regressor(out)

        return out
