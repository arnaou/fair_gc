import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, LayerNorm, Dropout
from typing import Dict, List, Optional
from torch_geometric.nn import MessagePassing, Set2Set, AttentionalAggregation

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


class EdgeNetwork(torch.nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = Sequential(
            Linear(2 * node_dim + edge_dim, hidden_dim),
            LayerNorm(hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, node_dim)
        )

    def forward(self, x):
        return self.network(x)


class AttentionMPNNConv(MessagePassing):
    """
    Enhanced Message Passing layer with attention mechanism and residual connections
    """

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            hidden_dim: int,
            num_heads: int = 4,
            dropout: float = 0.1
    ):
        super().__init__(aggr='add', node_dim=0)

        self.message_nn = EdgeNetwork(node_dim, edge_dim, hidden_dim, dropout)

        # Attention mechanism
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization for residual connections
        self.layer_norm1 = LayerNorm(node_dim)
        self.layer_norm2 = LayerNorm(node_dim)

        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        # First message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Residual connection and layer norm
        out = self.layer_norm1(x + self.dropout(out))

        # Self-attention
        attended_out, _ = self.attention(out.unsqueeze(0), out.unsqueeze(0), out.unsqueeze(0))
        attended_out = attended_out.squeeze(0)

        # Second residual connection and layer norm
        out = self.layer_norm2(out + self.dropout(attended_out))

        return out

    def message(self, x_i, x_j, edge_attr):
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_nn(msg_input)


class ImprovedMPNN(torch.nn.Module):
    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            hidden_dim: int,
            out_dim: int = 1,
            num_message_passing: int = 3,
            message_hidden_dim: int = 128,
            num_heads: int = 4,
            dropout: float = 0.1,
            set2set_steps: int = 6,
            mlp_hidden_dims: Optional[List[int]] = None
    ):
        super().__init__()

        self.num_message_passing = num_message_passing
        self.dropout = dropout

        # Initial node embedding with layer norm
        self.node_embedding = Sequential(
            Linear(node_dim, hidden_dim),
            LayerNorm(hidden_dim),
            ReLU(),
            Dropout(dropout)
        )

        # Message passing layers with different parameters
        self.convs = torch.nn.ModuleList([
            AttentionMPNNConv(
                node_dim=hidden_dim,
                edge_dim=edge_dim,
                hidden_dim=message_hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_message_passing)
        ])

        # State update function (GRU with layer norm)
        self.gru = GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gru_norm = LayerNorm(hidden_dim)

        # Global attention pooling as alternative to Set2Set
        self.global_attention = AttentionalAggregation(
            gate_nn=Sequential(
                Linear(hidden_dim, hidden_dim),
                LayerNorm(hidden_dim),
                ReLU(),
                Linear(hidden_dim, 1)
            )
        )

        # Set2Set pooling
        self.set2set = Set2Set(hidden_dim, processing_steps=set2set_steps)

        # Calculate input dimension for regressor
        # Set2Set doubles the hidden_dim, and we add hidden_dim from global attention
        regressor_input_dim = 3 * hidden_dim  # 2 * hidden_dim (Set2Set) + hidden_dim (GlobalAttention)

        # Regressor network with residual connections
        regressor_layers = []
        current_dim = regressor_input_dim  # Use calculated input dimension

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [hidden_dim, hidden_dim // 2]

        for dim in mlp_hidden_dims:
            regressor_layers.extend([
                Linear(current_dim, dim),
                LayerNorm(dim),
                ReLU(),
                Dropout(dropout)
            ])
            current_dim = dim

        regressor_layers.append(Linear(current_dim, out_dim))
        self.regressor = Sequential(*regressor_layers)

    def forward(self, data):
        x = self.node_embedding(data.x)
        edge_index, edge_attr = data.edge_index, data.edge_attr
        batch = data.batch

        # Get batch size from batch tensor
        if batch is None:
            batch_size = 1
        else:
            batch_size = int(batch.max()) + 1

        # Initial hidden state with correct dimensions
        h = torch.zeros(1, batch_size, self.gru.hidden_size, device=x.device)

        # Message passing with residual connections
        for conv in self.convs:
            x_new = conv(x, edge_index, edge_attr)
            # Residual connection - avoid inplace addition
            x = x + x_new if x.shape == x_new.shape else x_new

            # Process each batch separately
            x_list = []
            h_new = h.clone()  # Create a new tensor for hidden state updates

            for b in range(batch_size):
                batch_mask = batch == b
                # Get nodes for this batch
                x_batch = x[batch_mask]
                # Process through GRU (add batch and sequence dimensions)
                x_batch = x_batch.unsqueeze(0)  # Add batch dimension
                _, h_batch = self.gru(x_batch, h[:, b:b + 1, :].clone())  # Clone to avoid inplace ops
                # Update hidden state for this batch
                h_new[:, b:b + 1, :] = h_batch
                # Store processed features
                x_list.append(h_batch[0, 0].expand(torch.sum(batch_mask), -1))

            # Update hidden state
            h = h_new

            # Combine all batches back together
            x = torch.cat(x_list, dim=0)
            x = self.gru_norm(x)

        # Combine both pooling methods
        set2set_out = self.set2set(x, batch)
        attention_out = self.global_attention(x, batch)

        # Concatenate different pooling results
        out = torch.cat([set2set_out, attention_out], dim=-1)

        # Apply regressor network
        out = self.regressor(out)

        return out

    def get_attention_weights(self, data):
        """Method to extract attention weights for interpretability"""
        attention_weights = []
        x = self.node_embedding(data.x)

        for conv in self.convs:
            _, weights = conv.attention(
                x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0)
            )
            attention_weights.append(weights)

        return attention_weights