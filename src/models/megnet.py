# MEGNet
# Implementation inspired by https://github.com/deepchem/deepchem/blob/28195eb49b9962ecc81d47eb87a82dbafc36c5b2/deepchem/models/torch_models/layers.py#L1063
#

from typing import Union
import torch


import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import Set2Set
from torch_geometric.typing import Adj
from torch_scatter import scatter


__all__ = [
    'MEGNet_block',
    'MEGNet'
]


class MEGNet_block(nn.Module):
    """An implementation of the MatErials Graph Network (MEGNet) block by Chen et al. [1]. The MEGNet block
    operates on three different types of input graphs: nodes, edges and global variables. The global
    variables are *graph*-wise, for example, the melting point of a molecule or any other graph level information.

    Note that each element-wise update function is a three-depth MLP, so not many blocks are needed for good
    performance.

    References
    -----------
    [1] Chen, C., Ye, W., Zuo, Y., Zheng, C., & Ong, S. P. (2019). "Graph networks as a universal machine learning
    framework for molecules and crystals". Chemistry of Materials, 31 (9), 3564â3572.
    https://doi.org/10.1021/acs.chemmater.9b01294

    ------

    Parameters
    ----------------
    node_in_dim: int
        The number of input node features.
    edge_in_dim: int
        The number of input edge features.
    node_hidden_dim: int
        The dimension of the hidden node features. Default: 32
    edge_hidden_dim: int
        The dimension of the hidden edge features. Default: 32


    """


    def __init__(self, node_in_dim: int, edge_in_dim:int, node_hidden_dim: int, edge_hidden_dim: int):
        super(MEGNet_block, self).__init__()



        self.update_net_edge = nn.Sequential(
            nn.Linear(node_in_dim * 2 + edge_in_dim, edge_hidden_dim, bias=True),
            nn.Softplus(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim, bias=True),
            nn.Softplus(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim, bias=True),
        )

        self.update_net_node = nn.Sequential(
            nn.Linear(node_in_dim + edge_hidden_dim, node_hidden_dim, bias=True),
            nn.Softplus(),
            nn.Linear(node_hidden_dim, node_hidden_dim, bias=True),
            nn.Softplus(),
            nn.Linear(node_hidden_dim, node_hidden_dim, bias=True),
        )


    def update_edge_feats(self, edge_index: Adj, node_feats, edge_feats,
                          batch) -> Tensor:
        src_index, dst_index = edge_index

        out = torch.cat((node_feats[src_index], node_feats[dst_index],
                         edge_feats), dim=1)

        out = self.update_net_edge(out)

        return out


    def update_node_feats(self, edge_index: Adj, node_feats, edge_feats, batch=None):
        src_index, dst_index = edge_index
        # Compute mean edge features for each node by dst_index (each node
        # receives information from edges which have that node as its destination,
        # hence the computation uses dst_index to aggregate information)
        edge_features_mean_by_node = scatter(src=edge_feats,
                                             index=dst_index,
                                             dim=0,
                                             reduce='sum')
        out = torch.cat(
            (node_feats, edge_features_mean_by_node),dim=1)

        out = self.update_net_node(out)

        return out




    def forward(self, edge_index, x, edge_attr, batch=None):

        if batch is None:
            batch = x.new_zeros(x.size(0),dtype=torch.int64)

        edge_batch_map = batch[edge_index[0]]
        h_e = self.update_edge_feats(node_feats=x, edge_index=edge_index, edge_feats=edge_attr,
                                     batch=edge_batch_map)

        h_n = self.update_node_feats(node_feats=x, edge_index=edge_index, edge_feats=h_e,
                                      batch=batch)

        return h_e, h_n



class MEGNet(nn.Module):
    def __init__(
            self,
            node_in_dim: int,
            edge_in_dim: int,
            out_channels: int = 1,
            node_hidden_dim: int = 64,
            edge_hidden_dim: int = 64,
            depth: int = 2,
            mlp_out_hidden: Union[int, list] = 512,
            rep_dropout: float = 0.0
    ):
        super(MEGNet, self).__init__()

        self.depth = depth
        self.out_channels = out_channels

        self.embed_nodes = nn.Linear(node_in_dim, node_hidden_dim)
        self.embed_edges = nn.Linear(edge_in_dim, edge_hidden_dim)

        self.rep_dropout = nn.Dropout(rep_dropout)

        self.dense_layers_nodes = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_hidden_dim, node_hidden_dim*2),
                nn.ReLU(),
                nn.Linear(node_hidden_dim*2, node_hidden_dim),
            ) for _ in range(depth)])

        self.dense_layers_edges = nn.ModuleList([
            nn.Sequential(
                nn.Linear(edge_hidden_dim, edge_hidden_dim * 2),
                nn.Linear(edge_hidden_dim * 2, edge_hidden_dim),
            ) for _ in range(depth)])

        # Initialize ModuleList for MLP layers
        self.mlp_layers = nn.ModuleList()

        # Convert single int to list if necessary
        if isinstance(mlp_out_hidden, int):
            mlp_out_hidden = [node_hidden_dim * 2 + edge_hidden_dim * 2, mlp_out_hidden, mlp_out_hidden // 2]
        else:
            # Ensure first dimension matches input size
            mlp_out_hidden = [node_hidden_dim * 2 + edge_hidden_dim * 2] + mlp_out_hidden

        # Build MLP layers
        for i in range(len(mlp_out_hidden) - 1):
            self.mlp_layers.append(nn.Linear(mlp_out_hidden[i], mlp_out_hidden[i + 1]))
            self.mlp_layers.append(nn.ReLU())

        # Final output layer
        self.out_layer = nn.Linear(mlp_out_hidden[-1], out_channels)

        self.read_out_nodes = Set2Set(node_hidden_dim, processing_steps=3)
        self.read_out_edges = Set2Set(edge_hidden_dim, processing_steps=3)

        self.blocks = nn.ModuleList([
            MEGNet_block(
                node_in_dim=node_hidden_dim,
                edge_in_dim=edge_hidden_dim,
                node_hidden_dim=node_hidden_dim,
                edge_hidden_dim=edge_hidden_dim
            ) for _ in range(depth)
        ])

    def forward(self, data, return_lats: bool = False) -> Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h_n = self.embed_nodes(x)
        h_e = self.embed_edges(edge_attr)

        for i in range(self.depth):
            h_n = self.dense_layers_nodes[i](h_n)
            h_e = self.dense_layers_edges[i](h_e)

            h_e, h_n = self.blocks[i](edge_index, h_n, h_e, data.batch)

        src_index, dst_index = edge_index
        h_n = self.read_out_nodes(h_n, data.batch)
        h_e = self.read_out_edges(h_e, data.batch[src_index])

        if return_lats:
            return h_n

        out = torch.cat((h_n, h_e), dim=1)
        out = self.rep_dropout(out)

        # Apply MLP layers
        for layer in self.mlp_layers:
            out = layer(out)

        # Apply final output layer
        out = self.out_layer(out)

        return out