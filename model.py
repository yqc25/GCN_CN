import math

import torch
import torch_geometric
import networkx as nx
import torch.nn.functional as F
from torch_geometric.utils import degree, add_self_loops, from_networkx, to_networkx
from torch_geometric.nn import GATConv, GINConv, GCNConv, MessagePassing, GraphSAGE
from torch_geometric.data import Data


class GCNCN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bias = torch.nn.Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, cns):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # norm = deg_inv_sqrt[col]

        #######
        norm = norm * torch.exp(cns)
        #######

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


class HGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.5):
        super(HGCNEncoder, self).__init__()
        self.conv1 = GCNCN(in_channels, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self, x, edge_index, cns):
        x = self.conv1(x, edge_index, cns)

        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, dropout=0.5):
        super(EdgeDecoder, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, z, edge):
        x = z[edge[0]] * z[edge[1]]
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.mish(x)
        x = self.fc2(x)
        x = self.dropout(x)
        probs = F.sigmoid(x)
        return probs


class HGAE(torch.nn.Module):
    def __init__(self, encoder, edge_decoder):
        super(HGAE, self).__init__()
        self.encoder = encoder
        self.edge_decoder = edge_decoder

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.edge_decoder.reset_parameters()

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)
