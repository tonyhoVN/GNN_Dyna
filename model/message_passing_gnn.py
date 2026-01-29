import torch
from torch import nn
from torch_geometric.nn import MessagePassing


# =============================================
# Simple MLP
# =============================================
def MLP_layer_norm(channels):
    layers = []
    for i in range(len(channels)-1):
        layers.append(nn.Linear(channels[i], channels[i+1]))
        if i < len(channels)-2:
            layers.append(nn.ReLU())
    layers.append(nn.LayerNorm(channels[-1]))
    return nn.Sequential(*layers)

def MLP(channels):
    layers = []
    for i in range(len(channels)-1):
        layers.append(nn.Linear(channels[i], channels[i+1]))
        if i < len(channels)-2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

# =============================================
# GNN message passing block
# =============================================
class GraphNetBlockLayerNorm(MessagePassing):
    def __init__(self, edge_feat_dim, node_feat_dim, hidden_dim):
        super().__init__(aggr='add')

        # egde update net: eij' = f1(xi, xj, eij)
        self.edge_net = MLP_layer_norm([edge_feat_dim + 2*node_feat_dim, 
                             hidden_dim, 
                             hidden_dim])

        # redidual node update net: xi' = xi + f2(xi, sum(eij')) 
        self.node_net = MLP_layer_norm([hidden_dim + node_feat_dim, 
                             hidden_dim,
                             hidden_dim])
        
    def forward(self, x, edge_index, edge_feat):
        # ---- SAFE NO-EDGE CASE ----
        if edge_index.numel() == 0:
            return x, edge_feat    # No neighbors → no message passing
        
        # Redidual node update
        re_node_feat = self.propagate(edge_index, x=x, edge_attr=edge_feat)
        
        # Redidual edge update 
                # Edge update
        row, col = edge_index
        re_edge_features = self.edge_net(torch.cat([x[row], x[col], edge_feat], dim=-1))
        
        # Update
        x = x + re_node_feat
        edge_feat = edge_feat + re_edge_features

        return x, edge_feat

    def message(self, x_i, x_j, edge_attr):            
        msg = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_net(msg) 

    def update(self, aggr_out, x):
        # aggr_out: (N, H)
        tmp = torch.cat([aggr_out, x], dim=-1) 
        return self.node_net(tmp)

class GraphNetBlock(MessagePassing):
    def __init__(self, edge_feat_dim, node_feat_dim, hidden_dim):
        super().__init__(aggr='add')

        # egde update net: eij' = f1(xi, xj, eij)
        self.edge_net = MLP([edge_feat_dim + 2*node_feat_dim, 
                             hidden_dim, 
                             hidden_dim])

        # redidual node update net: xi' = xi + f2(xi, sum(eij')) 
        self.node_net = MLP([hidden_dim + node_feat_dim, 
                             hidden_dim,
                             hidden_dim])

    def forward(self, x, edge_index, edge_feat):
        # ---- SAFE NO-EDGE CASE ----
        if edge_index.numel() == 0:
            return x, edge_feat    # No neighbors → no message passing
        
        # Redidual node update
        re_node_feat = self.propagate(edge_index, x=x, edge_attr=edge_feat)
        
        # Redidual edge update 
                # Edge update
        row, col = edge_index
        re_edge_features = self.edge_net(torch.cat([x[row], x[col], edge_feat], dim=-1))
        
        # Update
        x = x + re_node_feat
        edge_feat = edge_feat + re_edge_features

        return x, edge_feat

    def message(self, x_i, x_j, edge_attr):            
        msg = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_net(msg) 

    def update(self, aggr_out, x):
        # aggr_out: (N, H)
        tmp = torch.cat([aggr_out, x], dim=-1) 
        return self.node_net(tmp)