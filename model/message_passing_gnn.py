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
            layers.append(nn.GELU())
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
    
class GraphNetSurfaceBlock(MessagePassing):
    def __init__(self, hidden_dim: int):
        super().__init__(aggr='add')

        # edge message: m_ij = f([dx, dy, dz, ||d||]) -> (E, H)
        self.edge_net = MLP([4, hidden_dim, hidden_dim])

        # node update: Δpos_i = g([sum_j m_ij, pos_i]) -> (N, 3)
        self.node_net = MLP([hidden_dim + 3, hidden_dim, hidden_dim])

    def forward(self, pos, edge_index):
        # pos: (N, 3), edge_index: (2, E)
        if edge_index is None or edge_index.numel() == 0:
            return pos
        return self.propagate(edge_index, pos=pos)

    def message(self, pos_i, pos_j):
        dist = pos_i - pos_j                          # (E, 3)
        r = torch.norm(dist, dim=-1, keepdim=True)    # (E, 1)
        msg = torch.cat([dist, r], dim=-1)            # (E, 4)
        return self.edge_net(msg)                     # (E, H)

    def update(self, aggr_out, pos):
        # aggr_out: (N, H)
        tmp = torch.cat([aggr_out, pos], dim=-1)      # (N, H+3)
        node_force_en = self.node_net(tmp)                     # (N, 3)
        return node_force_en                             # residual position update
