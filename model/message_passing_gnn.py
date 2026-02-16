import torch
from torch import nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F

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

def MLP(channels, layer_norm = False):
    layers = []
    for i in range(len(channels)-1):
        layers.append(nn.Linear(channels[i], channels[i+1]))
        if i < len(channels)-2:
            layers.append(nn.GELU())
    if layer_norm: 
        layers.append(nn.LayerNorm(channels[-1]))
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
    def __init__(self, edge_feat_dim, node_feat_dim, hidden_dim, layer_norm = False):
        super().__init__(aggr='add')

        # egde update net: eij' = f1(xi, xj, eij)
        self.edge_net = MLP([edge_feat_dim + 2*node_feat_dim, 
                             hidden_dim, 
                             hidden_dim], )

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

        # node update: embeded_contact_force = g([sum_j m_ij, pos_i]) -> (N, H)
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
        node_force_en = self.node_net(tmp)            # (N, H)
        return node_force_en                          # embedded contact force
    
class GraphNetSurfaceBlockForce(MessagePassing):
    def __init__(self, hidden_dim: int):
        super().__init__(aggr='add')
        self.hidden_dim = hidden_dim
        self.edge_net = MLP([8, 64, 64])
        self.node_net = MLP([64 + 1, hidden_dim, hidden_dim ])

    def forward(self, x, edge_index):
        if edge_index is None or edge_index.numel() == 0:
            return x.new_zeros(x.size(0), self.hidden_dim)
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x = [u(0:3), v(3:6), p0(6:9)]
        p_i = x_i[:, 0:3] + x_i[:, 6:9]
        p_j = x_j[:, 0:3] + x_j[:, 6:9]

        r = p_i - p_j                                # (E, 3)
        d = torch.norm(r, dim=-1, keepdim=True)      # (E, 1)
        r_hat = r / (d + 1e-8)                       # (E, 3)

        v_rel = x_i[:, 3:6] - x_j[:, 3:6]            # (E, 3)
        v_rel_n = (v_rel * r_hat).sum(-1, keepdim=True)  # (E, 1)

        msg = torch.cat([r_hat, d, v_rel, v_rel_n], dim=-1)  # (E, 8)
        return self.edge_net(msg)                      # (E, H)

    def update(self, aggr_out, x):
        aggr = torch.cat([aggr_out, x[:,-1]], dim=-1)
        return self.node_net(aggr)  # (N, H)

class GraphNetSurfaceBlockMHA_Dense(nn.Module):
    """
    Dense MultiHeadAttention with a graph adjacency mask derived from edge_index.

    Complexity: O(N^2) attention per graph (mask just blocks scores, but computation is still dense).
    """

    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Embed node state (here: pos only). You can replace with [pos, vel, material...]
        self.node_in = MLP([3, hidden_dim, hidden_dim])

        # PyTorch MHA: use batch_first=True => (B, N, H)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        # Output 3D update
        self.out_net = MLP([hidden_dim + 3, hidden_dim, 3])

    @staticmethod
    def edge_index_to_attn_mask(edge_index: torch.Tensor, num_nodes: int, device=None):
        """
        Returns attn_mask of shape (N, N) with -inf where attention is NOT allowed.
        By convention: attn_mask[i, j] is applied to score from query i attending to key j.
        """
        device = device or edge_index.device
        # allowed adjacency (query i can attend to key j)
        allowed = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)

        # Allow edges: i attends to j (edge_index[0]=i, edge_index[1]=j)
        i, j = edge_index[0], edge_index[1]
        allowed[i, j] = True

        # Always allow self-attention
        allowed.fill_diagonal_(True)

        # Build additive mask: 0 for allowed, -inf for blocked
        # dtype must be float for additive mask
        attn_mask = torch.full((num_nodes, num_nodes), float("-inf"), device=device)
        attn_mask[allowed] = 0.0
        return attn_mask

    def forward(self, pos: torch.Tensor, edge_index: torch.Tensor):
        """
        pos: (N, 3)
        edge_index: (2, E) with direction i<-j as (i, j) if you want i attend to j.
        returns: (N, 3) update (force or delta-pos)
        """
        N = pos.size(0)
        if edge_index is None or edge_index.numel() == 0:
            return torch.zeros_like(pos)

        x = self.node_in(pos)              # (N, H)
        x = x.unsqueeze(0)                 # (1, N, H) batch size 1

        attn_mask = self.edge_index_to_attn_mask(edge_index, N, device=pos.device)  # (N, N)

        # MHA: query=key=value=x
        out, _ = self.mha(x, x, x, attn_mask=attn_mask, need_weights=False)
        out = out.squeeze(0)               # (N, H)

        tmp = torch.cat([out, pos], dim=-1)  # (N, H+3)
        return self.out_net(tmp)             # (N, 3)