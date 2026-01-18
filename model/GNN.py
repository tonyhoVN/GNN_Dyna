import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
from sklearn.neighbors import KDTree


# =============================================
# 1) Simple MLP
# =============================================
def MLP(channels):
    layers = []
    for i in range(len(channels)-1):
        layers.append(nn.Linear(channels[i], channels[i+1]))
        if i < len(channels)-2:
            layers.append(nn.ReLU())
    layers.append(nn.LayerNorm(channels[-1]))
    return nn.Sequential(*layers)

# =============================================
# 2) Build radius edges using KDTree (CPU)
# =============================================
def build_radius_edges(coords, radius=0.05):
    """
    coords: (N,3) tensor (either CPU or CUDA)
    returns: edge_index on CPU (2,E)
    """
    coords_cpu = coords.detach().cpu().numpy()
    tree = KDTree(coords_cpu)

    # Query neighbors
    neighbors = tree.query_radius(coords_cpu, r=radius)

    src, dst = [], []
    for i, neigh in enumerate(neighbors):
        for j in neigh:
            if i != j:
                src.append(i)
                dst.append(j)

    if len(src) == 0:
        # no contact edges
        return torch.empty((2,0), dtype=torch.long)

    return torch.tensor([src, dst], dtype=torch.long)

# =============================================
# 3) Merge FEM edges + radius edges
# =============================================
def combine_edges(topo_edge_index, radius_edge_index):
    """
    topo_edge_index: (2,E1) on GPU
    radius_edge_index: (2,E2) on CPU
    """
    radius_edge_index = radius_edge_index.to(topo_edge_index.device)

    if radius_edge_index.numel() == 0:
        return topo_edge_index

    merged = torch.cat([topo_edge_index, radius_edge_index], dim=1)
    merged = torch.unique(merged, dim=1)
    return merged

# =============================================
# 4) Residual Message Passing Layer
#     message = MLP(x_i - x_j)
#     update  = x_i + LayerNorm(mean(messages))
# =============================================
class ResidualContactLayer(MessagePassing):
    def __init__(self, node_dim):
        super().__init__(aggr='mean')
        self.msg_mlp = MLP([node_dim, node_dim])
        self.ln = nn.LayerNorm(node_dim)

    def forward(self, x, edge_index):
        # ---- SAFE NO-EDGE CASE ----
        if edge_index.numel() == 0:
            return x    # No neighbors → no message passing
        
        # Normal message passing
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.msg_mlp(x_i - x_j)

    def update(self, aggr_out, x):
        return x + self.ln(aggr_out)

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
            return x    # No neighbors → no message passing
        
        # Redidual node update
        re_node_feat = self.propagate(edge_index, x=x, edge_attr=edge_feat)
        
        # Redidual edge update 
                # Edge update
        row, col = edge_index
        new_edge_features = self.edge_net(torch.cat([x[row], x[col], edge_feat], dim=-1))

    def message(self, x_i, x_j, edge_attr):            
        msg = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_net(msg) 

    def update(self, aggr_out, x):
        # aggr_out: (N, H)
        tmp = torch.cat([aggr_out, x], dim=-1) 
        return self.node_net(tmp)
    

class TemporalEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers = 3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_dim,     # 12 features
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )

        self.fc = MLP([hidden_dim]*2)

    def forward(self, x):
        """
        x: node info (N, F, T)
        output: temporal node embedded features (N, H)
        """
        x = x.permute(0, 2, 1).contiguous()   # (N, T, F)
        out, (h_n, c_n) = self.lstm(x)        # h_n: (n_layers, N, H)
        h = h_n[-1]                           # (N, H)
        return self.fc(h)                     # (N, H)
        

class MeshGraphNet(nn.Module):
    def __init__(self, in_dim, latent_dim=128, n_layers=4, out_dim=12):
        super().__init__()

        # LSTM encoder for temporal features
        self.temporal_en = TemporalEncoder(
            in_dim = in_dim,
            hidden_dim = latent_dim,
        )

        # Message-passing layers
        self.layers_topo = nn.ModuleList([
            ResidualContactLayer2(latent_dim)
            for _ in range(n_layers)
        ])

        self.layers_radius = ResidualContactLayer2(latent_dim)

        self.add_passage = MLP([latent_dim*2, latent_dim])

        # Decoder back to output space (12)
        self.decoder = MLP([latent_dim, latent_dim, out_dim])

    def forward(self, x, topo_edge_index):
        """
        x: (N, F, T)  e.g., (1940, 12, 3)
        topo_edge_index: (2, E) FEM edges on GPU
        """

        device = x.device

        # -----------------------------------
        # 1. Extract coordinates at last time step
        # -----------------------------------
        coords = x[:, :3, -1].to(device)  # (N,3)

        # -----------------------------------
        # 2. LSTM temporal encoder
        # -----------------------------------
        h0 = self.temporal_en(x)    # (N, H)
        h_topo = h0                 # (N, H)
        h_radius = h_topo.clone()   # (N, H)

        # -----------------------------------
        # 3. Build contact edges (KDTree on CPU)
        # -----------------------------------
        radius_edges = build_radius_edges(coords, radius=2.0)
        radius_edges = radius_edges.to(device)

        # # -----------------------------------
        # # 4. Combine edges
        # # -----------------------------------
        # full_edges = combine_edges(topo_edge_index, radius_edges)

        # # -----------------------------------
        # # 5. Message Passing
        # # -----------------------------------
        # if full_edges.numel() == 0:
        #     # no neighbors at all (rare)
        #     return self.decoder(h)

        for layer in self.layers_topo:
            h_topo = layer(h_topo, topo_edge_index)

        h_radius = self.layers_radius(h_radius, radius_edges)

        # combine both passages
        h = self.add_passage(torch.cat([h_topo, h_radius], dim=1))

        # -----------------------------------
        # 6. Output prediction
        # -----------------------------------
        return self.decoder(h)
    


if __name__ == "__main__":
    in_dim = 6
    hidden_dim = 10
    x = torch.randn(100, 6, 3)
    net = TemporalEncoder(in_dim, hidden_dim)
    y = net(x)
    print(y.shape)

