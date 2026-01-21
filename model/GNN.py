import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
from sklearn.neighbors import KDTree
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import add_self_loops, degree
from utils.data_loader import GraphData

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

# =============================================
# 1) Simple MLP
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
# 2) Build radius edges using KDTree (CPU)
# =============================================
def build_radius_edges(coords, exclude_edges, radius=0.05):
    """
    coords: (N,3) tensor (either CPU or CUDA)
    exclude_edges: (2,E) tensor on GPU
    radius: float, radius threshold
    returns: edge_index on CPU (2,E)
    """
    coords_cpu = coords.detach().cpu().numpy()
    exclude_edges_cpu = exclude_edges.detach().cpu().numpy()
    exclude_set = set()
    if exclude_edges_cpu.size != 0:
        for i in range(exclude_edges_cpu.shape[1]):
            src = int(exclude_edges_cpu[0, i])
            dst = int(exclude_edges_cpu[1, i])
            exclude_set.add((src, dst))
            exclude_set.add((dst, src))
    tree = KDTree(coords_cpu)

    # Query neighbors
    neighbors = tree.query_radius(coords_cpu, r=radius)

    src, dst = [], []
    for i, neigh in enumerate(neighbors):
        for j in neigh:
            if i != j and (i, j) not in exclude_set:
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


class GraphNetBlock(MessagePassing):
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
            return x    # No neighbors → no message passing
        
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
    

class TemporalEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers = 3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )

        self.fc = MLP_layer_norm([hidden_dim + 1, hidden_dim, hidden_dim])  # +1 for mass feature

    def forward(self, x, mass):
        """
        x: node info (N, F, T)
        mass: node mass (N,)
        output: temporal node embedded features (N, H)
        """
        # time series node encoder
        x = x.permute(0, 2, 1).contiguous()   # (N, T, F)
        out, (h_n, c_n) = self.lstm(x)        # h_n: (n_layers, N, H)
        h = h_n[-1]                           # (N, H)

        # Add mass as feature
        h = torch.cat([h, mass.unsqueeze(-1)], dim=-1)  # (N, H+1)

        return self.fc(h)                     # (N, H)

 
class MeshGraphNet(nn.Module):
    def __init__(self, node_dim, edge_dim , latent_dim=128, n_temp_layers=3, n_gnn_layers=4, out_dim=12):
        super().__init__()

        # LSTM encoder for node temporal features
        self.node_encoder = TemporalEncoder(
            in_dim = node_dim,
            hidden_dim = latent_dim,
            n_layers = n_temp_layers
        )

        # Edge encoder
        self.edge_encoder = MLP_layer_norm([edge_dim, latent_dim, latent_dim])

        # Message-passing layers
        self.n_temp_layers = n_temp_layers
        # self.layers_topo = nn.ModuleList([
        #     GraphNetBlock(
        #         edge_feat_dim=edge_dim,
        #         node_feat_dim=node_dim,
        #         hidden_dim=latent_dim)
        #     for _ in range(n_gnn_layers)
        # ])
        self.layers_topo = GraphNetBlock(
                edge_feat_dim=edge_dim,
                node_feat_dim=node_dim,
                hidden_dim=latent_dim)

        self.layers_radius = GraphNetBlock(
            edge_feat_dim=edge_dim,
            node_feat_dim=node_dim,
            hidden_dim=latent_dim)

        self.add_passage = MLP_layer_norm([latent_dim*2, latent_dim])

        # Decode node features (12)
        self.node_decoder = MLP_layer_norm([latent_dim, latent_dim, out_dim])

        # PINN net
        # self.pinn = PhysicsNet(hidden_dim=latent_dim)


    def forward(self, graph: GraphData):
        """
        x: (N, F, T)  e.g., (1940, 12, 3) feat over time 
        topo_edge_index: (2, E) FEM edges on GPU
        """

        device = graph.x.device

        # -----------------------------------
        # 1. Extract coordinates at last time step
        # -----------------------------------
        coords = graph.x[:, :3, -1]  # (N,3)
        topo_edge_index = graph.edge_index  # (2, E)

        # -----------------------------------
        # 2. Node and edge encoder
        # -----------------------------------
        h0 = self.node_encoder(graph.x, graph.node_mass)    # (N, H)
        h_topo = h0                 # (N, H)
        h_radius = h_topo.clone()   # (N, H)
        edge_feat = self.edge_encoder(graph.edge_attr)  # (E, D)

        # -----------------------------------
        # 3. Build contact edges (KDTree on CPU)
        # -----------------------------------
        radius_edges = build_radius_edges(coords, topo_edge_index, radius=2.0)
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

        # for layer in self.layers_topo:
        #     h_topo = layer(h_topo, topo_edge_index)

        for _ in range(self.n_temp_layers):
            h_topo = self.layers_topo(h_topo, topo_edge_index)

        h_radius = self.layers_radius(h_radius, radius_edges)

        # combine both passages
        h = self.add_passage(torch.cat([h_topo, h_radius], dim=1))

        # -----------------------------------
        # 6. Output prediction
        # -----------------------------------
        return self.node_decoder(h)
    
    def loss(self, pred, target):
        return F.mse_loss(pred, target)

class PhysicsNet(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()

        # Potential energy MLP
        self.encode_node = MLP_layer_norm([3, hidden_dim, hidden_dim])
        self.encode_element = MLP_layer_norm([hidden_dim + 1, hidden_dim, 1])

    def forward(self, graph: GraphData):
        node_feat = graph.x[:, :, -1]  # (N, F)
        x = node_feat[:, :3]    # (N, 3) coordinates
        v = node_feat[:, 3:6]  # (N, 3) velocities
        m = graph.node_mass.to(node_feat.device)    # (N, ) mass

        # Kinetic energy
        kinetic_energy = self.global_kinetic_energy(m, v)  # scalar

        # Potential energy 
        element_to_nodes = graph.element_to_nodes  # {element_id: [node_ids, ]}
        element_materials = graph.element_materials  # {element_id: material_id}

        internal_energy = self.global_internal_energy(element_to_nodes, element_materials, x)
        return kinetic_energy, internal_energy
    
    def global_kinetic_energy(self, m, v):
        K = 0.5 * m * torch.norm(v, dim=1, keepdim=True) ** 2
        return K.sum()
    
    def global_internal_energy(self, element_to_nodes, element_materials, x):
        U = torch.zeros((), device=x.device)

        node_latent = self.encode_node(x)  # (N, H)
        for element_id, node_ids in element_to_nodes.items():
            element_coords = node_latent[node_ids].sum(dim=0)  # (,H)
            element_material = element_materials[element_id] # scalar
            element_feat = torch.cat([element_coords, element_material], dim=0)
            # Compute element-level potential energy
            element_pe = self.encode_element(element_feat)  # (1,)

            U += element_pe.squeeze()

        return U

    def loss(self, kinetic_energy_pred, internal_energy_pred, kinetic_energy_true, internal_energy_true):
        # delta_x_pre = delta_pred[:, 3]
        # delta_disp_pre = delta_pred[:, -3:]
        # loss_consistence = F.mse_loss(delta_x_pre, delta_disp_pre[:, 0])
        loss_kinetic = F.mse_loss(kinetic_energy_pred, kinetic_energy_true)
        loss_internal = F.mse_loss(internal_energy_pred, internal_energy_true)
        return loss_kinetic + loss_internal

if __name__ == "__main__":
    in_dim = 6
    hidden_dim = 10
    x = torch.randn(100, 6, 3)
    print(f"x shape: {x[:,:,-1].shape}")
    net = TemporalEncoder(in_dim, hidden_dim)
    y = net(x)
    print(y.shape)

