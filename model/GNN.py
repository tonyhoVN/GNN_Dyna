'''
All models for GNN

@author: Anh Tung Ho
'''

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.neighbors import KDTree
from utils.data_loader import GraphData
import numpy as np
from model.message_passing_gnn import *

TIME_SERIES = 3

# =============================================
# Build radius edges using KDTree (CPU)
# =============================================
def build_radius_edges(coords, exclude_edges, radius=0.05):
    """
    coords: (N,3) tensor (either CPU or CUDA)
    exclude_edges: (2,E) tensor on GPU
    radius: float, radius threshold
    returns: edge_index (2,E) and edge_attr (E,1) distance on CPU
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

    src, dst, dist = [], [], []
    for i, neigh in enumerate(neighbors):
        for j in neigh:
            if i != j and (i, j) not in exclude_set:
                src.append(i)
                dst.append(j)
                dist.append(np.linalg.norm(coords_cpu[i] - coords_cpu[j]))

    if len(src) == 0:
        # no contact edges
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(dist, dtype=torch.float).unsqueeze(1)
    return edge_index, edge_attr

# =============================================
# Merge FEM edges + radius edges
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


class TemporalEncoder1(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers = 3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )

        self.fc = MLP([hidden_dim + 1, hidden_dim, hidden_dim])  # +1 for mass feature

    def forward(self, x, mass):
        """
        x: node info (N, F * T)
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
    
class TemporalEncoder(nn.Module):
    def __init__(
            self, 
            in_dim, 
            hidden_dim, 
            n_layers = 3,
            layer_norm = False,
            use_mass = True,
            use_pos = True,
            ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )

        self.use_mass = use_mass
        self.use_pos = use_pos
        extra_dim = (1 if use_mass else 0) + (3 if use_pos else 0)

        self.fc = MLP([hidden_dim + extra_dim, hidden_dim, hidden_dim], layer_norm)  # +1 for mass feature
                                                                 # +3 for position feature
    def forward(self, x, mass, pos):
        """
        x: node info (N, F * T)
        mass: node mass (N,)
        output: temporal node embedded features (N, H)
        """
        # time series node encoder
        x = x.permute(0, 2, 1).contiguous()   # (N, T, F)
        out, (h_n, c_n) = self.lstm(x)        # h_n: (n_layers, N, H)
        h = h_n[-1]                           # (N, H)

        # Add other features
        extras = []
        if self.use_mass:
            extras.append(mass.unsqueeze(-1))
        if self.use_pos:
            extras.append(pos)
        h = torch.cat([h, *extras], dim=-1)  # (N, H+1+3)

        # h = torch.cat([h, pos, mass.unsqueeze(-1)], dim=-1)  # (N, H+1+3)

        return self.fc(h)                     # (N, H)
    
class GRUResidualDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=6, n_layers=2):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.head = MLP([hidden_dim, hidden_dim, out_dim])

    def forward(self, x_seq, dt=None):
        # x_seq: (N, F, T)  -> convert to (N, T, F)
        x_seq = x_seq.permute(0, 2, 1).contiguous()
        N, T, _ = x_seq.shape

        out, h_n = self.gru(x_seq)       # out: (N, T, H)
        h_last = out[:, -1, :]            # (N, H)
        return self.head(h_last)          # (N, 6) residual rates


class NormalEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        self.fc = MLP([in_dim, hidden_dim, hidden_dim])

    def forward(self, x):
        """
        x: node info (N, F)
        output: node embedded features (N, H)
        """
        return self.fc(x)                     # (N, H)

class EdgeEncoder(nn.Module):
    def __init__(self, 
                 num_materials: int, 
                 mat_emb_dim: int, 
                 numeric_dim: int, 
                 out_dim: int,
                 layer_norm = False):
        super().__init__()
        self.mat_emb = nn.Embedding(num_materials, mat_emb_dim)
        self.mlp = MLP([mat_emb_dim + numeric_dim, out_dim, out_dim], layer_norm)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        # edge_attr: (E, 2) -> [material_id, length]
        mat_id = edge_attr[:, 0].long()
        numeric = edge_attr[:, 1:]
        emb = self.mat_emb(mat_id)
        feat = torch.cat([emb, numeric], dim=-1)
        return self.mlp(feat)
 
class MeshGraphNet(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim, latent_dim=128, n_temp_layers=3, n_gnn_layers=4, num_materials=2, mat_emb_dim=4):
        super().__init__()

        # LSTM encoder for node temporal features
        self.node_topo_encoder = TemporalEncoder(
            in_dim = node_dim,
            hidden_dim = latent_dim,
            n_layers = n_temp_layers
        )

        self.node_radius_encoder = MLP([3, latent_dim, latent_dim])

        # Edge encoder (material id embedding + numeric features)
        self.edge_encoder = EdgeEncoder(num_materials=num_materials, mat_emb_dim=mat_emb_dim, numeric_dim=max(edge_dim - 1, 0), out_dim=latent_dim)

        # Message-passing layers
        self.n_gnn_layers = n_gnn_layers
        # self.layers_topo = nn.ModuleList([
        #     GraphNetBlock(
        #         edge_feat_dim=edge_dim,
        #         node_feat_dim=node_dim,
        #         hidden_dim=latent_dim)
        #     for _ in range(n_gnn_layers)
        # ])
        self.layers_topo = GraphNetBlock(
                edge_feat_dim=latent_dim,
                node_feat_dim=latent_dim,
                hidden_dim=latent_dim)

        self.layers_radius = GraphNetBlock(
            edge_feat_dim=1,
            node_feat_dim=latent_dim,
            hidden_dim=latent_dim)

        self.add_passage = MLP([latent_dim + latent_dim, latent_dim])

        # Decode node features (12)
        self.node_decoder = MLP([latent_dim, latent_dim, out_dim])

        # PINN net
        # self.pinn = PhysicsNet(hidden_dim=latent_dim)


    def forward(self, graph: GraphData):
        """
        x: (N, F * T)  e.g., (1940, 12 * 3) feat over time 
        topo_edge_index: (2, E) FEM edges on GPU
        """

        device = graph.x.device
        x = graph.x

        # -----------------------------------
        # 1. Extract coordinates at last time step
        # -----------------------------------
        coords = x[:, -12 : -9]  # (N,3)
        topo_edge_index = graph.edge_index  # (2, E)
        # xv = x[:, -6]  # (N,6) position + velocity

        # -----------------------------------
        # 2. Node and edge encoder
        # -----------------------------------
        h_topo = self.node_topo_encoder(x, graph.node_mass)    # (N, H)
        h_radius = self.node_radius_encoder(coords)   # (N, H)
        edge_feat = self.edge_encoder(graph.edge_attr)  # (E, D)

        # -----------------------------------
        # 3. Build contact edges (KDTree on CPU)
        # -----------------------------------
        radius_edge_index, radius_edge_attr = build_radius_edges(coords, topo_edge_index, radius=15.0)
        radius_edge_index = radius_edge_index.to(device)
        radius_edge_attr = radius_edge_attr.to(device)

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

        # MP over topo edges
        for _ in range(self.n_gnn_layers):
            h_topo, edge_feat = self.layers_topo(h_topo, topo_edge_index, edge_feat)
        
        # MP over radius edges
        # for _ in range(5):
        h_radius, _ = self.layers_radius(h_radius, radius_edge_index, radius_edge_attr)

        # breakpoint()

        # combine both passages
        h = self.add_passage(torch.cat([h_topo, h_radius], dim=1))
        # h = self.add_passage(h_topo + h_radius)

        # -----------------------------------
        # 6. Output prediction
        # -----------------------------------
        delta_pred = self.node_decoder(h)  # (N, out_dim)
        
        return delta_pred
    
    def step(self, x, delta_pred, delta_t):
        return x + delta_pred * delta_t
    
    def loss(self, pred, target):
        return F.mse_loss(pred, target)


class EncodeDecodeGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim, latent_dim=128, n_temp_layers=3, n_gnn_layers=4, num_materials=2, mat_emb_dim=4):
        super().__init__()

        # LSTM encoder for node temporal features
        self.node_topo_encoder = TemporalEncoder(
            in_dim = node_dim,
            hidden_dim = latent_dim,
            n_layers = n_temp_layers
        )

        # Edge encoder (material id embedding + numeric features)
        self.edge_encoder = EdgeEncoder(num_materials=num_materials, mat_emb_dim=mat_emb_dim, numeric_dim=max(edge_dim - 1, 0), out_dim=latent_dim)

        # Message-passing layers
        self.n_gnn_layers = n_gnn_layers
        self.layers_topo = nn.ModuleList([
            GraphNetBlock(
                edge_feat_dim=latent_dim,
                node_feat_dim=latent_dim,
                hidden_dim=latent_dim)
            for _ in range(n_gnn_layers)
        ])

        # Message-passing layers for nodes on surface
        self.layers_surf = GraphNetSurfaceBlock(
            hidden_dim=latent_dim)

        self.add_passage = MLP([latent_dim + latent_dim, latent_dim])

        # Decode node features (12)
        self.node_decoder = MLP([latent_dim, latent_dim, out_dim])

    def forward(self, graph):
        """
        x: (N, F, T)  e.g., (1940, 12, 3) feat over time 
        topo_edge_index: (2, E) FEM edges on GPU
        """

        # -----------------------------------
        # 1. Extract feature at time step t
        # -----------------------------------
        x_t = graph.x[:,:,-1]

        # -----------------------------------
        # 2. Encode node time series feature
        # -----------------------------------
        h_topo = self.node_topo_encoder(graph.x, graph.node_mass)    # (N, H)
        
        # 3. Encode edge feature
        edge_feat = self.edge_encoder(graph.edge_attr)  # (E, D)

        # 4. Message passing for surface nodes -> node force
        h_surf = self.layers_surf(graph.pos, graph.edge_surf_index)

        surface_mask = torch.zeros(h_surf.size(0), device=h_surf.device, dtype=h_surf.dtype)
        surface_mask.index_fill_(0, graph.edge_surf_index.view(-1), 1.0)
        h_final = h_topo + h_surf * surface_mask.unsqueeze(-1)

        # 5. Message passing with neighbor nodes
        for layer in self.layers_topo:
            h_final, edge_feat = layer(h_final, graph.edge_index, edge_feat)
        
        # -----------------------------------
        # 6. Output prediction
        # -----------------------------------
        delta_pred = self.node_decoder(h_final)  # (N, out_dim)
        y_t = x_t + delta_pred * graph.delta_t.unsqueeze(-1)
        
        return y_t
    
    def loss(self, pred, target):
        return F.mse_loss(pred, target) 


class EncodeDecodeGNNGeneral(nn.Module):
    def __init__(self, 
                 node_encoder, 
                 edge_encorder, 
                 gnn_topo, 
                 gnn_surface,
                 node_decoder
    ):
        super().__init__()
        self.node_encoder = node_encoder
        self.edge_encoder = edge_encorder
        self.layers_topo  = gnn_topo
        self.layers_surf  = gnn_surface
        self.node_decoder = node_decoder

    def forward(self, graph):
        # 1. Extract feature at time step t
        x_t = graph.x[:,:,-1] # (N, H)

        # 2. Encode node time series feature
        h_topo = self.node_encoder(graph.x, graph.node_mass, graph.x_initial)    # (N, H)

        # 3. Encode edge feature
        edge_feat = self.edge_encoder(graph.edge_attr)  # (E, D)

        # 5. Message passing with surface nodes for contact force
        if self.layers_surf is None:
            h_surf = torch.zeros_like(h_topo)
        # elif isinstance(self.layers_surf, nn.ModuleList):
        #     if len(self.layers_surf) == 0:
        #         h_surf = torch.zeros_like(h_topo)
        #     else:
        #         h_surf = self.layers_surf[0](graph.pos, graph.edge_surf_index)
        else:
            h_surf = self.layers_surf(graph.pos, graph.edge_surf_index) # (N, H)

        # 6. Combine node features 
        surface_mask = torch.zeros(h_surf.size(0), device=h_surf.device, dtype=h_surf.dtype)
        surface_mask.index_fill_(0, graph.edge_surf_index.view(-1), 1.0)
        h_final = h_topo + h_surf * surface_mask.unsqueeze(-1) # (N, H)

        # 4. Message passing with neighbor nodes for internal force
        for layer in self.layers_topo:
            h_final, edge_feat = layer(h_final, graph.edge_index, edge_feat) # (N, H)

        # 7. Decode and output predict
        delta_pred = self.node_decoder(h_final)  # (N, out_dim)
        y_t = x_t + delta_pred * graph.delta_t.unsqueeze(-1) 

        return y_t
    
    def loss(self, pred, target):
        return F.mse_loss(pred, target)
    
class EncoderDecodeGNNForce(EncodeDecodeGNNGeneral):
    def __init__(self, 
                 node_encoder, 
                 edge_encorder, 
                 gnn_topo,
                 head_topo, 
                 gnn_surface,
                 head_surface,
                 node_decoder
    ):
        super().__init__(node_encoder, 
                 edge_encorder, 
                 gnn_topo, 
                 gnn_surface,
                 node_decoder)
        
        self.head_topo = head_topo
        self.head_surface = head_surface

    def forward(self, graph):
        # 1. Extract feature at time step t
        X_t = graph.x[:,:,-1] # (N, 9)
        a_t = X_t[:, 0:3] # (N, 3) acc
        v_t = X_t[:, 3:6] # (N, 3) vel
        u_t = X_t[:, 6:9] # (N, 3) disp
        x_t = torch.cat([u_t, v_t, graph.x_initial, graph.node_mass.unsqueeze(-1)], dim=-1) # (N, 9)  

        # 2. Encode node feature
        h_node = self.node_encoder(x_t)    # (N, H)
        h_surf = h_node.clone()
        h_topo = h_node.clone()

        # 3. Encode edge feature
        edge_feat = self.edge_encoder(graph.edge_attr)  # (E, D)

        # 4. Message passing with neighbor nodes for internal force
        for layer in self.layers_topo:
            h_topo, edge_feat = layer(h_topo, graph.edge_index, edge_feat) # (N, H)
        
        force_int = self.head_topo(h_topo) # (N, 3)

        # 5. Message passing with surface nodes for contact force
        if self.layers_surf is None:
            h_surf = torch.zeros_like(h_surf)
        else:
            # TODO: select feature for surf
            h_surf = self.layers_surf(x_t, graph.edge_surf_index) # (N, H)

        force_ext = self.head_surface(h_surf) # (N, 3)

        # 6. Combine node features 
        surface_mask = torch.zeros(force_int.size(0), device=force_int.device, dtype=force_int.dtype)
        surface_mask.index_fill_(0, graph.edge_surf_index.view(-1), 1.0)
        force_total = force_int + force_ext * surface_mask.unsqueeze(-1) # (N, 3)

        # 7. Predict acceleration
        # a_t_pred = force_total / (graph.node_mass.unsqueeze(-1) * 1000) #scale to gram
        a_t_pred = force_total
        # Prepare for decode 
        x_for_dec = graph.x.clone()
        x_for_dec[:,0:3,-1] = a_t_pred.detach()


        # 8. Baseline update  
        dt = graph.delta_t.unsqueeze(-1)  # (N,1)
        y_base = self.base_update(u_t, v_t, a_t_pred, dt)

        # 7. Decode residual with time series
        y_t = self.update(x_for_dec, y_base, dt)

        # Loss
        loss = self.loss_total(y_t, graph.y[:,3:], a_t_pred, a_t)
        return y_t, loss
    
    def base_update(self, u_t, v_t, a_t, dt):
        v_t1 = v_t + a_t * dt
        u_t1 = u_t + v_t * dt + 0.5 * a_t * (dt**2)
        return torch.cat([v_t1, u_t1], dim=-1) # (N, 6)

    def update(self, x, y_base, dt):
        delta_y_rate = self.node_decoder(x, dt)
        y_final = y_base + delta_y_rate * dt
        return y_final
    
    def loss_total(self, y_t, Y_t, a_t_pred, a_t):
        L_acc = self.loss(a_t_pred, a_t)
        L_state = self.loss(y_t, Y_t)
        return L_acc + L_state

        

class PhysicsNet(nn.Module):
    """
    PINN net to predict system energies
    """
    def __init__(self, hidden_dim=128, mat_emb_dim: int = 2):
        super().__init__()

        # Potential energy MLP
        self.encode_node = MLP_layer_norm([6, hidden_dim, hidden_dim])
        self.encode_element = MLP([hidden_dim + mat_emb_dim, hidden_dim, 1])
        self.mat_emb = nn.Embedding(2, mat_emb_dim)

    def forward(self, graph: GraphData):
        x = graph.x
        if x.dim() == 2:
            if x.shape[1] % 12 != 0:
                raise ValueError(f"Expected x feature dim to be multiple of 12, got {x.shape[1]}")
            num_series = x.shape[1] // 12
            x = x.reshape(x.shape[0], 12, num_series)
        node_feat = x[:, :, -1]  # (N, F)
        x_0 = graph.x_initial  # (N, 3)
        v = node_feat[:, 6:9]  # (N, 3) velocities
        u = node_feat[:, 9:]    # (N, 3) displacement
        m = graph.node_mass    # (N, ) mass

        # Kinetic energy
        kinetic_energy = self.global_kinetic_energy(m, v)  # scalar

        # Potential energy 
        if graph.element_node_ids is not None and graph.element_material_ids is not None:
            internal_energy = self.global_internal_energy_fast(
                u,
                x_0,
                graph.element_node_ids,
                graph.element_node_mask,
                graph.element_material_ids,
            )
        else:
            element_to_nodes = graph.element_to_nodes  # {element_id: [node_ids, ]}
            element_materials = graph.element_materials  # {element_id: material_id}
            internal_energy = self.global_internal_energy(element_to_nodes, element_materials, u, x_0)  # scalar
        return kinetic_energy, internal_energy
    
    def global_kinetic_energy(self, m, v):
        K = 0.5 * m * (v.pow(2).sum(dim=1))
        return K.sum()
    
    def global_internal_energy(self, element_to_nodes, element_materials, u, x_0):
        U = torch.zeros((), device=u.device)

        # node_latent = self.encode_node(u)  # (N, H)
        for element_id, node_ids in element_to_nodes.items():
            u_e = u[node_ids]            # (k, 3)
            x0_e = x_0[node_ids]         # (k, 3)

            u_rel = u_e - u_e.mean(dim=0, keepdim=True)
            x0_rel = x0_e - x0_e.mean(dim=0, keepdim=True)

            features = torch.cat([u_rel, x0_rel], dim=1)     # (k, 6)
            element_coords = self.encode_node(features).mean(dim=0)  # (H,)

            mat_id = torch.as_tensor(element_materials[element_id],
                                    device=u.device, dtype=torch.long)
            mat_feat = self.mat_emb(mat_id-1)                  # (mat_emb_dim,)

            element_feat = torch.cat([element_coords, mat_feat], dim=0)
            element_ie = self.encode_element(element_feat)   # (1,)

            U = U + F.softplus(element_ie).squeeze()

        return U

    def global_internal_energy_fast(self, u, x_0, element_node_ids, element_node_mask, element_material_ids):
        safe_ids = element_node_ids.clamp(min=0)
        u_e = u[safe_ids]     # (E, K, 3)
        x0_e = x_0[safe_ids]  # (E, K, 3)
        if element_node_mask is not None:
            mask = element_node_mask.unsqueeze(-1)
            u_e = u_e * mask
            x0_e = x0_e * mask
            counts = element_node_mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)
            u_mean = u_e.sum(dim=1, keepdim=True) / counts
            x0_mean = x0_e.sum(dim=1, keepdim=True) / counts
        else:
            u_mean = u_e.mean(dim=1, keepdim=True)
            x0_mean = x0_e.mean(dim=1, keepdim=True)

        u_rel = u_e - u_mean
        x0_rel = x0_e - x0_mean

        features = torch.cat([u_rel, x0_rel], dim=2)  # (E, K, 6)
        elem_latent = self.encode_node(features.reshape(-1, 6)).view(features.size(0), features.size(1), -1)
        if element_node_mask is not None:
            elem_latent = elem_latent * element_node_mask.unsqueeze(-1)
            elem_latent = elem_latent.sum(dim=1) / counts.squeeze(-1)
        else:
            elem_latent = elem_latent.mean(dim=1)

        mat_feat = self.mat_emb(element_material_ids.long() - 1)
        element_feat = torch.cat([elem_latent, mat_feat], dim=1)
        element_ie = self.encode_element(element_feat).squeeze(-1)
        return F.softplus(element_ie).sum()

    def loss(self, kinetic_energy_pred, internal_energy_pred, kinetic_energy_true, internal_energy_true):
        # delta_x_pre = delta_pred[:, 3]
        # delta_disp_pre = delta_pred[:, -3:]
        # loss_consistence = F.mse_loss(delta_x_pre, delta_disp_pre[:, 0])
        loss_kinetic = F.l1_loss(kinetic_energy_pred, kinetic_energy_true)
        loss_internal = F.l1_loss(internal_energy_pred, internal_energy_true)
        return loss_kinetic + loss_internal

if __name__ == "__main__":
    in_dim = 6
    hidden_dim = 10
    x = torch.randn(100, 6, 3)
    print(f"x shape: {x[:,:,-1].shape}")
    net = TemporalEncoder(in_dim, hidden_dim)
    y = net(x)
    print(y.shape)

