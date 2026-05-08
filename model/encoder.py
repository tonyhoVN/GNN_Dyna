from model.message_passing_gnn import MLP
import torch.nn as nn
import torch

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

def compute_distance_feature(pos: torch.Tensor, edge_index: torch.Tensor):
    if edge_index is None or edge_index.numel() == 0:
        return pos.new_zeros((0, pos.shape[1] + 1))  # (0, 4) for r_hat(3) + d(1)

    src, dst = edge_index[0], edge_index[1]
    r = pos[dst] - pos[src]
    d = torch.norm(r, dim=-1, keepdim=True)
    return r, d

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
            use_boundary = True
            ):
        super().__init__()

        lstm_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=lstm_dim,
            num_layers=n_layers,
            batch_first=True
        )

        self.use_mass = use_mass
        self.use_pos = use_pos
        self.use_boundary = use_boundary
        extra_dim = (1 if use_boundary else 0) + (1 if use_mass else 0) + (3 if use_pos else 0)

        self.fc = MLP([lstm_dim + extra_dim, hidden_dim, hidden_dim], layer_norm)  # +1 for mass feature
                                                                 # +3 for position feature
    def forward(self, x, mass, pos, boundary):
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
        if self.use_boundary:
            extras.append(boundary.unsqueeze(-1))
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


class TopologyEdgeEncoder(nn.Module):
    def __init__(self, edge_feat_dim: int, hidden_dim: int, layer_norm: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = MLP([edge_feat_dim + 8, hidden_dim, hidden_dim], layer_norm)

    def forward(self, edge_attr: torch.Tensor, pos0: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        if edge_index is None or edge_index.numel() == 0:
            return edge_attr.new_zeros((0, self.hidden_dim))

        src, dst = edge_index[0], edge_index[1]

        r0 = pos0[dst] - pos0[src]
        d0 = torch.norm(r0, dim=-1, keepdim=True)
        r0_hat = r0 / (d0 + 1e-8)

        r = pos[dst] - pos[src]
        d = torch.norm(r, dim=-1, keepdim=True)
        r_hat = r / (d + 1e-8)

        edge_feat = torch.cat([r0_hat, d0, r_hat, d, edge_attr], dim=-1)
        return self.mlp(edge_feat)
    
class SurfaceEdgeEncoder(nn.Module):
    def __init__(self, hidden_dim: int, threshold: float = 22.0, layer_norm: bool = False):
        super().__init__()
        self.threshold  = threshold
        self.hidden_dim = hidden_dim
        # r_hat(3) + d(1) + v_rel(3) + v_normal_mag(1) + v_tangential_mag(1) = 9
        self.mlp = MLP([7, hidden_dim, hidden_dim], layer_norm)

    def forward(self, pos: torch.Tensor, vel: torch.Tensor, edge_surf_index: torch.Tensor):
        if edge_surf_index is None or edge_surf_index.numel() == 0:
            return pos.new_zeros((0, self.hidden_dim)), edge_surf_index

        src, dst = edge_surf_index[0], edge_surf_index[1]

        # Relative position 
        r = pos[src] - pos[dst]                          # (E, 3)
        d = torch.norm(r, dim=-1)                        # (E,)

        # Filter edges within threshold
        keep           = d <= self.threshold             # (E,) bool mask
        edge_surf_index = edge_surf_index[:, keep]       # (2, E_keep)

        if edge_surf_index.numel() == 0:
            return pos.new_zeros((0, self.hidden_dim)), edge_surf_index

        src, dst = edge_surf_index[0], edge_surf_index[1]

        # Contact features 
        r              = r[keep]                         # (E_keep, 3)
        d              = d[keep].unsqueeze(-1)           # (E_keep, 1)
        r_hat = r / (d + 1e-8)                          # (E_keep, 3)
        # gap   = d - self.threshold                       # (E_keep, 1) negative = penetrating

        # Relative velocity decomposition
        v_rel            = vel[src] - vel[dst]                            # (E_keep, 3)
        # v_normal_mag     = (v_rel * r_hat).sum(dim=-1, keepdim=True)     # (E_keep, 1)
        # v_normal         = v_normal_mag * r_hat                          # (E_keep, 3)
        # v_tangential_mag = torch.norm(v_rel - v_normal, dim=-1, keepdim=True)  # (E_keep, 1)

        # Concatenate
        edge_surf_feat = torch.cat([
            r_hat,            # (E_keep, 3)
            d,                # (E_keep, 1)
            v_rel,            # (E_keep, 3)
            # v_normal_mag,     # (E_keep, 1)
            # v_tangential_mag, # (E_keep, 1)
        ], dim=-1)            # (E_keep, 9)

        return self.mlp(edge_surf_feat), edge_surf_index  # return filtered index too