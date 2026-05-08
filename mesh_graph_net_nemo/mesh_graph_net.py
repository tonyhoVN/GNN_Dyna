import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import scatter as pyg_scatter

from model.GNN import SurfaceEdgeEncoder
from model.message_passing_gnn import GraphNetBlock, GraphNetSurfaceBlock, MLP
from utils.data_loader import GraphData
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
import physicsnemo.models.gnn_layers.utils as physicsnemo_gnn_utils


class _TorchScatterCompat:
    @staticmethod
    def scatter(src, index, dim=0, dim_size=None, reduce="sum"):
        return pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)


if not hasattr(physicsnemo_gnn_utils, "torch_scatter"):
    physicsnemo_gnn_utils.torch_scatter = _TorchScatterCompat


class SharedHybridMeshGraphNetProcessor(nn.Module):
    def __init__(self, edge_block: nn.Module, node_block: nn.Module, num_steps: int):
        super().__init__()
        self.edge_block = edge_block
        self.node_block = node_block
        self.num_steps = int(num_steps)

    def forward(self, node_features, mesh_edge_features, world_edge_features, graph):
        for _ in range(self.num_steps):
            mesh_edge_features, world_edge_features, node_features = self.edge_block(
                mesh_edge_features,
                world_edge_features,
                node_features,
                graph,
            )
            mesh_edge_features, world_edge_features, node_features = self.node_block(
                mesh_edge_features,
                world_edge_features,
                node_features,
                graph,
            )
        return node_features


def build_knn_graph(pos: torch.Tensor, k: int, batch: torch.Tensor | None = None) -> torch.Tensor:
    try:
        return knn_graph(pos, k=k, batch=batch, loop=False)
    except ImportError:
        print("no knn")
        pass

    if batch is None:
        batch = pos.new_zeros(pos.size(0), dtype=torch.long)

    edge_parts = []
    for batch_id in torch.unique(batch):
        node_ids = torch.where(batch == batch_id)[0]
        if node_ids.numel() <= 1:
            continue

        k_eff = min(int(k), node_ids.numel() - 1)
        pos_b = pos[node_ids]
        dist = torch.cdist(pos_b, pos_b)
        dist.fill_diagonal_(float("inf"))
        nn_local = dist.topk(k=k_eff, largest=False, dim=1).indices

        target = node_ids.repeat_interleave(k_eff)
        source = node_ids[nn_local.reshape(-1)]
        edge_parts.append(torch.stack([source, target], dim=0))

    if not edge_parts:
        return pos.new_empty((2, 0), dtype=torch.long)
    return torch.cat(edge_parts, dim=1)


class MeshGraphNetNodeEncoder(nn.Module):
    def __init__(self, history_len: int, hidden_dim: int, layer_norm: bool = False):
        super().__init__()
        self.history_len = int(history_len)
        self.mlp = MLP([6 * self.history_len, hidden_dim, hidden_dim], layer_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use velocity + displacement history only: (N, 6, T) -> (N, 6*T).
        vu_hist = x[:, 3:, -self.history_len :].contiguous()
        return self.mlp(vu_hist.flatten(start_dim=1))


class MeshGraphNetEdgeEncoder(nn.Module):
    def __init__(self, hidden_dim: int, layer_norm: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = MLP([8, hidden_dim, hidden_dim], layer_norm)

    def forward(self, pos0: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0:
            return pos.new_zeros((0, self.hidden_dim))

        src, dst = edge_index[0], edge_index[1]

        r0 = pos0[dst] - pos0[src]
        d0 = torch.norm(r0, dim=-1, keepdim=True)
        r0_hat = r0 / (d0 + 1e-8)

        r = pos[dst] - pos[src]
        d = torch.norm(r, dim=-1, keepdim=True)
        r_hat = r / (d + 1e-8)

        return self.mlp(torch.cat([r0, d0, r, d], dim=-1))
    
class MeshGraphNetEdgeEncoderContact(nn.Module):
    def __init__(self, hidden_dim: int, threshold: float = 22.0, layer_norm: bool = False):
        super().__init__()
        self.threshold  = threshold
        self.hidden_dim = hidden_dim
        # r_hat(3) + d(1) + v_rel(3) + v_normal_mag(1) + v_tangential_mag(1) = 9
        self.mlp = MLP([4, hidden_dim, hidden_dim], layer_norm)

    def forward(self, pos: torch.Tensor,edge_surf_index: torch.Tensor):
        if edge_surf_index is None or edge_surf_index.numel() == 0:
            return pos.new_zeros((0, self.hidden_dim)), edge_surf_index

        src, dst = edge_surf_index[0], edge_surf_index[1]

        # Relative position 
        r = pos[src] - pos[dst]                          # (E, 3)
        d = torch.norm(r, dim=-1)                        # (E,)

        # Filter edges within threshold
        keep           = d < self.threshold             # (E,) bool mask
        edge_surf_index = edge_surf_index[:, keep]       # (2, E_keep)

        if edge_surf_index.numel() == 0:
            return pos.new_zeros((0, self.hidden_dim)), edge_surf_index

        src, dst = edge_surf_index[0], edge_surf_index[1]

        # Contact features 
        r              = r[keep]                         # (E_keep, 3)
        d              = d[keep].unsqueeze(-1)           # (E_keep, 1)

        # Concatenate
        edge_surf_feat = torch.cat([
            r,            # (E_keep, 3)
            d,                # (E_keep, 1)
        ], dim=-1)            # (E_keep, 9)

        return self.mlp(edge_surf_feat), edge_surf_index  # return filtered index too


class MeshGraphNetDirectTemp(nn.Module):
    def __init__(
        self,
        history_len: int = 5,
        hidden_dim: int = 64,
        n_gnn_layers: int = 10,
        shared_layers: bool = True,
        surface_k: int = 10,
        topo_layer_norm: bool = False,
        surface_layer_norm: bool = False,
        encoder_layer_norm: bool = True,
        out_dim: int = 6,
    ):
        super().__init__()
        self.history_len = int(history_len)
        self.n_gnn_layers = int(n_gnn_layers)
        self.surface_k = int(surface_k)

        self.node_encoder = MeshGraphNetNodeEncoder(self.history_len, hidden_dim, encoder_layer_norm)
        self.edge_encoder = MeshGraphNetEdgeEncoder(hidden_dim, encoder_layer_norm)

        if shared_layers:
            self.layers_topo = nn.ModuleList(
                [GraphNetBlock(hidden_dim, hidden_dim, hidden_dim, layer_norm=topo_layer_norm)]
            )
        else:
            self.layers_topo = nn.ModuleList(
                [
                    GraphNetBlock(hidden_dim, hidden_dim, hidden_dim, layer_norm=topo_layer_norm)
                    for _ in range(self.n_gnn_layers)
                ]
            )

        self.layers_surf = GraphNetSurfaceBlock(
            hidden_dim=hidden_dim,
            layer_norm=surface_layer_norm,
        )
        self.surface_edge_encoder = SurfaceEdgeEncoder(
            hidden_dim=hidden_dim,
            threshold=float("inf"),
            layer_norm=encoder_layer_norm,
        )
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, graph: GraphData) -> torch.Tensor:
        x_t = graph.x[:, :, -1]
        v_t = x_t[:, 3:6]

        h = self.node_encoder(graph.x)
        edge_feat = self.edge_encoder(graph.x_initial, graph.pos, graph.edge_index)

        n_layers = self.n_gnn_layers if len(self.layers_topo) == 1 else len(self.layers_topo)

        batch = getattr(graph, "batch", None)
        edge_surf_index = build_knn_graph(graph.pos, k=self.surface_k, batch=batch)
        edge_surf_feat, edge_surf_index = self.surface_edge_encoder(graph.pos, v_t, edge_surf_index)

        for k in range(n_layers):
            topo_layer = self.layers_topo[k % len(self.layers_topo)]
            h, edge_feat = topo_layer(h, graph.edge_index, edge_feat)

            h = self.layers_surf(h, edge_surf_index, edge_surf_feat)

        return self.node_decoder(h)


class MeshGraphNetDirect(nn.Module):
    def __init__(
        self,
        history_len: int = 5,
        hidden_dim: int = 64,
        n_gnn_layers: int = 10,
        shared_layers: bool = True,
        surface_k: int = 10,
        threshold: float = 20.0,
        out_dim: int = 6,
        layer_norm: bool = False,
        topo_layer_norm: bool | None = None,
        surface_layer_norm: bool | None = None,
        encoder_layer_norm: bool | None = None,
    ):
        super().__init__()
        self.history_len = int(history_len)
        self.surface_k = int(surface_k)
        self.n_gnn_layers = int(n_gnn_layers)
        self.shared_layers = bool(shared_layers)

        edge_layer_norm = layer_norm if encoder_layer_norm is None else encoder_layer_norm

        self.edge_encoder = MeshGraphNetEdgeEncoder(
            hidden_dim=hidden_dim,
            layer_norm=edge_layer_norm,
        )

        self.edge_encoder_contact = MeshGraphNetEdgeEncoderContact(
            hidden_dim=hidden_dim,
            layer_norm=edge_layer_norm,
            threshold=threshold
        )

        self.mesh_gnn = HybridMeshGraphNet(
            input_dim_nodes=out_dim * self.history_len,  # velocity + displacement history only
            output_dim=out_dim,
            input_dim_edges=hidden_dim,
            processor_size=1 if self.shared_layers else self.n_gnn_layers,
            hidden_dim_processor=hidden_dim,
            hidden_dim_node_encoder=hidden_dim,
            hidden_dim_edge_encoder=hidden_dim,
            hidden_dim_node_decoder=hidden_dim,
        )
        if self.shared_layers:
            edge_block, node_block = self.mesh_gnn.processor.processor_layers
            self.mesh_gnn.processor = SharedHybridMeshGraphNetProcessor(
                edge_block=edge_block,
                node_block=node_block,
                num_steps=self.n_gnn_layers,
            )

    def forward(self, graph: GraphData) -> torch.Tensor:
        # Reshape node features to (N, 6*T) for velocity + displacement history only
        node_feats = graph.x[:, 3:, -self.history_len :].contiguous().flatten(start_dim=1)

        # Topology edge features
        mesh_edge_feat = self.edge_encoder(graph.x_initial, graph.pos, graph.edge_index) 

        # Surface edge features
        # batch = getattr(graph, "batch", None)
        # edge_surf_index = build_knn_graph(graph.pos, k=self.surface_k, batch=batch)
        edge_surf_feat, edge_surf_index = self.edge_encoder_contact(graph.pos, graph.edge_surf_index)

        hybrid_graph = Data(
            edge_index=torch.cat([graph.edge_index, edge_surf_index], dim=1),
            num_nodes=graph.num_nodes,
        )

        return self.mesh_gnn(
            node_feats,  # (N, 6*T)
            mesh_edge_feat,
            edge_surf_feat,
            hybrid_graph,
        )
