import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from model.GNN import TemporalEncoder, EncodeDecodeGNNGeneral, EdgeEncoder
from model.message_passing_gnn import MLP, GraphNetBlock, GraphNetSurfaceBlock
from torch import nn


@dataclass
class ModelConfig:
    hidden_dim: int
    node_encoder: Dict[str, Any]
    edge_encoder: Dict[str, Any]
    gnn_topology: Dict[str, Any]
    gnn_surface: Dict[str, Any]

    @staticmethod
    def from_json(path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        model_cfg = raw.get("model", raw)
        return ModelConfig(
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            node_encoder=model_cfg.get("node_encoder", {}),
            edge_encoder=model_cfg.get("edge_encoder", {}),
            gnn_topology=model_cfg.get("gnn_topology", {}),
            gnn_surface=model_cfg.get("gnn_surface", {}),
        )


def create_gnn_model(
    config: ModelConfig,
    node_feat_dim: int,
    edge_feat_dim: int,
    out_dim: int,
) -> EncodeDecodeGNNGeneral:
    hidden_dim = config.hidden_dim
    lstm_layers = int(config.node_encoder.get("lstm_layers", 3))
    use_mass = bool(config.node_encoder.get("use_mass", True))
    use_pos = bool(config.node_encoder.get("use_pos", True))
    num_materials = int(config.edge_encoder.get("num_materials", 2))
    mat_emb_dim = int(config.edge_encoder.get("mat_emb_dim", 4))

    # Node encoder
    node_encoder = TemporalEncoder(
        in_dim=node_feat_dim,
        hidden_dim=hidden_dim,
        n_layers=lstm_layers,
        use_mass=use_mass,
        use_pos=use_pos,
    )

    # Edge encoder (material id embedding + numeric features)
    numeric_dim = max(edge_feat_dim - 1, 0)
    edge_encoder = EdgeEncoder(
        num_materials=num_materials,
        mat_emb_dim=mat_emb_dim,
        numeric_dim=numeric_dim,
        out_dim=hidden_dim,
    )

    # Topo message-passing layers
    n_topo_layers = int(config.gnn_topology.get("n_gnn_layers", 5))
    layers_topo = nn.ModuleList(
        [
            GraphNetBlock(
                edge_feat_dim=hidden_dim,
                node_feat_dim=hidden_dim,
                hidden_dim=hidden_dim,
            )
            for _ in range(n_topo_layers)
        ]
    )

    # Surface message-passing layer (single block)
    surface_enabled = bool(config.gnn_surface.get("enabled", True))
    layers_surface: Optional[nn.Module]
    if surface_enabled:
        layers_surface = GraphNetSurfaceBlock(hidden_dim=hidden_dim)
    else:
        layers_surface = None

    # Node decoder
    node_decoder_layers = [hidden_dim, hidden_dim, out_dim]
    node_decoder = MLP(node_decoder_layers)

    return EncodeDecodeGNNGeneral(
        node_encoder,
        edge_encoder,
        layers_topo,
        layers_surface,
        node_decoder,
    )
