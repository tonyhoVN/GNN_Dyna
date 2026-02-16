import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from model.GNN import (
    TemporalEncoder,
    EncodeDecodeGNNGeneral,
    EncoderDecodeGNNForce,
    EdgeEncoder,
    GRUResidualDecoder,
)
from model.message_passing_gnn import (
    MLP,
    GraphNetBlock,
    GraphNetSurfaceBlock,
    GraphNetSurfaceBlockForce
)
from torch import nn


@dataclass
class ModelConfig:
    hidden_dim: int
    node_encoder: Dict[str, Any]
    edge_encoder: Dict[str, Any]
    gnn_topology: Dict[str, Any]
    gnn_surface: Dict[str, Any]
    head_topo: Dict[str, Any]
    head_surface: Dict[str, Any]
    decoder: Dict[str, Any]
    mlp: Dict[str, Any]

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
            head_topo=model_cfg.get("head_topo", {}),
            head_surface=model_cfg.get("head_surface", {}),
            decoder=model_cfg.get("decoder", {}),
            mlp=model_cfg.get("mlp", {}),
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
        layer_norm=bool(config.node_encoder.get("layer_norm", True))
    )

    # Edge encoder (material id embedding + numeric features)
    numeric_dim = max(edge_feat_dim - 1, 0)
    edge_encoder = EdgeEncoder(
        num_materials=num_materials,
        mat_emb_dim=mat_emb_dim,
        numeric_dim=numeric_dim,
        out_dim=hidden_dim,
        layer_norm=bool(config.edge_encoder.get("layer_norm", True))
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
    node_decoder = MLP(node_decoder_layers, 
                       layer_norm=bool(config.mlp.get("layer_norm", False)))

    return EncodeDecodeGNNGeneral(
        node_encoder,
        edge_encoder,
        layers_topo,
        layers_surface,
        node_decoder,
    )

def create_gnn_force_model(
    config: ModelConfig,
) -> EncoderDecodeGNNForce:
    hidden_dim = config.hidden_dim
    
    
    # Force model node encoder is a plain MLP on x_t = [u_t, v_t, x0]
    node_feat_dim = int(config.node_encoder.get("feat_dim", 9))
    node_encoder = MLP([node_feat_dim, hidden_dim, hidden_dim], 
                       layer_norm = bool(config.node_encoder.get("layer_norm", False)))

    # Edge encoder (material id embedding + numeric features)
    num_materials = int(config.edge_encoder.get("num_materials", 2))
    edge_feat_dim = int(config.edge_encoder.get("feat_dim", 2))
    mat_emb_dim = int(config.edge_encoder.get("mat_emb_dim", 4))
    numeric_dim = max(edge_feat_dim - 1, 0)

    edge_encoder = EdgeEncoder(
        num_materials=num_materials,
        mat_emb_dim=mat_emb_dim,
        numeric_dim=numeric_dim,
        out_dim=hidden_dim,
        layer_norm=bool(config.edge_encoder.get("layer_norm")),
    )

    # Internal force message passing
    n_topo_layers = int(config.gnn_topology.get("n_gnn_layers", 4))
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

    # Contact force branch
    surface_enabled = bool(config.gnn_surface.get("enabled", True))
    layers_surface: Optional[nn.Module]
    if surface_enabled:
        layers_surface = GraphNetSurfaceBlockForce(hidden_dim=hidden_dim)
    else:
        layers_surface = None

    head_topo_out = int(config.head_topo.get("out_dim", 3))
    head_surface_out = int(config.head_surface.get("out_dim", 3))
    head_topo = MLP([hidden_dim, hidden_dim, head_topo_out])
    head_surface = MLP([hidden_dim, hidden_dim, head_surface_out])

    # Residual decoder over time series x
    # Current EncoderDecodeGNNForce.update() calls node_decoder(x, dt) without y_base.
    decoder_hidden = int(config.decoder.get("hidden_dim", hidden_dim))
    decoder_layers = int(config.decoder.get("n_layers", 2))
    decoder_out_dim = int(config.decoder.get("out_dim", 6))
    decoder_in_dim = int(config.decoder.get("in_dim", 9))
    
    node_decoder = GRUResidualDecoder(
        in_dim=decoder_in_dim,
        hidden_dim=decoder_hidden,
        out_dim=decoder_out_dim,
        n_layers=decoder_layers,
    )

    return EncoderDecodeGNNForce(
        node_encoder=node_encoder,
        edge_encorder=edge_encoder,
        gnn_topo=layers_topo,
        head_topo=head_topo,
        gnn_surface=layers_surface,
        head_surface=head_surface,
        node_decoder=node_decoder,
    )
