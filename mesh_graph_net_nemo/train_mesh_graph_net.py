import argparse
import json
import math
import os
import random
from datetime import datetime
from glob import glob

import torch
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from mesh_graph_net_nemo.mesh_graph_net import MeshGraphNetDirect
from utils.data_loader import FEMDataset


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MeshGraphNet direct one-step baseline")
    parser.add_argument("--config", type=str, default="mesh_graph_net_nemo/mesh_graph_net.json", help="Path to config JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling and model init")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--save-every", type=int, default=None, help="Override save-every")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data dir")
    parser.add_argument("--file-glob", type=str, default=None, help="Override file glob")
    return parser.parse_args()


def load_raw_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_mesh_graph_net(raw_cfg: dict) -> MeshGraphNetDirect:
    model_cfg = raw_cfg.get("model", {})
    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    node_cfg = model_cfg.get("node_encoder", {})
    topo_cfg = model_cfg.get("gnn_topology", {})
    surf_cfg = model_cfg.get("gnn_surface", {})
    decoder_cfg = model_cfg.get("decoder", {})

    return MeshGraphNetDirect(
        history_len=int(node_cfg.get("history_len", 5)),
        hidden_dim=hidden_dim,
        n_gnn_layers=int(topo_cfg.get("n_gnn_layers", 10)),
        shared_layers=bool(topo_cfg.get("shared_layers", True)),
        surface_k=int(surf_cfg.get("k", 10)),
        threshold=float(surf_cfg.get("threshold", 20.0)),
        topo_layer_norm=bool(topo_cfg.get("layer_norm", False)),
        surface_layer_norm=bool(surf_cfg.get("layer_norm", False)),
        encoder_layer_norm=bool(node_cfg.get("layer_norm", True)),
        out_dim=int(decoder_cfg.get("out_dim", 6)),
    )


def split_files(npz_files, data_cfg, split_cfg):
    percent = float(data_cfg.get("percent", 100)) / 100.0
    k = min(max(1, math.ceil(len(npz_files) * percent)), len(npz_files))
    npz_files = npz_files[:k]

    train_num = max(1, int(float(split_cfg.get("train", 0.85)) * k))
    train_num = min(train_num, k)
    valid_num = int(float(split_cfg.get("valid", 0.15)) * k)
    train_files = npz_files[:train_num]
    valid_files = npz_files[train_num : train_num + valid_num]
    if not valid_files and train_files:
        valid_files = train_files[-1:]
    return train_files, valid_files


def main():
    args = parse_args()
    set_seed(args.seed)
    raw = load_raw_config(args.config)

    data_cfg = raw.get("data", {})
    split_cfg = raw.get("split", {"train": 0.85, "valid": 0.15, "test": 0.0})
    train_cfg = raw.get("training", {})
    model_cfg = raw.get("model", {})

    data_dir = args.data_dir or data_cfg.get("data_dir", "data")
    file_glob = args.file_glob or data_cfg.get("file_glob", "*.npz")
    epochs = args.epochs if args.epochs is not None else int(train_cfg.get("epochs", 100))
    batch_size = args.batch_size if args.batch_size is not None else int(train_cfg.get("batch_size", 8))
    learning_rate = (
        args.learning_rate if args.learning_rate is not None else float(train_cfg.get("learning_rate", 1e-4))
    )
    save_every = args.save_every if args.save_every is not None else int(train_cfg.get("save_every", 10))

    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(root)
    print(root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = os.path.join(root, data_dir)
    npz_files = sorted(glob(os.path.join(data_path, file_glob)))
    if not npz_files:
        raise FileNotFoundError(f"No files found in {data_path} with pattern {file_glob}")

    npz_files_train, npz_files_valid = split_files(npz_files, data_cfg, split_cfg)
    hist_len = int(model_cfg.get("node_encoder", {}).get("history_len", 5))
    pred_horizon = int(model_cfg.get("decoder", {}).get("pred_horizon", 1))
    geometry_path = os.path.join(data_path, "geometry_shared.npz")

    datasets_train = [
        FEMDataset(path, geometry_path=geometry_path, history_len=hist_len, predict_horizon=pred_horizon)
        for path in npz_files_train
    ]
    datasets_valid = [
        FEMDataset(path, geometry_path=geometry_path, history_len=hist_len, predict_horizon=pred_horizon)
        for path in npz_files_valid
    ]
    dataset_train = ConcatDataset(datasets_train)
    dataset_valid = ConcatDataset(datasets_valid)
    print(f"Total training samples: {len(dataset_train)}")
    print(f"Total validation samples: {len(dataset_valid)}")

    train_generator = torch.Generator().manual_seed(args.seed)
    valid_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, generator=train_generator)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, generator=valid_generator)

    model = create_mesh_graph_net(raw).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    train_loss_fn = torch.nn.MSELoss()
    val_loss_fn = torch.nn.L1Loss()

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params} (trainable: {num_trainable})")

    model_dir = os.path.join(root, "save_model")
    log_dir = os.path.join(root, "train_log")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log_mesh_graph_net_{timestamp}.txt")
    model_path = os.path.join(model_dir, f"mesh_graph_net_{timestamp}.pt")

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_graphs in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
            batch_graphs = batch_graphs.to(device)
            batch_graphs.delta_t = batch_graphs.delta_t[batch_graphs.batch]

            pred = model(batch_graphs)
            target = batch_graphs.y[:, 0, 3:]
            loss = train_loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_graphs in valid_loader:
                batch_graphs = batch_graphs.to(device)
                batch_graphs.delta_t = batch_graphs.delta_t[batch_graphs.batch]
                pred = model(batch_graphs)
                target = batch_graphs.y[:, 0, 3:]
                val_loss += torch.norm(pred - target, dim=1).mean().item()

        avg_val_loss = val_loss / max(len(valid_loader), 1)
        scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.6f} - val(one-step): {avg_val_loss:.6f}")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{avg_loss}\t{avg_val_loss}\n")

        if (epoch + 1) % save_every == 0 and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Saved model to {model_path} with val loss {best_val_loss:.6f}")

    print(f"Saved loss history to {log_path}")


if __name__ == "__main__":
    main()
