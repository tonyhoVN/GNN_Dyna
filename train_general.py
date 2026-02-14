import argparse
import json
import os
from datetime import datetime
from glob import glob

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from utils.data_loader import FEMDataset
from model.model_creation import ModelConfig, create_gnn_model
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN from JSON config")
    parser.add_argument("--config", type=str, default="config/contact_gnn.json", help="Path to config JSON")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Override hidden dim")
    parser.add_argument("--save-every", type=int, default=None, help="Override save-every")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data dir")
    parser.add_argument("--file-glob", type=str, default=None, help="Override file glob")
    return parser.parse_args()


def load_raw_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    raw = load_raw_config(args.config)

    data_cfg = raw.get("data", {})
    split_cfg = raw.get("split", {"train": 0.7, "valid": 0.2, "test": 0.1})
    train_cfg = raw.get("training", {})

    data_dir = args.data_dir or data_cfg.get("data_dir", "data")
    file_glob = args.file_glob or data_cfg.get("file_glob", "*.npz")

    epochs = args.epochs if args.epochs is not None else int(train_cfg.get("epochs", 100))
    batch_size = args.batch_size if args.batch_size is not None else int(train_cfg.get("batch_size", 8))
    learning_rate = (
        args.learning_rate if args.learning_rate is not None else float(train_cfg.get("learning_rate", 1e-4))
    )
    save_every = args.save_every if args.save_every is not None else int(train_cfg.get("save_every", 10))

    model_cfg = ModelConfig.from_json(args.config)
    if args.hidden_dim is not None:
        model_cfg.hidden_dim = args.hidden_dim

    root = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = wandb.init(
        entity="tonyho-stony-brook-university",
        project="Physics-informed Graph neural net",
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "hidden_dim": model_cfg.hidden_dim,
            "architecture": "EncodeDecodeGNNGeneral",
            "dataset": "LSDyna",
            "epochs": epochs,
            "timestamp": timestamp,
            "config_path": args.config,
        },
    )

    data_path = os.path.join(root, data_dir)
    npz_files = sorted(glob(os.path.join(data_path, file_glob)))
    if not npz_files:
        raise FileNotFoundError(f"No files found in {data_path} with pattern {file_glob}")

    datasets = [FEMDataset(path) for path in npz_files]
    dataset = ConcatDataset(datasets)
    total_samples = len(dataset)
    print(f"Total samples: {total_samples}")

    train_size = int(split_cfg.get("train", 0.7) * total_samples)
    valid_size = int(split_cfg.get("valid", 0.2) * total_samples)
    test_size = total_samples - train_size - valid_size
    train_dataset, valid_dataset, _ = random_split(
        dataset,
        [train_size, valid_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    ref_graph = dataset[0]
    node_dim = ref_graph.x.shape[1]
    out_dim = ref_graph.y.shape[1]
    edge_dim = ref_graph.edge_attr.shape[1] if ref_graph.edge_attr is not None else 1

    model = create_gnn_model(model_cfg, node_dim, edge_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params} (trainable: {num_trainable})")

    model_dir = os.path.join(root, "save_model")
    log_dir = os.path.join(root, "train_log")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log_{timestamp}.txt")
    model_path = os.path.join(model_dir, f"gnn_general_{timestamp}.pt")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_graphs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            batch_graphs = batch_graphs.to(device)
            # if batch_graphs.delta_t is not None and batch_graphs.delta_t.numel() == batch_graphs.num_graphs:
            batch_graphs.delta_t = batch_graphs.delta_t + 0.002 * torch.randn_like(batch_graphs.delta_t)
            batch_graphs.delta_t = batch_graphs.delta_t[batch_graphs.batch]

            y_predict = model(batch_graphs)
            batch_loss = model.loss(y_predict, batch_graphs.y)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_graphs in valid_loader:
                batch_graphs = batch_graphs.to(device)
                # if batch_graphs.delta_t is not None and batch_graphs.delta_t.numel() == batch_graphs.num_graphs:
                batch_graphs.delta_t = batch_graphs.delta_t[batch_graphs.batch]
                y_predict = model(batch_graphs)
                val_loss += model.loss(y_predict, batch_graphs.y).item()

        avg_val_loss = val_loss / max(len(valid_loader), 1)
        model.train()
        
        # Log to wandb
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.6f} - val: {avg_val_loss:.6f}")
        run.log({"train_loss": avg_loss, "val_loss": avg_val_loss})
        
        # Log loss to file
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{avg_loss}\n")

        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), model_path)
            print(f"Saved model to {model_path}")

    print(f"Saved loss history to {log_path}")
    run.finish()


if __name__ == "__main__":
    main()
