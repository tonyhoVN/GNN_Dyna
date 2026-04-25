import argparse
import json
import os
import random
from datetime import datetime
from glob import glob

import torch
from torch.utils.data import ConcatDataset, random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from model.model_creation import ModelConfig, create_gnn_model
from utils.data_loader import FEMDataset


seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train recurrent GNN from JSON config")
    parser.add_argument("--config", type=str, default="config/gnn_contact_direct_recurrent.json", help="Path to config JSON")
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


def weighted_sequence_mse(pred_seq, target_seq, percent):
    # pred_seq, target_seq: (N, H, C)
    horizon = pred_seq.shape[1]
    
    # Linear decay weights
    # weights = torch.arange(horizon, 0, -1, device=pred_seq.device, dtype=pred_seq.dtype)

    # Eponetial decay weights
    t = torch.arange(horizon, device=pred_seq.device, dtype=pred_seq.dtype)
    
    # tau = 0.4*horizon 
    # weights = torch.exp(-t / tau)  # Exponential decay

    alpha_start = 20
    alpha_end = 2
    if percent < 0.8:
        alpha = alpha_start - (alpha_start - alpha_end) * (percent / 0.8)
    else:
        alpha = alpha_end
    weights = torch.exp(-alpha * (t / horizon))
    weights = weights / weights.sum()
    
    per_h = ((pred_seq - target_seq) ** 2).mean(dim=(0, 2))  # (H,)
    return (per_h * weights).sum()


def main():
    args = parse_args()
    raw = load_raw_config(args.config)

    # Extract config values with command-line overrides
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

    # Load model config for model creation
    model_cfg = ModelConfig.from_json(args.config)
    root = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##### Load dataset #####
    data_path = os.path.join(root, data_dir)
    npz_files = sorted(glob(os.path.join(data_path, file_glob)))
    if not npz_files:
        raise FileNotFoundError(f"No files found in {data_path} with pattern {file_glob}")

    # Subsample files
    percent = float(data_cfg.get("percent", 100))
    k = max(1, int(len(npz_files) * (percent / 100.0)))
    # npz_files = random.sample(npz_files, k)
    npz_files = npz_files[:k]

    # Create dataset and dataloaders
    hist_len = int(model_cfg.node_encoder["history_len"])
    pred_horizon = int(model_cfg.decoder["pred_horizon"])
    geometry_path = os.path.join(data_path, "geometry_shared.npz")
    datasets = [FEMDataset(path, geometry_path=geometry_path, history_len=hist_len, predict_horizon=pred_horizon) for path in npz_files]
    dataset = ConcatDataset(datasets)
    total_samples = len(dataset)
    print(f"Total samples: {total_samples}")

    # Split dataset
    train_size = int(split_cfg.get("train", 0.7) * total_samples)
    valid_size = int(split_cfg.get("valid", 0.2) * total_samples)
    test_size = total_samples - train_size - valid_size
    train_dataset, valid_dataset, _ = random_split(
        dataset,
        [train_size, valid_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    ##### Create model, optimizer, scheduler #####
    model = create_gnn_model(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    one_step_mse = torch.nn.MSELoss()

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params} (trainable: {num_trainable})")

    model_dir = os.path.join(root, "save_model")
    log_dir = os.path.join(root, "train_log")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log_recurrent_{timestamp}.txt")
    model_path = os.path.join(model_dir, f"gnn_recurrent_{timestamp}.pt")

    best_val_loss = float("inf")
    
    ##### Training loop #####
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_graphs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            batch_graphs = batch_graphs.to(device)
            batch_graphs.delta_t = batch_graphs.delta_t[batch_graphs.batch]

            # Add noise to input features during training for regularization
            noise = 0.01 * torch.randn_like(batch_graphs.x)
            batch_graphs.x = batch_graphs.x + noise

            pred_seq = model(batch_graphs)  # (N_total, H, 6)
            y_target = batch_graphs.y[:, :, 3:]  # (N, H, 6)

            percent = epoch / epochs
            batch_loss = weighted_sequence_mse(pred_seq, y_target, percent=percent)

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += batch_loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)

        # Validation: one-step only (first predicted step)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_graphs in valid_loader:
                batch_graphs = batch_graphs.to(device)
                batch_graphs.delta_t = batch_graphs.delta_t[batch_graphs.batch]
                pred_seq = model(batch_graphs)
                y_target = batch_graphs.y
                if y_target.dim() == 3:
                    y_target = y_target[:, 0, :]
                val_loss += one_step_mse(pred_seq[:, 0, :], y_target[:, 3:]).item()

        avg_val_loss = val_loss / max(len(valid_loader), 1)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.6f} - val(one-step): {avg_val_loss:.6f}")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{avg_loss}\t{avg_val_loss}\n")

        if (epoch + 1) % save_every == 0 and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Saved model to {model_path} with val loss {best_val_loss:.6f}")

    print(f"Saved loss history to {log_path}")


if __name__ == "__main__":
    main()
