import argparse
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, random_split
from tqdm import tqdm

from model.GNN import PhysicsNet
from utils.data_loader import FEMDataset

import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Train PhysicsNet for energy prediction")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save model checkpoint every N epochs",
    )
    return parser.parse_args()


def collate_graphs(batch):
    return batch


def main():
    # Parse arguments
    args = parse_args()
    root = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = os.path.join(root, "data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup wandb
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="tonyho-stony-brook-university",
        # Set the wandb project where this run will be logged.
        project="Physics-informed Graph neural net",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": 1e-4,
            "architecture": "PhysicsNet",
            "dataset": "LSDyna",
            "epochs": args.epochs,
        },
    )

    # Load dataset
    npz_files = sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.lower().endswith(".npz")
    )
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    datasets = [FEMDataset(path) for path in npz_files]
    dataset = ConcatDataset(datasets)
    total_samples = len(dataset)
    print(f"Total samples: {total_samples}")

    train_size = int(0.85 * total_samples)
    valid_size = total_samples - train_size
    train_dataset, valid_dataset = random_split(
        dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=collate_graphs
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=4, shuffle=False, collate_fn=collate_graphs
    )

    # Create model and optimizer
    model = PhysicsNet(hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Print model summary
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params} (trainable: {num_trainable})")

    # Setup logging and model saving paths
    model_dir = os.path.join(root, "model")
    log_dir = os.path.join(root, "train")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"pinn_log_{timestamp}.txt")
    model_path = os.path.join(model_dir, f"physicsnet_{timestamp}.pt")

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        # Training 
        total_loss = 0.0
        num_graphs = 0
        for graphs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch"):
            optimizer.zero_grad()
            batch_loss = 0.0
            for graph in graphs:
                graph = graph.to(device)
                if graph.total_kinetic_energy is None or graph.total_internal_energy is None:
                    raise ValueError("Dataset missing total energy targets.")

                k_pred, u_pred = model(graph)
                k_true = graph.total_kinetic_energy[0]
                u_true = graph.total_internal_energy[0]

                loss = model.loss(k_pred, u_pred, k_true, u_true)
                batch_loss = batch_loss + loss
                num_graphs += 1

            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        avg_loss = total_loss / max(num_graphs, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_graphs = 0
        with torch.no_grad():
            for graphs in valid_loader:
                for graph in graphs:
                    graph = graph.to(device)
                    if graph.total_kinetic_energy is None or graph.total_internal_energy is None:
                        raise ValueError("Dataset missing total energy targets.")
                    k_pred, u_pred = model(graph)
                    loss = model.loss(
                        k_pred,
                        u_pred,
                        graph.total_kinetic_energy,
                        graph.total_internal_energy,
                    )
                    val_loss += loss.item()
                    val_graphs += 1
        avg_val = val_loss / max(val_graphs, 1)
        model.train()

        # Print and log epoch results
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {avg_loss:.6f} - val: {avg_val:.6f}")
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{avg_loss}\t{avg_val}\n")

        run.log({"train_loss": avg_loss, "val_loss": avg_val})

        # Save model checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), model_path)
            print(f"Saved model to {model_path}")

    run.finish()
    print(f"Saved loss history to {log_path}")


if __name__ == "__main__":
    main()
