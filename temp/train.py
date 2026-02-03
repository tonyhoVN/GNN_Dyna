import os
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.data_loader import FEMDataset
from model.GNN import MeshGraphNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train MeshGraphNet")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save model checkpoint every N epochs",
    )
    return parser.parse_args()

args = parse_args()

# Load dataset
root = os.getcwd()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data_dir = os.path.join(root, "data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

train_loader = DataLoader(
    train_dataset, batch_size=5, shuffle=True, collate_fn=lambda batch: batch
)
valid_loader = DataLoader(
    valid_dataset, batch_size=5, shuffle=False, collate_fn=lambda batch: batch
)

# Initialize model and optimizer
ref_dataset = datasets[0]
node_dim = ref_dataset.X_list.shape[2]
out_dim = ref_dataset.Y_list.shape[2]
edge_dim = ref_dataset.edge_attr.shape[1] if ref_dataset.edge_attr is not None else 1
model = MeshGraphNet(node_dim=node_dim, edge_dim=edge_dim, out_dim=out_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Print model summary
num_params = sum(p.numel() for p in model.parameters())
num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {num_params} (trainable: {num_trainable})")

# Save parameters and setup logging
epochs = args.epochs
save_every = args.save_every
loss_history = []
model_dir = os.path.join(root, "model")
log_dir = os.path.join(root, "train")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"log_{timestamp}.txt")
model_path = os.path.join(model_dir, f"meshgraphnet_{timestamp}.pt")


def train():
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        num_graphs = 0
        for graphs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            optimizer.zero_grad()
            batch_loss = 0.0
            for graph in graphs:
                graph = graph.to(device)
                pred = model(graph)
                loss = F.mse_loss(pred, graph.y)
                batch_loss = batch_loss + loss
                num_graphs += 1

            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        # Compute average loss for the epoch
        avg_loss = total_loss / max(num_graphs, 1)
        loss_history.append(avg_loss)
        # Validation loss
        model.eval()
        val_loss = 0.0
        val_graphs = 0
        with torch.no_grad():
            for graphs in valid_loader:
                for graph in graphs:
                    graph = graph.to(device)
                    pred = model(graph)
                    val_loss += F.mse_loss(pred, graph.y).item()
                    val_graphs += 1
        avg_val_loss = val_loss / max(val_graphs, 1)
        model.train()

        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.6f} - val: {avg_val_loss:.6f}")

        # Log loss to file
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{avg_loss}\n")

        # Save model checkpoint
        if (epoch + 1) % save_every == 0:
            # save_path = os.path.join(model_dir, f"meshgraphnet_{timestamp}_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Saved model to {model_path}")

    print(f"Saved loss history to {log_path}")

def plot_loss():
    # Load training loss and plot
    with open(log_path, "r", encoding="utf-8") as f:
        loaded_loss = [float(line.strip()) for line in f if line.strip()]

    plt.plot(loaded_loss, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    pass
    # train()
    # plot_loss()