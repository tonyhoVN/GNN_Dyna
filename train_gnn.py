import os
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, random_split
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.data_loader import FEMDataset
from model.GNN import EncodeDecodeGNN

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
    if name.lower().endswith("data.npz")
)
if not npz_files:
    raise FileNotFoundError(f"No data.npz files found in {data_dir}")

datasets = [FEMDataset(path) for path in npz_files]
dataset = ConcatDataset(datasets)
total_samples = len(dataset)
print(f"Total samples: {total_samples}")

# Split to train and test
train_size = int(0.7 * total_samples)
valid_size = int(0.2 * total_samples)
test_size = total_samples - train_size - valid_size
train_dataset, valid_dataset, test_dataset = random_split(
    dataset,
    [train_size, valid_size, test_size],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(
    train_dataset, batch_size=5, shuffle=True
)

valid_loader = DataLoader(
    valid_dataset, batch_size=5, shuffle=False
)

# Initialize model and optimizer
ref_dataset = dataset[0]
node_dim = ref_dataset.x.shape[1]
out_dim = ref_dataset.y.shape[1]
edge_dim = ref_dataset.edge_attr.shape[1]
model = EncodeDecodeGNN(node_dim=node_dim, 
                        edge_dim=edge_dim, 
                        out_dim=out_dim,
                        latent_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Print model summary
num_params = sum(p.numel() for p in model.parameters())
num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {num_params} (trainable: {num_trainable})")

# Save parameters and setup logging
epochs = args.epochs
save_every = args.save_every
loss_history = []
model_dir = os.path.join(root, "save_model")
log_dir = os.path.join(root, "train")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"log_{timestamp}.txt")
model_path = os.path.join(model_dir, f"meshgraphnet_{timestamp}.pt")


def train():
    # Training loop
    model.train()
    for epoch in range(epochs):
        # Compute loss for batch
        total_loss = 0.0
        for batch_graphs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            batch_graphs = batch_graphs.to(device)
            batch_graphs.delta_t = batch_graphs.delta_t[batch_graphs.batch]

            y_predict = model(batch_graphs)
            batch_loss = model.loss(y_predict, batch_graphs.y)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        # Compute average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_graphs in valid_loader:
                batch_graphs = batch_graphs.to(device)
                batch_graphs.delta_t = batch_graphs.delta_t[batch_graphs.batch]
                y_predict = model(batch_graphs)
                batch_loss = model.loss(y_predict, batch_graphs.y)
                val_loss += batch_loss.item()

        avg_val_loss = val_loss / len(valid_loader)
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
    train()
    # plot_loss()