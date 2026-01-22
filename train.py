import os
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.data_loader import FEMDataset
from model.GNN import MeshGraphNet

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
print(f"Total samples: {len(dataset)}")

loader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=lambda batch: batch)

# Initialize model and optimizer
ref_dataset = datasets[0]
node_dim = ref_dataset.X_list.shape[2]
out_dim = ref_dataset.Y_list.shape[2]
edge_dim = ref_dataset.edge_attr.shape[1] if ref_dataset.edge_attr is not None else 1
model = MeshGraphNet(node_dim=node_dim, edge_dim=edge_dim, out_dim=out_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# Training loop
epochs = 100
save_every = 10
loss_history = []
model_dir = os.path.join(root, "model")
log_dir = os.path.join(root, "train")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"log_{timestamp}.txt")
model_path = os.path.join(model_dir, f"meshgraphnet_{timestamp}.pt")


model.train()
for epoch in range(epochs):
    total_loss = 0.0
    num_graphs = 0
    for graphs in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
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
    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.6f}")

    # Log loss to file
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{avg_loss}\n")

    # Save model checkpoint
    if (epoch + 1) % save_every == 0:
        # save_path = os.path.join(model_dir, f"meshgraphnet_{timestamp}_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

print(f"Saved loss history to {log_path}")

# Load training loss and plot
with open(log_path, "r", encoding="utf-8") as f:
    loaded_loss = [float(line.strip()) for line in f if line.strip()]

plt.plot(loaded_loss, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Avg Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()
