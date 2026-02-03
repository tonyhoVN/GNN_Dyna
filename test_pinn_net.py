import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.GNN import PhysicsNet
from utils.data_loader import FEMDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Test PhysicsNet on energy targets")
    parser.add_argument("--weights", type=str, default=r"model\physicsnet_20260129_021738.pt", help="Path to model weights (.pt)")
    parser.add_argument("--data", type=str, default=r"data\20260122_094307_gnn_data.npz", help="Path to a .npz file")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches to evaluate")
    return parser.parse_args()


def collate_graphs(batch):
    return batch


def main():
    args = parse_args()
    root = os.getcwd()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = args.data
    if not data_path:
        data_dir = os.path.join(root, "data")
        npz_files = sorted(
            os.path.join(data_dir, name)
            for name in os.listdir(data_dir)
            if name.lower().endswith(".npz")
        )
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")
        data_path = npz_files[0]

    dataset = FEMDataset(data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)

    model = PhysicsNet(hidden_dim=64).to(device)
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights: {args.weights}")

    model.eval()
    total_loss = 0.0
    total_mae_k = 0.0
    total_mae_u = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, graphs in enumerate(loader):
            if batch_idx >= args.num_batches:
                break
            for graph in graphs:
                graph = graph.to(device)
                if graph.total_kinetic_energy is None or graph.total_internal_energy is None:
                    raise ValueError("Dataset missing total energy targets.")

                k_pred, u_pred = model(graph)
                k_true = graph.total_kinetic_energy[0]
                u_true = graph.total_internal_energy[0]

                loss = model.loss(k_pred, u_pred, k_true, u_true)
                total_loss += loss.item()
                total_mae_k += torch.abs(k_pred - k_true).item()
                total_mae_u += torch.abs(u_pred - u_true).item()
                count += 1
                print(total_mae_u/u_true)

    if count == 0:
        raise RuntimeError("No graphs evaluated. Check dataset or num-batches.")

    print(f"Samples: {count}")
    print(f"Avg loss: {total_loss / count:.6f}")
    print(f"Avg MAE kinetic: {total_mae_k / count:.6f}")
    print(f"Avg MAE internal: {total_mae_u / count:.6f}")


if __name__ == "__main__":
    main()
