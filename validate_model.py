import argparse
import json
import os
import random
import time
from glob import glob

import torch
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader

from model.model_creation import ModelConfig, create_gnn_model
from utils.data_loader import FEMDataset


seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate recurrent GNN with one-step and rollout losses")
    parser.add_argument(
        "--config",
        type=str,
        default="config/gnn_contact_direct_recurrent_infer.json",
        help="Path to config JSON",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Override data dir")
    parser.add_argument("--file-glob", type=str, default=None, help="Override file glob")
    parser.add_argument("--pt-file", type=str, default=None, help="Path to model .pt file for validation")
    parser.add_argument("--start-index", type=int, default=20, help="Start index for rollout in each series")
    parser.add_argument("--rollout-steps", type=int, default=150, help="Number of rollout steps per series")
    return parser.parse_args()


def load_raw_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def expand_dt(graph):
    if hasattr(graph, "delta_t") and graph.delta_t is not None:
        dt = graph.delta_t
        if dt.dim() == 0:
            graph.delta_t = dt.repeat(graph.num_nodes)
        elif dt.dim() == 1 and dt.numel() == 1:
            graph.delta_t = dt.repeat(graph.num_nodes)
        elif hasattr(graph, "batch") and dt.dim() == 1 and dt.numel() != graph.num_nodes:
            graph.delta_t = dt[graph.batch]


def set_rollout_context(graph, x_hist):
    # Match benchmark.py: current node positions come from the latest state in the history.
    graph.x = x_hist
    graph.pos = x_hist[:, 6:9, -1] + graph.x_initial
    expand_dt(graph)


def predict_next_state(model, graph, x_hist):
    set_rollout_context(graph, x_hist)
    x_last = x_hist[:, :, -1]

    pred = model(graph)
    if pred.dim() == 3:
        vu_next = pred[:, 0, :]
    else:
        vu_next = pred

    next_state = x_last.clone()
    next_state[:, 3:] = vu_next
    return next_state


def main():
    args = parse_args()
    raw = load_raw_config(args.config)

    data_cfg = raw.get("data", {})
    data_dir = args.data_dir or data_cfg.get("data_dir", "data")
    file_glob = args.file_glob or data_cfg.get("file_glob", "*.npz")

    model_cfg = ModelConfig.from_json(args.config)
    root = os.getcwd()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = os.path.join(root, data_dir)
    npz_files = sorted(glob(os.path.join(data_path, file_glob)))
    if not npz_files:
        raise FileNotFoundError(f"No files found in {data_path} with pattern {file_glob}")
    npz_files = npz_files[-10:]
    hist_len = int(model_cfg.node_encoder.get("history_len", 5))
    pred_horizon = int(model_cfg.decoder.get("pred_horizon", 1))
    datasets = [
        # Let each NPZ resolve its own geometry_path when present.
        FEMDataset(path, history_len=hist_len, predict_horizon=pred_horizon)
        for path in npz_files
    ]
    dataset = ConcatDataset(datasets)
    total_samples = len(dataset)
    print(f"Total samples: {total_samples}")

    valid_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = create_gnn_model(model_cfg).to(device)
    one_step_mse = torch.nn.MSELoss()

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params} (trainable: {num_trainable})")

    if args.pt_file is not None:
        model_path = os.path.join(root, args.pt_file)
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    one_step_loss_sum = 0.0
    valid_items = 0

    with torch.no_grad():
        for batch_graphs in valid_loader:
            batch_graphs = batch_graphs.to(device)
            batch_graphs.delta_t = batch_graphs.delta_t[batch_graphs.batch]
            pred_seq = model(batch_graphs)
            y_target = batch_graphs.y
            if y_target.dim() == 3:
                y_target = y_target[:, 0, :]

            one_step_pred = pred_seq[:, 0, :] if pred_seq.dim() == 3 else pred_seq
            one_step_loss_sum += one_step_mse(one_step_pred, y_target[:, 3:]).item()
            valid_items += 1

    # breakpoint()
    rollout_series_losses = []
    for series_idx, series_dataset in enumerate(datasets):
        max_steps = len(series_dataset) - args.start_index
        rollout_steps = min(args.rollout_steps, max_steps)
        if rollout_steps <= 0:
            print(f"Skip series {series_idx}: insufficient length for start index {args.start_index}")
            continue

        start_graph = series_dataset[args.start_index].to(device)
        x_hist = start_graph.x.clone()
        x_initial = start_graph.x_initial
        series_rollout_err = 0.0

        with torch.no_grad():
            for step in range(rollout_steps):
                idx = args.start_index + step
                gt_graph = series_dataset[idx]
                gt_y = gt_graph.y.to(device)
                if gt_y.dim() == 3:
                    gt_y = gt_y[:, 0, :]

                graph_in = gt_graph.to(device)
                next_state = predict_next_state(model, graph_in, x_hist)
                # breakpoint()
                gt_pos = x_initial + gt_y[:, 6:9]
                pred_pos = x_initial + next_state[:, 6:9]
                step_err = torch.norm(pred_pos - gt_pos, dim=1).mean()
                series_rollout_err += step_err.item()

                x_hist = torch.cat([x_hist[:, :, 1:], next_state.unsqueeze(-1)], dim=2)

        rollout_series_losses.append(series_rollout_err / max(1, rollout_steps))
        # print(
        #     f"Series {series_idx}: rollout position MAE = {rollout_series_losses[-1]:.6f} "
        #     f"over {rollout_steps} steps"
        # )

    avg_one_step_loss = one_step_loss_sum / max(valid_items, 1)
    avg_rollout_loss = sum(rollout_series_losses) / max(len(rollout_series_losses), 1)

    print(f"Val-loss(one-step): {avg_one_step_loss:.6f}")
    print(f"Val-loss(rollout): {avg_rollout_loss:.6f}")


if __name__ == "__main__":
    main()
