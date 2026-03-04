import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from ansys.dpf import core as dpf

from model.model_creation import ModelConfig, create_gnn_model
from utils.data_loader import FEMDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Rollout animation for recurrent direct model.")
    parser.add_argument("--config", type=str, default="config/gnn_contact_direct_recurrent.json")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--npz-path", type=str, required=True)
    parser.add_argument("--geometry-path", type=str, default=None)
    parser.add_argument("--d3plot-dir", type=str, default=None)
    parser.add_argument("--start-index", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=100)
    parser.add_argument("--interval-ms", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _derive_d3plot_dir(npz_path, root_dir):
    base = os.path.basename(npz_path)
    suffix = "_time_data.npz"
    if not base.endswith(suffix):
        return None
    case_name = base[: -len(suffix)]
    candidate = os.path.join(root_dir, "output", case_name)
    return candidate if os.path.isdir(candidate) else None


def _expand_dt(graph):
    if hasattr(graph, "delta_t") and graph.delta_t is not None:
        dt = graph.delta_t
        if dt.dim() == 0:
            graph.delta_t = dt.repeat(graph.num_nodes)
        elif dt.dim() == 1 and dt.numel() == 1:
            graph.delta_t = dt.repeat(graph.num_nodes)
        elif hasattr(graph, "batch") and dt.dim() == 1 and dt.numel() != graph.num_nodes:
            graph.delta_t = dt[graph.batch]


def _build_rollout(model, dataset, start_index, rollout_steps, device):
    start_graph = dataset[start_index].to(device)
    x_hist = start_graph.x.clone()  # (N, F, T)
    x_initial = start_graph.x_initial

    gt_positions = []
    pred_positions = []

    with torch.no_grad():
        for step in range(rollout_steps):
            sample_idx = start_index + step
            gt_graph = dataset[sample_idx]
            gt_y = gt_graph.y
            if gt_y.dim() == 3:
                gt_y = gt_y[:, 0, :]
            gt_positions.append((x_initial.detach().cpu() + gt_y[:, 6:9]).numpy())

            graph_in = dataset[sample_idx].to(device)
            graph_in.x = x_hist
            graph_in.pos = x_hist[:, 6:9, -1] + x_initial
            _expand_dt(graph_in)

            pred_seq = model(graph_in)  # (N, horizon, 6) where [v,u]
            vu_next = pred_seq[:, 0, :]  # only first step for rollout

            pred_pos = (x_initial + vu_next[:, 3:6]).detach().cpu().numpy()
            pred_positions.append(pred_pos)

            x_last = x_hist[:, :, -1]
            next_state = x_last.clone()
            next_state[:, 3:] = vu_next
            x_hist = torch.cat([x_hist[:, :, 1:], next_state.unsqueeze(-1)], dim=2)

    return np.asarray(gt_positions), np.asarray(pred_positions), x_initial.detach().cpu().numpy()


def _animate_and_save(gt_positions, pred_positions, x_initial, base_grid, interval_ms, output_mp4_path):
    gt_grid = base_grid.copy(deep=True)
    pred_grid = base_grid.copy(deep=True)

    gt_disp0 = np.linalg.norm(gt_positions[0] - x_initial, axis=1)
    pr_disp0 = np.linalg.norm(pred_positions[0] - x_initial, axis=1)
    disp_all = np.concatenate([gt_disp0, pr_disp0])
    clim = [float(disp_all.min()), float(disp_all.max())]

    gt_grid.points = gt_positions[0]
    pred_grid.points = pred_positions[0]
    gt_grid["disp_mag"] = gt_disp0
    pred_grid["disp_mag"] = pr_disp0

    plotter = pv.Plotter(shape=(1, 2))
    plotter.subplot(0, 0)
    plotter.add_text("Ground Truth", font_size=12)
    plotter.add_mesh(gt_grid, scalars="disp_mag", cmap="turbo", clim=clim, show_edges=True, copy_mesh=False)
    plotter.subplot(0, 1)
    plotter.add_text("Prediction", font_size=12)
    plotter.add_mesh(pred_grid, scalars="disp_mag", cmap="turbo", clim=clim, show_edges=True, copy_mesh=False)
    plotter.link_views()

    n_frames = gt_positions.shape[0]
    plotter.open_movie(output_mp4_path, framerate=max(1, int(round(1000.0 / interval_ms))))
    plotter.show(auto_close=False, interactive_update=True)

    for f in range(n_frames):
        gt_p = gt_positions[f]
        pr_p = pred_positions[f]
        gt_disp = np.linalg.norm(gt_p - x_initial, axis=1)
        pr_disp = np.linalg.norm(pr_p - x_initial, axis=1)
        try:
            plotter.update_coordinates(gt_p, mesh=gt_grid, render=False)
            plotter.update_coordinates(pr_p, mesh=pred_grid, render=False)
            plotter.update_scalars(gt_disp, mesh=gt_grid, render=False)
            plotter.update_scalars(pr_disp, mesh=pred_grid, render=False)
        except Exception:
            gt_grid.points = gt_p
            pred_grid.points = pr_p
            gt_grid["disp_mag"] = gt_disp
            pred_grid["disp_mag"] = pr_disp
            gt_grid.Modified()
            pred_grid.Modified()
        plotter.render()
        plotter.write_frame()
        plotter.update()
        time.sleep(max(0.001, interval_ms / 1000.0))

    plotter.close()
    print(f"Saved recurrent rollout animation to: {output_mp4_path}")


def _save_mae_plot(gt_positions, pred_positions, output_png_path):
    mae_per_step = np.mean(np.linalg.norm(pred_positions - gt_positions, axis=2), axis=1)
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(mae_per_step)), mae_per_step, marker="o")
    plt.xlabel("Rollout step")
    plt.ylabel("MAE (position norm)")
    plt.title("Recurrent Rollout Error Over Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=150)
    plt.close()
    print(f"Saved recurrent rollout MAE plot to: {output_png_path}")


def main():
    args = parse_args()
    device = torch.device(args.device)

    dpf.start_local_server(ansys_path=r"C:\Program Files\ANSYS Inc\v242", as_global=True)

    dataset = FEMDataset(args.npz_path, geometry_path=args.geometry_path)
    if not (0 <= args.start_index < len(dataset)):
        raise IndexError(f"start-index {args.start_index} out of range for dataset size {len(dataset)}")

    model_cfg = ModelConfig.from_json(args.config)
    model = create_gnn_model(model_cfg).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    max_steps = len(dataset) - args.start_index
    rollout_steps = min(args.rollout_steps, max_steps)
    if rollout_steps <= 0:
        raise ValueError("rollout-steps is too large for selected start-index.")

    gt_positions, pred_positions, x_initial = _build_rollout(
        model, dataset, args.start_index, rollout_steps, device
    )

    root = os.getcwd()
    d3plot_dir = args.d3plot_dir or _derive_d3plot_dir(args.npz_path, root)
    if not d3plot_dir:
        raise FileNotFoundError("Could not derive d3plot directory. Please provide --d3plot-dir.")

    d3plot_path = os.path.join(d3plot_dir, "d3plot")
    ds = dpf.DataSources()
    ds.set_result_file_path(d3plot_path, "d3plot")
    dpf_model = dpf.Model(ds)
    base_grid = dpf_model.metadata.meshed_region.grid

    results_dir = os.path.join(root, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_mp4_path = os.path.join(results_dir, "rollout_prediction_recurrent.mp4")
    output_png_path = os.path.join(results_dir, "rollout_prediction_recurrent_mae.png")

    _animate_and_save(gt_positions, pred_positions, x_initial, base_grid, args.interval_ms, output_mp4_path)
    _save_mae_plot(gt_positions, pred_positions, output_png_path)


if __name__ == "__main__":
    main()
