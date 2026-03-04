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
    parser = argparse.ArgumentParser(description="Animate rollout prediction on deformed mesh (DPF-style).")
    parser.add_argument("--config", type=str, default="config/gnn_contact_general.json", help="Path to model config JSON.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model checkpoint (.pt).")
    parser.add_argument("--npz-path", type=str, required=True, help="Path to one time-data npz file.")
    parser.add_argument("--geometry-path", type=str, default=None, help="Optional explicit geometry npz path.")
    parser.add_argument("--d3plot-dir", type=str, default=None, help="Directory containing d3plot file (optional).")
    parser.add_argument("--start-index", type=int, default=10, help="Start sample index for rollout.")
    parser.add_argument("--rollout-steps", type=int, default=100, help="Number of rollout steps.")
    parser.add_argument("--interval-ms", type=int, default=50, help="Animation frame interval in milliseconds.")
    parser.add_argument("--warp-scale", type=float, default=1.0, help="Scale factor for visual displacement warp.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    return parser.parse_args()


def _derive_d3plot_dir(npz_path, root_dir):
    base = os.path.basename(npz_path)
    suffix = "_time_data.npz"
    if not base.endswith(suffix):
        return None
    case_name = base[: -len(suffix)]
    candidate = os.path.join(root_dir, "output", case_name)
    return candidate if os.path.isdir(candidate) else None


def _predict_next_state(model, graph, x_hist):
    graph.x = x_hist
    graph.pos = x_hist[:, 6:9, -1] + graph.x_initial

    # Match training behavior: make delta_t node-wise.
    if hasattr(graph, "delta_t") and graph.delta_t is not None:
        dt = graph.delta_t
        if dt.dim() == 0:
            graph.delta_t = dt.repeat(graph.num_nodes)
        elif dt.dim() == 1 and dt.numel() == 1:
            graph.delta_t = dt.repeat(graph.num_nodes)
        elif hasattr(graph, "batch") and dt.dim() == 1 and dt.numel() != graph.num_nodes:
            graph.delta_t = dt[graph.batch]
    pred = model(graph)
    x_last = x_hist[:, :, -1]

    if pred.shape[1] == x_last.shape[1]:
        return pred
    if pred.shape[1] == x_last.shape[1] - 3:
        next_state = x_last.clone()
        next_state[:, 3:] = pred
        return next_state

    raise ValueError(f"Unsupported output dim {pred.shape[1]} for state dim {x_last.shape[1]}.")


def _get_pred_pos(x_initial, state):
    if state.shape[1] < 9:
        raise ValueError(f"Need at least 9 state features, got {state.shape[1]}.")
    disp = state[:, 6:9]
    return x_initial + disp


def _build_rollout(model, dataset, start_index, rollout_steps, device):
    start_graph = dataset[start_index].to(device)
    x_hist = start_graph.x.clone()  # (N, F, T)
    x_initial = start_graph.x_initial

    pred_positions = []
    gt_positions = []

    with torch.no_grad():
        for step in range(rollout_steps):
            sample_idx = start_index + step

            # Ground truth position at next step
            gt_graph = dataset[sample_idx]
            gt_y = gt_graph.y
            if gt_y.dim() == 3:
                gt_y = gt_y[:, 0, :]  # first future step
            gt_disp = gt_y[:, 6:9]
            gt_pos_next = x_initial.detach().cpu() + gt_disp
            gt_positions.append(gt_pos_next.numpy())

            # Predict position from GNN next state
            graph_in = dataset[sample_idx].to(device)
            next_state = _predict_next_state(model, graph_in, x_hist)
            pred_pos = _get_pred_pos(x_initial, next_state).detach().cpu().numpy()
            pred_positions.append(pred_pos)

            # Update history with new state for next prediction
            x_hist = torch.cat([x_hist[:, :, 1:], next_state.unsqueeze(-1)], dim=2)

    return np.asarray(gt_positions), np.asarray(pred_positions), x_initial.detach().cpu().numpy()


def _plot_rollout_mae(gt_positions, pred_positions, output_png_path):
    mae_per_step = np.mean(np.linalg.norm(pred_positions - gt_positions, axis=2), axis=1)
    steps = np.arange(len(mae_per_step))

    plt.figure(figsize=(8, 4))
    plt.plot(steps, mae_per_step, marker="o")
    plt.xlabel("Rollout step")
    plt.ylabel("MAE (position norm)")
    plt.title("Rollout Error Over Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=150)
    plt.close()
    print(f"Saved rollout MAE plot to: {output_png_path}")


def _animate_mesh(gt_positions, pred_positions, x_initial, base_grid, interval_ms, warp_scale, output_mp4_path):
    gt_grid = base_grid.copy(deep=True)
    pred_grid = base_grid.copy(deep=True)

    gt_disp0 = np.linalg.norm(gt_positions[0] - x_initial, axis=1)
    pr_disp0 = np.linalg.norm(pred_positions[0] - x_initial, axis=1)
    disp_all = np.concatenate([gt_disp0, pr_disp0])
    clim = [float(disp_all.min()), float(disp_all.max())]

    # gt_grid.points = x_initial + (gt_positions[0] - x_initial) * warp_scale
    # pred_grid.points = x_initial + (pred_positions[0] - x_initial) * warp_scale
    gt_grid.points = gt_positions[0]
    pred_grid.points = pred_positions[0]
    gt_grid["disp_mag"] = gt_disp0
    pred_grid["disp_mag"] = pr_disp0

    plotter = pv.Plotter(shape=(1, 2))

    plotter.subplot(0, 0)
    plotter.add_text("Ground Truth", font_size=12)
    # plotter.add_axes_at_origin(xlabel="X", ylabel="Y", zlabel="Z", line_width=3, labels_off=False)
    plotter.add_axes_at_origin(labels_off=True)
    plotter.add_mesh(
        gt_grid,
        scalars="disp_mag",
        cmap="turbo",
        clim=clim,
        show_edges=True,
        copy_mesh=False,
        name="gt_mesh",
    )

    plotter.subplot(0, 1)
    plotter.add_text("Prediction", font_size=12)
    plotter.add_axes_at_origin(labels_off=True)
    plotter.add_mesh(
        pred_grid,
        scalars="disp_mag",
        cmap="turbo",
        clim=clim,
        show_edges=True,
        copy_mesh=False,
        name="pred_mesh",
    )

    plotter.link_views()

    n_frames = gt_positions.shape[0]
    interval_s = max(0.001, interval_ms / 1000.0)

    plotter.open_movie(output_mp4_path, framerate=max(1, int(round(1000.0 / interval_ms))))
    plotter.show(auto_close=False, interactive_update=True)

    for f in range(n_frames):
        gt_p = gt_positions[f]
        pr_p = pred_positions[f]
        # gt_points = x_initial + (gt_p - x_initial) * warp_scale
        # pr_points = x_initial + (pr_p - x_initial) * warp_scale
        gt_points = gt_p
        pr_points = pr_p
        gt_disp = np.linalg.norm(gt_p - x_initial, axis=1)
        pr_disp = np.linalg.norm(pr_p - x_initial, axis=1)

        try:
            print(f"Updating frame {f+1}/{n_frames}...")
            plotter.update_coordinates(gt_points, mesh=gt_grid, render=False)
            plotter.update_coordinates(pr_points, mesh=pred_grid, render=False)
            plotter.update_scalars(gt_disp, mesh=gt_grid, render=False)
            plotter.update_scalars(pr_disp, mesh=pred_grid, render=False)
        except Exception:
            print(f"Error updating frame {f+1}, skipping update.")
            gt_grid.points = gt_points
            pred_grid.points = pr_points
            gt_grid["disp_mag"] = gt_disp
            pred_grid["disp_mag"] = pr_disp
            gt_grid.Modified()
            pred_grid.Modified()

        plotter.render()
        plotter.write_frame()
        plotter.update()
        time.sleep(interval_s)

    plotter.close()
    print(f"Saved rollout animation to: {output_mp4_path}")


def main():
    # Get arguments
    args = parse_args()

    # Start DPF server and load dataset/model
    server = dpf.start_local_server(ansys_path=r"C:\Program Files\ANSYS Inc\v242", as_global=True)

    # Load dataset from npz file (with geometry)
    dataset = FEMDataset(args.npz_path, geometry_path=args.geometry_path)
    if not (0 <= args.start_index < len(dataset)):
        raise IndexError(f"start-index {args.start_index} out of range for dataset size {len(dataset)}")

    # Load GNN model
    device = torch.device(args.device)
    model_cfg = ModelConfig.from_json(args.config)
    model = create_gnn_model(model_cfg).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Validate rollout parameters
    max_steps = len(dataset) - args.start_index
    rollout_steps = min(args.rollout_steps, max_steps)
    if rollout_steps <= 0:
        raise ValueError("rollout-steps is too large for selected start-index.")

    # Build rollout predictions
    gt_positions, pred_positions, x_initial = _build_rollout(
        model, dataset, args.start_index, rollout_steps, device
    )

    # Load d3plot and animate
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
    output_mp4_path = os.path.join(results_dir, "rollout_prediction.mp4")

    _animate_mesh(
        gt_positions=gt_positions,
        pred_positions=pred_positions,
        x_initial=x_initial,
        base_grid=base_grid,
        interval_ms=args.interval_ms,
        warp_scale=args.warp_scale,
        output_mp4_path=output_mp4_path,
    )
    mae_plot_path = os.path.join(results_dir, "rollout_prediction_mae.png")
    _plot_rollout_mae(gt_positions, pred_positions, mae_plot_path)


if __name__ == "__main__":
    main()
