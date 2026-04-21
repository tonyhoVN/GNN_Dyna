import argparse
import json
import os
import time
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from ansys.dpf import core as dpf

from model.model_buckling1 import EncodeProcessDecode
from model.model_creation import ModelConfig, create_gnn_model
from utils.data_loader import FEMDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark rollout for baseline / one-step / recurrent models.")
    parser.add_argument(
        "--benchmark-config",
        type=str,
        default="benchmark/rollout_benchmark.json",
        help="Path to benchmark json file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    return parser.parse_args()


def load_json(path):
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


def predict_next_state(mode, model, graph, x_hist):
    graph.x = x_hist
    x_last = x_hist[:, :, -1]

    if mode in ("one_step_prediction", "recurrent_prediction"):
        graph.pos = x_hist[:, 6:9, -1] + graph.x_initial
        expand_dt(graph)

    if mode == "baseline":
        vu_next = model(graph)  # (N,6)
        next_state = x_last.clone()
        next_state[:, 3:] = vu_next
        return next_state

    if mode == "one_step_prediction":
        pred = model(graph)
        if pred.shape[1] == x_last.shape[1]:
            return pred
        if pred.shape[1] == x_last.shape[1] - 3:
            next_state = x_last.clone()
            next_state[:, 3:] = pred
            return next_state
        raise ValueError(f"Unsupported one-step output shape: {tuple(pred.shape)}")

    if mode == "recurrent_prediction":
        pred_seq = model(graph)  # (N, H, 6)
        vu_next = pred_seq[:, 0, :]
        next_state = x_last.clone()
        next_state[:, 3:] = vu_next
        return next_state

    raise ValueError(f"Unknown model mode: {mode}")


def rollout_positions(model, mode, dataset, start_index, rollout_steps, device):
    start_graph = dataset[start_index].to(device)
    x_hist = start_graph.x.clone()
    x_initial = start_graph.x_initial

    gt_positions = []
    pred_positions = []

    with torch.no_grad():
        for step in range(rollout_steps):
            idx = start_index + step
            gt_graph = dataset[idx]
            gt_y = gt_graph.y
            if gt_y.dim() == 3:
                gt_y = gt_y[:, 0, :]
            gt_positions.append((x_initial.detach().cpu() + gt_y[:, 6:9]).numpy())

            graph_in = dataset[idx].to(device)
            next_state = predict_next_state(mode, model, graph_in, x_hist)
            pred_positions.append((x_initial + next_state[:, 6:9]).detach().cpu().numpy())

            x_hist = torch.cat([x_hist[:, :, 1:], next_state.unsqueeze(-1)], dim=2)

    gt_positions = np.asarray(gt_positions)
    pred_positions = np.asarray(pred_positions)
    return gt_positions, pred_positions, x_initial.detach().cpu().numpy()


def compute_error_curves(gt_positions, pred_positions, plate_node_count=289):
    # Per-frame per-node position error magnitude: (T, N)
    err = np.linalg.norm(pred_positions - gt_positions, axis=2)
    n_nodes = err.shape[1]
    n_plate = min(int(plate_node_count), n_nodes)
    if n_plate <= 0:
        raise ValueError("plate_node_count must be > 0 and dataset must have at least 1 node.")
    return {
        "mae_all": np.mean(err, axis=1),
        "mae_plate": np.mean(err[:, :n_plate], axis=1),
        "max_all": np.max(err, axis=1),
    }


def derive_d3plot_dir(npz_path, root_dir):
    base = os.path.basename(npz_path)
    suffix = "_time_data.npz"
    if not base.endswith(suffix):
        return None
    case_name = base[: -len(suffix)]
    candidate = os.path.join(root_dir, "output1", case_name)
    return candidate if os.path.isdir(candidate) else None


def build_model(model_entry, device):
    cfg_path = model_entry["config"]
    raw_cfg = load_json(cfg_path)
    model_cfg = ModelConfig.from_json(cfg_path)
    model_type = raw_cfg.get("model", {}).get("type", "")

    if model_type == "baseline" or model_entry.get("mode", "") == "baseline":
        model = EncodeProcessDecode(
            node_feat_size=model_cfg.node_encoder["feat_dim"],
            output_size=model_cfg.decoder["out_dim"],
            latent_size=model_cfg.hidden_dim,
            edge_feat_size=model_cfg.edge_encoder["feat_dim"],
            message_passing_steps=model_cfg.gnn_topology["n_gnn_layers"],
        ).to(device)
        mode = "baseline"
    else:
        model = create_gnn_model(model_cfg).to(device)
        mode = model_entry.get("mode", "")
        if not mode:
            if model_type == "direct_recurrent":
                mode = "recurrent_prediction"
            else:
                mode = "one_step_prediction"

    ckpt_path = model_entry.get("save_pt", model_entry.get("model_path", ""))
    if not ckpt_path:
        raise ValueError(f"Missing checkpoint path for model entry: {model_entry.get('name', '')}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model, mode


def resolve_test_files(test_cfg):
    data_dir = test_cfg["data_dir"]
    file_glob = test_cfg["file_glob"]
    if isinstance(file_glob, str):
        files = sorted(glob(os.path.join(data_dir, file_glob)))
    else:
        files = [os.path.join(data_dir, f) for f in file_glob]
        files = [f for f in files if os.path.isfile(f)]
    return files


def _plot_metric_subplot(metric_key, metric_title, ylabel, all_metrics_by_model):
    for model_name, metric_map in all_metrics_by_model.items():
        curves = metric_map.get(metric_key, [])
        if not curves:
            continue
        min_len = min(len(c) for c in curves)
        arr = np.stack([c[:min_len] for c in curves], axis=0)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        x = np.arange(min_len)
        plt.plot(x, mean, label=model_name)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Rollout step")
    plt.ylabel(ylabel)
    plt.title(metric_title)
    plt.grid(True, alpha=0.3)
    plt.legend()


def plot_benchmark_curves(all_metrics_by_model, output_png):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    _plot_metric_subplot(
        metric_key="mae_all",
        metric_title="All Mesh MAE (Mean +/- Std)",
        ylabel="Position MAE",
        all_metrics_by_model=all_metrics_by_model,
    )

    plt.subplot(1, 3, 2)
    _plot_metric_subplot(
        metric_key="mae_plate",
        metric_title="Plate MAE (Nodes 0:289, Mean +/- Std)",
        ylabel="Position MAE",
        all_metrics_by_model=all_metrics_by_model,
    )

    plt.subplot(1, 3, 3)
    _plot_metric_subplot(
        metric_key="max_all",
        metric_title="All Mesh Max Error (Mean +/- Std)",
        ylabel="Max Position Error",
        all_metrics_by_model=all_metrics_by_model,
    )

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Saved benchmark plot to: {output_png}")


def animate_four_panel(first_file, model_results, interval_ms, output_mp4):
    # model_results: dict[name] -> (gt_positions, pred_positions, x_initial)
    first_key = next(iter(model_results))
    gt_positions, _, x_initial = model_results[first_key]
    baseline_pred = model_results["baseline"][1]
    one_step_pred = model_results["one_step_prediction"][1]
    recurrent_pred = model_results["recurrent_prediction"][1]

    root = os.getcwd()
    d3plot_dir = derive_d3plot_dir(first_file, root)
    if not d3plot_dir:
        raise FileNotFoundError("Could not derive d3plot dir from first test file.")
    ds = dpf.DataSources()
    ds.set_result_file_path(os.path.join(d3plot_dir, "d3plot"), "d3plot")
    dpf_model = dpf.Model(ds)
    base_grid = dpf_model.metadata.meshed_region.grid

    gt_grid = base_grid.copy(deep=True)
    b_grid = base_grid.copy(deep=True)
    o_grid = base_grid.copy(deep=True)
    r_grid = base_grid.copy(deep=True)

    gt_grid.points = gt_positions[0]
    b_grid.points = baseline_pred[0]
    o_grid.points = one_step_pred[0]
    r_grid.points = recurrent_pred[0]

    disp0 = [
        np.linalg.norm(gt_positions[0] - x_initial, axis=1),
        np.linalg.norm(baseline_pred[0] - x_initial, axis=1),
        np.linalg.norm(one_step_pred[0] - x_initial, axis=1),
        np.linalg.norm(recurrent_pred[0] - x_initial, axis=1),
    ]
    clim = [float(np.min(np.concatenate(disp0))), float(np.max(np.concatenate(disp0)))]
    for grid, disp in zip((gt_grid, b_grid, o_grid, r_grid), disp0):
        grid["disp_mag"] = disp

    plotter = pv.Plotter(shape=(2, 2))
    panels = [
        (0, 0, "Ground Truth", gt_grid),
        (0, 1, "Baseline", b_grid),
        (1, 0, "One Step", o_grid),
        (1, 1, "Recurrent", r_grid),
    ]
    for r, c, title, grid in panels:
        plotter.subplot(r, c)
        plotter.add_text(title, font_size=11)
        plotter.add_mesh(grid, scalars="disp_mag", cmap="turbo", clim=clim, show_edges=True, copy_mesh=False)
    
    # Add frame number in ground truth panel below title
    plotter.subplot(0, 0)
    plotter.add_text(text="Step 0", position="upper_right", font_size=11, name="frame_text")    
    
    plotter.link_views()

    n_frames = len(gt_positions)
    plotter.open_movie(output_mp4, framerate=max(1, int(round(1000.0 / interval_ms))))
    plotter.show(auto_close=False, interactive_update=True)

    for f in range(n_frames):
        frames = [gt_positions[f], baseline_pred[f], one_step_pred[f], recurrent_pred[f]]
        grids = [gt_grid, b_grid, o_grid, r_grid]
        for grid, pts in zip(grids, frames):
            disp = np.linalg.norm(pts - x_initial, axis=1)
            try:
                plotter.update_coordinates(pts, mesh=grid, render=False)
                plotter.update_scalars(disp, mesh=grid, render=False)
            except Exception:
                grid.points = pts
                grid["disp_mag"] = disp
                grid.Modified()

        # Update frame number text
        plotter.add_text(text=f"Step {f}", position="upper_right", font_size=11, name="frame_text", render=False)

        # Render and write frame
        plotter.render()
        plotter.write_frame()
        plotter.update()
        time.sleep(max(0.001, interval_ms / 1000.0))

    plotter.close()
    print(f"Saved benchmark animation to: {output_mp4}")


def main():
    args = parse_args()
    root = os.getcwd()
    device = torch.device(args.device)

    bench_cfg = load_json(args.benchmark_config)
    test_cfg = bench_cfg["test_data"]
    model_entries = bench_cfg["models"]

    files = resolve_test_files(test_cfg)
    if not files:
        raise FileNotFoundError("No test files found from benchmark config.")

    os.makedirs("results", exist_ok=True)
    dpf.start_local_server(ansys_path=r"C:\Program Files\ANSYS Inc\v242", as_global=True)

    models = {}
    for entry in model_entries:
        model, mode = build_model(entry, device)
        models[mode] = (entry["name"], model)

    required_modes = ["baseline", "one_step_prediction", "recurrent_prediction"]
    for m in required_modes:
        if m not in models:
            raise ValueError(f"Missing required model mode '{m}' in benchmark config.")

    start_index = int(test_cfg.get("start_index", 10))
    rollout_steps_cfg = int(test_cfg.get("rollout_steps", 50))
    interval_ms = int(test_cfg.get("interval_ms", 100))

    metric_keys = ["mae_all", "mae_plate", "max_all"]
    all_metrics = {models[m][0]: {k: [] for k in metric_keys} for m in required_modes}
    first_file_rollouts = {}

    for i, npz_path in enumerate(files):
        dataset = FEMDataset(npz_path)
        max_steps = len(dataset) - start_index
        rollout_steps = min(rollout_steps_cfg, max_steps)
        if rollout_steps <= 0:
            print(f"Skip {npz_path}: insufficient length for start index {start_index}")
            continue

        per_file = {}
        for mode in required_modes:
            model_name, model = models[mode]
            gt_pos, pred_pos, x0 = rollout_positions(model, mode, dataset, start_index, rollout_steps, device)
            curves = compute_error_curves(gt_pos, pred_pos, plate_node_count=289)
            for k in metric_keys:
                all_metrics[model_name][k].append(curves[k])
            per_file[mode] = (gt_pos, pred_pos, x0)

        if i == 0:
            first_file_rollouts = per_file

    plot_path = os.path.join(root, "results", "rollout_benchmark_mae.png")
    plot_benchmark_curves(all_metrics, plot_path)

    if first_file_rollouts:
        anim_path = os.path.join(root, "results", "rollout_benchmark.mp4")
        animate_four_panel(files[0], first_file_rollouts, interval_ms, anim_path)


if __name__ == "__main__":
    main()



