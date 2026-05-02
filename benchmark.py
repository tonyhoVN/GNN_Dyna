import argparse
import json
import math
import os
import time
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from ansys.dpf import core as dpf

from model.model_buckling1 import EncodeProcessDecode
from model.model_creation import ModelConfig, create_gnn_model
from utils.data_loader import FEMDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark rollout for models listed in the benchmark config.")
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
    parser.add_argument(
        "--plate-node-count",
        type=int,
        default=289,
        help="Number of plate nodes in the dataset (used to separate plate vs ball nodes for error metrics).",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path_str, root_dir):
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = Path(root_dir) / path
    return str(path)


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
    graph.x = x_hist
    if hasattr(graph, "x_initial") and x_hist.shape[1] >= 9:
        graph.pos = x_hist[:, 6:9, -1] + graph.x_initial
    expand_dt(graph)


def infer_next_state_from_prediction(pred, x_last):
    if pred.dim() == 3:
        pred = pred[:, 0, :]

    if pred.shape == x_last.shape:
        return pred

    tail_start = 3 if x_last.shape[1] > 3 else 0
    if pred.shape[1] == x_last.shape[1] - tail_start:
        next_state = x_last.clone()
        next_state[:, tail_start:] = pred
        return next_state

    if pred.shape[1] < x_last.shape[1]:
        next_state = x_last.clone()
        next_state[:, -pred.shape[1] :] = pred
        return next_state

    raise ValueError(f"Unsupported prediction shape {tuple(pred.shape)} for state shape {tuple(x_last.shape)}")


def predict_next_state(model_bundle, graph, x_hist):
    set_rollout_context(graph, x_hist)
    x_last = x_hist[:, :, -1]

    t0 = time.perf_counter()
    pred = model_bundle["model"](graph)
    predict_time = time.perf_counter() - t0

    next_state = infer_next_state_from_prediction(pred, x_last)
    return next_state, predict_time


def rollout_positions(model_bundle, dataset, start_index, rollout_steps, device):
    start_graph = dataset[start_index].to(device)
    x_hist = start_graph.x.clone()
    x_initial = start_graph.x_initial

    gt_positions = []
    pred_positions = []
    total_predict_time = 0.0

    with torch.no_grad():
        for step in range(rollout_steps):
            idx = start_index + step
            gt_graph = dataset[idx]
            gt_y = gt_graph.y
            if gt_y.dim() == 3:
                gt_y = gt_y[:, 0, :]
            gt_positions.append((x_initial.detach().cpu() + gt_y[:, 6:9]).numpy())

            graph_in = dataset[idx].to(device)
            next_state, predict_time = predict_next_state(model_bundle, graph_in, x_hist)
            total_predict_time += predict_time
            pred_positions.append((x_initial + next_state[:, 6:9]).detach().cpu().numpy())

            x_hist = torch.cat([x_hist[:, :, 1:], next_state.unsqueeze(-1)], dim=2)

    gt_positions = np.asarray(gt_positions)
    pred_positions = np.asarray(pred_positions)
    avg_predict_time = total_predict_time / max(1, rollout_steps)
    return gt_positions, pred_positions, x_initial.detach().cpu().numpy(), avg_predict_time


def compute_error_curves(gt_positions, pred_positions, plate_node_count=289):
    err = np.linalg.norm(pred_positions - gt_positions, axis=2)
    n_nodes = err.shape[1]
    n_plate = min(int(plate_node_count), n_nodes)
    if n_plate <= 0:
        raise ValueError("plate_node_count must be > 0 and dataset must have at least 1 node.")
    n_ball = max(0, n_nodes - n_plate)
    return {
        "mae_all": np.mean(err, axis=1),
        "mae_plate": np.mean(err[:, :n_plate], axis=1),
        "mae_ball": np.mean(err[:, n_plate:], axis=1) if n_ball > 0 else np.mean(err, axis=1),
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


def build_model_bundle(model_entry, root_dir, device):
    cfg_path = resolve_path(model_entry["config"], root_dir)
    raw_cfg = load_json(cfg_path)
    model_cfg = ModelConfig.from_json(cfg_path)
    model_type = raw_cfg.get("model", {}).get("type", "")

    predict_mode = model_entry.get("mode", model_entry.get("predict_mode", ""))
    if not predict_mode:
        if model_type == "baseline":
            predict_mode = "baseline"
        elif model_type == "direct_recurrent":
            predict_mode = "recurrent_prediction"
        else:
            predict_mode = "one_step_prediction"

    if model_type == "baseline" or predict_mode == "baseline":
        model = EncodeProcessDecode(
            node_feat_size=model_cfg.node_encoder["feat_dim"],
            output_size=model_cfg.decoder["out_dim"],
            latent_size=model_cfg.hidden_dim,
            edge_feat_size=model_cfg.edge_encoder["feat_dim"],
            message_passing_steps=model_cfg.gnn_topology["n_gnn_layers"],
        ).to(device)
    else:
        model = create_gnn_model(model_cfg).to(device)

    ckpt_path = model_entry.get("save_pt", model_entry.get("model_path", ""))
    if not ckpt_path:
        raise ValueError(f"Missing checkpoint path for model entry: {model_entry.get('name', '')}")
    ckpt_path = resolve_path(ckpt_path, root_dir)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    return {
        "name": model_entry["name"],
        "model": model,
        "mode": predict_mode,
        "history_len": int(model_cfg.node_encoder.get("history_len", 5)),
        "pred_horizon": int(model_cfg.decoder.get("pred_horizon", 1)),
        "config_path": cfg_path,
    }


def resolve_test_files(test_cfg, root_dir):
    data_dir = resolve_path(test_cfg["data_dir"], root_dir)
    file_glob = test_cfg["file_glob"]
    # if isinstance(file_glob, str):
    #     files = sorted(glob(os.path.join(data_dir, file_glob)))
    # else:
    files = [os.path.join(data_dir, f) for f in file_glob]
    files = [f for f in files if os.path.isfile(f)]
    return files


def _plot_accumulated_metric_subplot(metric_key, metric_title, ylabel, all_metrics_by_model):
    for model_name, metric_map in all_metrics_by_model.items():
        curves = metric_map.get(metric_key, [])
        if not curves:
            continue
        min_len = min(len(c) for c in curves)
        arr = np.stack([np.cumsum(c[:min_len]) for c in curves], axis=0)
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
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    _plot_accumulated_metric_subplot(
        metric_key="mae_all",
        metric_title="Accumulated MAE - Entire Mesh",
        ylabel="Accumulated Position MAE",
        all_metrics_by_model=all_metrics_by_model,
    )

    plt.subplot(1, 3, 2)
    _plot_accumulated_metric_subplot(
        metric_key="mae_ball",
        metric_title="Accumulated MAE - Ball",
        ylabel="Accumulated Position MAE",
        all_metrics_by_model=all_metrics_by_model,
    )

    plt.subplot(1, 3, 3)
    _plot_accumulated_metric_subplot(
        metric_key="mae_plate",
        metric_title="Accumulated MAE - Plate",
        ylabel="Accumulated Position MAE",
        all_metrics_by_model=all_metrics_by_model,
    )

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Saved benchmark plot to: {output_png}")


def animate_model_grid(first_file, model_results, model_names, interval_ms, output_mp4):
    first_key = next(iter(model_results))
    gt_positions, _, x_initial = model_results[first_key]

    root = os.getcwd()
    d3plot_dir = derive_d3plot_dir(first_file, root)
    if not d3plot_dir:
        raise FileNotFoundError("Could not derive d3plot dir from first test file.")
    ds = dpf.DataSources()
    ds.set_result_file_path(os.path.join(d3plot_dir, "d3plot"), "d3plot")
    dpf_model = dpf.Model(ds)
    base_grid = dpf_model.metadata.meshed_region.grid

    titles = ["Ground Truth"] + list(model_names)
    initial_points = [gt_positions[0]] + [model_results[name][1][0] for name in model_names]
    n_panels = len(titles)
    ncols = 2 if n_panels > 1 else 1
    nrows = math.ceil(n_panels / ncols)

    grids = []
    disp0 = [np.linalg.norm(pts - x_initial, axis=1) for pts in initial_points]
    clim = [float(np.min(np.concatenate(disp0))), float(np.max(np.concatenate(disp0)))]

    plotter = pv.Plotter(shape=(nrows, ncols))
    for idx, (title, pts, disp) in enumerate(zip(titles, initial_points, disp0)):
        r, c = divmod(idx, ncols)
        grid = base_grid.copy(deep=True)
        grid.points = pts
        grid["disp_mag"] = disp
        grids.append(grid)
        plotter.subplot(r, c)
        plotter.add_text(title, font_size=11)
        plotter.add_mesh(
            grid,
            scalars="disp_mag",
            cmap="turbo",
            clim=clim,
            show_edges=False,
            copy_mesh=False,
            show_scalar_bar=False,
        )

    plotter.subplot(0, 0)
    plotter.add_text(text="Step 0", position="upper_right", font_size=11, name="frame_text")
    plotter.link_views()

    n_frames = len(gt_positions)
    plotter.open_movie(output_mp4, framerate=max(1, int(round(1000.0 / interval_ms))))
    # plotter.show(auto_close=False, interactive_update=True)

    for f in range(n_frames):
        frames = [gt_positions[f]] + [model_results[name][1][f] for name in model_names]
        for grid, pts in zip(grids, frames):
            disp = np.linalg.norm(pts - x_initial, axis=1)
            try:
                plotter.update_coordinates(pts, mesh=grid, render=False)
                plotter.update_scalars(disp, mesh=grid, render=False)
            except Exception:
                grid.points = pts
                grid["disp_mag"] = disp
                grid.Modified()

        plotter.add_text(text=f"Step {f}", position="upper_right", font_size=11, name="frame_text", render=False)
        plotter.render()
        plotter.write_frame()
        plotter.update()
        time.sleep(max(0.001, interval_ms / 1000.0))

    # plotter.close()
    print(f"Saved benchmark animation to: {output_mp4}")


def main():
    args = parse_args()
    root = os.getcwd()
    device = torch.device(args.device)

    bench_cfg = load_json(args.benchmark_config)
    test_cfg = bench_cfg["test_data"]
    model_entries = bench_cfg["models"]

    files = resolve_test_files(test_cfg, root)
    if not files:
        raise FileNotFoundError("No test files found from benchmark config.")

    os.makedirs("results", exist_ok=True)
    dpf.start_local_server(ansys_path=r"C:\Program Files\ANSYS Inc\v242", as_global=True)

    model_bundles = [build_model_bundle(entry, root, device) for entry in model_entries]

    start_index = int(test_cfg.get("start_index", 10))
    rollout_steps_cfg = int(test_cfg.get("rollout_steps", 50))
    interval_ms = int(test_cfg.get("interval_ms", 100))
    geometry_path = resolve_path(os.path.join(test_cfg["data_dir"], test_cfg.get("geometry_path")), root)

    metric_keys = ["mae_all", "mae_plate", "mae_ball"]
    all_metrics = {bundle["name"]: {k: [] for k in metric_keys} for bundle in model_bundles}
    time_stats = {bundle["name"]: {"total_time": 0.0, "total_steps": 0} for bundle in model_bundles}
    best_file_rollouts = {}
    best_file_path = None
    best_score = float("inf")

    for npz_path in files:
        per_file = {}
        per_file_scores = []
        for bundle in model_bundles:
            if geometry_path:
                dataset = FEMDataset(
                    npz_path,
                    geometry_path=geometry_path,
                    history_len=bundle["history_len"],
                    predict_horizon=1,
                )
            else:
                dataset = FEMDataset(
                    npz_path,
                    history_len=bundle["history_len"],
                    predict_horizon=1,
                )

            max_steps = len(dataset) - start_index
            rollout_steps = min(rollout_steps_cfg, max_steps)
            if rollout_steps <= 0:
                print(f"Skip {npz_path} for {bundle['name']}: insufficient length for start index {start_index}")
                continue

            gt_pos, pred_pos, x0, avg_step_time = rollout_positions(bundle, dataset, start_index, rollout_steps, device)
            curves = compute_error_curves(gt_pos, pred_pos, plate_node_count=args.plate_node_count)
            for key in metric_keys:
                all_metrics[bundle["name"]][key].append(curves[key])
            time_stats[bundle["name"]]["total_time"] += avg_step_time * rollout_steps
            time_stats[bundle["name"]]["total_steps"] += rollout_steps
            per_file[bundle["name"]] = (gt_pos, pred_pos, x0)
            per_file_scores.append(float(np.sum(curves["mae_plate"])))

        if per_file_scores and len(per_file) == len(model_bundles):
            file_score = float(np.mean(per_file_scores))
            if file_score < best_score:
                best_score = file_score
                best_file_path = npz_path
                best_file_rollouts = per_file

    plot_path = os.path.join(root, "results", "rollout_benchmark_mae.png")
    plot_benchmark_curves(all_metrics, plot_path)

    print("Average time per rollout prediction:")
    for bundle in model_bundles:
        model_name = bundle["name"]
        total_time = time_stats[model_name]["total_time"]
        total_steps = time_stats[model_name]["total_steps"]
        avg_ms = 1000.0 * total_time / max(1, total_steps)
        print(f"  {model_name}: {avg_ms:.3f} ms / rollout step")

    if best_file_rollouts:
        anim_path = os.path.join(root, "results", "rollout_benchmark.mp4")
        print(f"Animating best file by accumulated MAE: {best_file_path}")
        animate_model_grid(
            best_file_path,
            best_file_rollouts,
            [bundle["name"] for bundle in model_bundles],
            interval_ms,
            anim_path,
        )


if __name__ == "__main__":
    main()
