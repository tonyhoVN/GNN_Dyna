import argparse
import os

import numpy as np
import torch
from ansys.dpf import core as dpf
from ansys.dpf.core.plotter import DpfPlotter

from model.GNN import MeshGraphNet
from utils.data_loader import FEMDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load MeshGraphNet and plot one-step velocity/displacement prediction vs ground truth."
    )
    parser.add_argument("--model_path", required=True, help="Path to MeshGraphNet .pt file")
    parser.add_argument("--npz_path", required=True, help="Path to *_gnn_data.npz file")
    parser.add_argument(
        "--d3plot_dir",
        default=None,
        help="Folder containing d3plot (default: derive from npz filename)",
    )
    parser.add_argument(
        "--sample_index", 
        type=int, 
        default=30, 
        help="Dataset sample index")
    parser.add_argument(
        "--time_index",
        type=int,
        default=None,
        help="DPF time index for ground truth (default: sample_index + history_len)",
    )
    parser.add_argument(
        "--disp_index",
        type=int,
        default=30,
        help="Time index to plot displacement field",
    )
    parser.add_argument(
        "--ansys_path",
        default=r"C:\Program Files\ANSYS Inc\v242",
        help="ANSYS install path for DPF server",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device",
    )
    return parser.parse_args()


def derive_d3plot_dir(npz_path, root):
    base = os.path.basename(npz_path)
    suffix = "_gnn_data.npz"
    if base.endswith(suffix):
        folder = base[: -len(suffix)]
        candidate = os.path.join(root, "output", folder)
        if os.path.isdir(candidate):
            return candidate
    return None


def create_vector_field(node_ids, values, meshed_region=None):
    try:
        field = dpf.fields_factory.create_3d_vector_field(
            len(node_ids), location=dpf.locations.nodal
        )
    except AttributeError:
        field = dpf.Field(
            nentities=len(node_ids),
            nature=dpf.natures.vector,
            location=dpf.locations.nodal,
            dimensionality=3,
        )
    field.scoping.ids = node_ids
    field.data = values
    if meshed_region is not None:
        field.meshed_region = meshed_region
    return field


def create_scalar_field(node_ids, values, meshed_region=None):
    try:
        field = dpf.fields_factory.create_scalar_field(
            len(node_ids), location=dpf.locations.nodal
        )
    except AttributeError:
        field = dpf.Field(
            nentities=len(node_ids),
            nature=dpf.natures.scalar,
            location=dpf.locations.nodal,
            dimensionality=1,
        )
    field.scoping.ids = node_ids
    field.data = values
    if meshed_region is not None:
        field.meshed_region = meshed_region
    return field


def plot_disp_subplots(gt_field, pred_field, err_field, meshed_region, title):
    ncols = 3 if err_field is not None else 2
    plotter = DpfPlotter(shape=(1, ncols))
    pv_plotter = plotter._internal_plotter._plotter

    pv_plotter.subplot(0, 0)
    plotter.add_field(gt_field, meshed_region=meshed_region)
    pv_plotter.add_text(f"{title} (ground truth)", position="upper_left", font_size=12)

    pv_plotter.subplot(0, 1)
    plotter.add_field(pred_field, meshed_region=meshed_region)
    pv_plotter.add_text(f"{title} (prediction)", position="upper_left", font_size=12)

    if err_field is not None:
        pv_plotter.subplot(0, 2)
        plotter.add_field(err_field, meshed_region=meshed_region)
        pv_plotter.add_text(f"{title} (abs error)", position="upper_left", font_size=12)

    plotter.show_figure()


def main():
    args = parse_args()

    if args.ansys_path and os.path.isdir(args.ansys_path):
        dpf.start_local_server(ansys_path=args.ansys_path, as_global=True)

    root = os.getcwd()
    d3plot_dir = args.d3plot_dir or derive_d3plot_dir(args.npz_path, root)
    if not d3plot_dir:
        raise FileNotFoundError(
            "d3plot_dir not found; pass --d3plot_dir or use *_gnn_data.npz filename."
        )

    d3plot_path = d3plot_dir
    if os.path.isdir(d3plot_path):
        d3plot_path = os.path.join(d3plot_path, "d3plot")

    dataset = FEMDataset(args.npz_path)
    graph = dataset[args.sample_index]

    node_dim = graph.x.shape[1]
    out_dim = graph.y.shape[1]
    edge_dim = graph.edge_attr.shape[1] if graph.edge_attr is not None else 1

    model = MeshGraphNet(node_dim=node_dim, edge_dim=edge_dim, out_dim=out_dim).to(args.device)
    state = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state)
    model.eval()

    graph_device = graph.to(args.device)
    with torch.no_grad():
        pred_change = model(graph_device)

    delta_t = float(graph_device.delta_t)
    last_avd = graph_device.x[:, 3:12, -1]
    next_avd = last_avd + pred_change * delta_t
    pred_acc = next_avd[:, 0:3].detach().cpu().numpy()
    pred_vel = next_avd[:, 3:6].detach().cpu().numpy()
    pred_disp = next_avd[:, 6:9].detach().cpu().numpy()
    pred_coords = (
        graph_device.x_initial + next_avd[:, 6:9]
    ).detach().cpu().numpy()

    ds = dpf.DataSources()
    ds.set_result_file_path(d3plot_path, "d3plot")
    dpf_model = dpf.Model(ds)

    meshed_region = dpf_model.metadata.meshed_region
    velocity_fc = dpf_model.results.velocity.on_all_time_freqs.eval()
    displacement_fc = dpf_model.results.displacement.on_all_time_freqs.eval()
    history_len = graph.x.shape[2]
    gt_time_index = args.time_index
    if gt_time_index is None:
        gt_time_index = args.sample_index + history_len
    gt_vel_field = velocity_fc[gt_time_index]
    gt_disp_field = displacement_fc[gt_time_index]

    pred_vel_field = create_vector_field(
        gt_vel_field.scoping.ids, pred_vel, meshed_region=meshed_region
    )
    pred_disp_field = create_vector_field(
        gt_disp_field.scoping.ids, pred_disp, meshed_region=meshed_region
    )
    gt_disp = np.asarray(gt_disp_field.data)
    disp_error = np.linalg.norm(pred_disp - gt_disp, axis=1)
    err_field = create_scalar_field(
        gt_disp_field.scoping.ids, disp_error, meshed_region=meshed_region
    )
    err_field.name = "disp_abs_error"

    # print(f"Plotting ground truth velocity at time index {gt_time_index}")
    # gt_vel_field.plot()
    # print(f"Plotting predicted velocity at time index {gt_time_index}")
    # pred_vel_field.plot()

    print(f"Plotting displacement subplots at time index {gt_time_index}")
    plot_disp_subplots(
        gt_disp_field,
        pred_disp_field,
        None,
        meshed_region,
        title=f"Displacement t={gt_time_index}",
    )

    err_field.plot()

    # over_time_disp = displacement_fc
    # print(f"Plotting displacement at time index {args.disp_index}")
    # over_time_disp[args.disp_index].plot()


if __name__ == "__main__":
    main()
