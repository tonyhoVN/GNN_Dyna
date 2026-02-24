import os
import numpy as np
from ansys.dpf import post
from ansys.dpf import core as dpf
import argparse
from utils.ansys_utils import run_cfile_ls_prepost
from utils.ls_prepost_utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN from JSON config")
    parser.add_argument(
        "--skip-time", 
        dest="skip_time", 
        action="store_true", 
        help="Skip building temporal data"
    )
    parser.set_defaults(skip_time=False)
    return parser.parse_args()

# Build edges for GNN
def build_edges_gnn_from_mesh(mesh):
    """
    Build directed edges (2 per undirected mesh edge) for a GNN from a mesh.

    Returns
    -------
    edges_gnn : np.ndarray, shape (E, 2), dtype=int
        Directed edges in zero-based node indexing.
    edges_attr : np.ndarray, shape (E,)
        Edge attribute per directed edge (material id/value per element).
    """
    edges_list = []
    attr_list = []

    for element_index, element in enumerate(mesh.elements):
        etype = str(element.type).split(".")[-1]  # e.g. element_types.Hex8 -> "Hex8"
        node_ids = element.node_ids

        if etype not in REAL_EDGES:
            continue

        # element-level attribute (material for this element)
        mat = mesh.materials.array[element_index] - 1  # zero-based

        for i, j in REAL_EDGES[etype]:
            n1 = int(node_ids[i]) - 1  # zero-based
            n2 = int(node_ids[j]) - 1  # zero-based

            # add n1 -> n2 if target n2 is free
            # Allow: SPC->normal, normal->normal
            if n2 not in SPC_NODES:
                edges_list.append((n1, n2))
                attr_list.append(MATERIAL_PROPERTIES[mat])

            # add n2 -> n1 if target n1 is free
            if n1 not in SPC_NODES:
                edges_list.append((n2, n1))
                attr_list.append(MATERIAL_PROPERTIES[mat])

    if not edges_list:
        return np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=np.asarray(mesh.materials.array).dtype)

    edges_gnn = np.asarray(edges_list, dtype=int)
    edges_attr = np.asarray(attr_list)

    # Remove duplicate edges AND keep edges_attr aligned:
    # unique rows with indices of first occurrence
    _, unique_idx = np.unique(edges_gnn, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)  # keep a stable order
    return edges_gnn[unique_idx], edges_attr[unique_idx]


def build_geometry_data(data_folder: str):
    d3plot_path = os.path.join(data_folder, "d3plot")
    keyword_path = os.path.join(data_folder, "ball_plate.k")
    cfile_path = os.path.join(root, "cfile", "element_mass_all.cfile")
    msg_path = os.path.join(root, "cfile", "lspost.msg")

    # Load model and mesh
    model = load_model_metadata(d3plot_path)
    mesh = load_mesh(d3plot_path)

    # Build element to nodes and material mapping
    element_to_nodes = build_element_connectivity(mesh)
    element_to_material = build_element_material_map(mesh)

    # Compute node masses
    write_mass_cfile(cfile_path, keyword_path, element_to_nodes)
    run_cfile_ls_prepost(cfile_path)
    element_masses = read_element_mass_map(msg_path)
    node_mass, missing_elements = compute_node_masses(element_masses, element_to_nodes)

    print(f"Elements: {len(element_to_nodes)}")
    print(f"Materials: {len(element_to_material)}")
    print(f"Element masses parsed: {len(element_masses)}")
    print(f"Node masses computed: {len(node_mass)}")
    if missing_elements:
        print(f"Missing element masses: {len(missing_elements)}")

    # Prepare node mass array aligned with zero-based node indexing
    node_mass_data_gnn = np.array(list(node_mass.values()))

    # Build edges connection and edge attribute for GNN
    edges_gnn, edges_attr = build_edges_gnn_from_mesh(mesh) # edges_attr: material id per edge

    # Node dis
    initial_fc: dpf.FieldsContainer = model.results.initial_coordinates.on_all_time_freqs.eval()
    coords0 = np.asarray(initial_fc[0].data)

    edge_distance = np.linalg.norm(
        coords0[edges_gnn[:, 0]] - coords0[edges_gnn[:, 1]],
        axis=1,
    ).reshape(-1, 1)

    edges_attr = np.hstack([edges_attr.astype(float), edge_distance.astype(float)])

    # Convert element and node IDs to zero-based indexing
    element_to_nodes_zero = {
        int(eid) - 1: [int(nid) - 1 for nid in nodes]
        for eid, nodes in element_to_nodes.items()
    }
    element_to_material_zero = {
        int(eid) - 1: int(mat) - 1
        for eid, mat in element_to_material.items()
    }

    # Separate solids and shells elements
    solids = {eid: n for eid, n in element_to_nodes_zero.items() if len(n) == 8}
    shells = {eid: n for eid, n in element_to_nodes_zero.items() if len(n) == 4}

    # Construct surface node lists (zero-based)
    plate_surf_nodes = [nid - 1 for nid in get_nodes_on_surface(keyword_path, part_id=1)]
    ball_surf_nodes = [nid - 1 for nid in get_nodes_on_surface(keyword_path, part_id=2)]
    # Connection between nodes of 2 surfaces 
    edge_surf_index = []
    for i in plate_surf_nodes:
        for j in ball_surf_nodes:
            edge_surf_index.append((i,j))
            if i not in SPC_NODES:
                edge_surf_index.append((j,i))
    edge_surf_index = np.asarray(edge_surf_index, dtype=int) #(S, 2)

    # Fixed boundary nodes with zero-based indexing
    boundary_nodes = np.zeros(node_mass_data_gnn.shape[0], dtype=np.float32) # (N,)
    spc_ = [ind for ind in SPC_NODES]
    boundary_nodes[spc_] = 1 # Zerobased index
    
    return {
        "initial_coords": coords0,
        "node_mass": node_mass_data_gnn,
        "boundary_constraint": boundary_nodes,
        "element_id_solids": np.array(list(solids.keys())),
        "element_id_shells": np.array(list(shells.keys())),
        "element_to_nodes_solids": np.array(list(solids.values())),
        "element_to_nodes_shells": np.array(list(shells.values())),
        "element_materials": np.array(list(element_to_material_zero.values())),
        "edge_index": edges_gnn,
        "edge_attr": edges_attr,
        "edge_surf_index": edge_surf_index
    }


def process_gnn_data(data_folder: str, geometry_path: str):
    d3plot_path = os.path.join(data_folder, "d3plot")
    save_data_path = os.path.join(
        root, "data", os.path.basename(data_folder) + "_time_data.npz"
    )

    model = load_model_metadata(d3plot_path)

    ##########
    all_node_features = []   # list of T tensors
    predict_features = []
    all_node_pos = [] # Position of all nodes in graph
    delta_T = []
    total_kinetic_energy = []
    total_internal_energy = []
    num_series = 3


    # Get node features
    coordinates_fc: dpf.FieldsContainer = model.results.coordinates.on_all_time_freqs.eval()  # mm unit
    acceleration_fc: dpf.FieldsContainer = model.results.acceleration.on_all_time_freqs.eval()  # mm/ms^2 unit
    velocity_fc: dpf.FieldsContainer = model.results.velocity.on_all_time_freqs.eval()  # mm/ms unit
    displacement_fc: dpf.FieldsContainer = model.results.displacement.on_all_time_freqs.eval()  # mm unit
    kinetic_energy = model.results.global_kinetic_energy.eval()
    internal_energy = model.results.global_internal_energy.eval()
    # stress_field_container:dpf.FieldsContainer = model.results.stress.on_all_time_freqs.eval() # MPa unit
    time = np.array(model.metadata.time_freq_support.time_frequencies.data)  # ms unit
    # delta_t = time[1:] - time[:-1]

    # Step1: Extract node features for entire node at every time step
    num_steps = len(time)
    node_feat_series = []
    for i in range(num_steps):  # Start from num_series-1 to T-1
        acc = np.asarray(acceleration_fc[i].data)
        vel = np.asarray(velocity_fc[i].data)
        disp = np.asarray(displacement_fc[i].data)
        node_feat_series.append(np.concatenate([acc, vel, disp], axis=1))  # (N, 9)
    node_feat_series = np.stack(node_feat_series, axis=2)  # (N, 9, T)

    # Step2: Get prediction features and energies
    for t in range(num_series - 1, num_steps - 1):
        # Extract node features for entire node at time t with history
        node_feat = node_feat_series[:, :, t - num_series + 1: t + 1]  # (N, 9, num_series)
        # node_feat = node_feat.reshape(node_feat.shape[0], -1)  # (N, 9*num_series)
        all_node_features.append(node_feat)

        # Position of nodes at time t
        all_node_pos.append(coordinates_fc[t].data)

        # Delta time
        delta_t = time[t + 1] - time[t]
        delta_T.append(delta_t)

        # Node feature prediction
        x_t_1 = node_feat_series[:, :, t + 1]
        predict_features.append(x_t_1) # (N, 9)

        # Store total energies include current and next step
        # current step energy used for train Physics net
        # next step energy used for train GNN net
        total_kinetic_energy.append(np.array([kinetic_energy[0].data[t],
                                            kinetic_energy[0].data[t+1]])) # (2, )
        total_internal_energy.append(np.array([internal_energy[0].data[t],
                                            internal_energy[0].data[t+1]])) # (2, )

    np.savez_compressed(
        save_data_path,
        X_list=np.array(all_node_features),  # (num_samples, N, 9, num_series)
        Y_list=np.array(predict_features),  # (num_samples, N, 9)
        pos_list = np.array(all_node_pos),  # (num_samples, N, 3)
        total_internal_energy=total_internal_energy,  # (num_samples, 2)
        total_kinetic_energy=total_kinetic_energy,  # (num_samples, 2)
        Delta_t=np.array(delta_T), # (num_samples,)
        geometry_path=geometry_path,
    )

    print(f"GNN data saved to: {save_data_path}")

if __name__ == "__main__":
    # Load server
    root = os.getcwd()
    server = dpf.start_local_server(ansys_path=r"C:\Program Files\ANSYS Inc\v242", as_global=True)
    args = parse_args()
    
    # Process all data folders in output/
    data_folders = [
        os.path.join(root, "output", d)
        for d in os.listdir(os.path.join(root, "output"))
        if os.path.isdir(os.path.join(root, "output", d))
    ]

    # Save geometry information
    geometry_path = os.path.join(root, "data", "geometry_shared.npz")
    geometry_data = build_geometry_data(data_folders[0])
    np.savez_compressed(geometry_path, **geometry_data)
    print(f"Geometry data saved to: {geometry_path}")

    # Save time series information
    if not args.skip_time:
        for folder in data_folders:
            print(f"Processing time series of folder: {folder}")
            process_gnn_data(folder, geometry_path)
