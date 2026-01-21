import os
import numpy as np
from ansys.dpf import post
from ansys.dpf import core as dpf

from utils.ansys_utils import run_cfile_ls_prepost
from utils.ls_prepost_utils import *


# root = os.path.dirname(os.path.abspath(__file__))
root = os.getcwd()
d3plot_path = os.path.join(root, "output", "d3plot")
keyword_path = os.path.join(root, "output", "ball_plate.k")
cfile_path = os.path.join(root, "cfile", "element_mass_all.cfile")
msg_path = os.path.join(root, "cfile", "lspost.msg")


# Load server
server = dpf.start_local_server(ansys_path=r"C:\Program Files\ANSYS Inc\v242", as_global=True)

# Load model and mesh
model = load_model_metadata(d3plot_path)
mesh = load_mesh(d3plot_path)

# Read mesh
element_to_nodes = build_element_connectivity(mesh)
element_to_material = build_element_material_map(mesh)

#### Calculate mass of elements
# Create ls-dyna cfile
write_mass_cfile(cfile_path, keyword_path, element_to_nodes)
# Run cfile
# run_cfile_ls_prepost(cfile_path)
# Get mass elements
element_masses = read_element_mass_map(msg_path)
node_mass, missing_elements = compute_node_masses(element_masses, element_to_nodes)

print(f"Elements: {len(element_to_nodes)}")
print(f"Materials: {len(element_to_material)}")
print(f"Element masses parsed: {len(element_masses)}")
print(f"Node masses computed: {len(node_mass)}")
if missing_elements:
    print(f"Missing element masses: {len(missing_elements)}")

node_mass_data_gnn = np.array(list(node_mass.values()))


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
        mat = mesh.materials.array[element_index]

        for i, j in REAL_EDGES[etype]:
            n1 = int(node_ids[i]) - 1  # zero-based
            n2 = int(node_ids[j]) - 1  # zero-based

            # add both directions
            edges_list.append((n1, n2))
            attr_list.append(mat)

            edges_list.append((n2, n1))
            attr_list.append(mat)

    if not edges_list:
        return np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=np.asarray(mesh.materials.array).dtype)

    edges_gnn = np.asarray(edges_list, dtype=int)
    edges_attr = np.asarray(attr_list)

    # Remove duplicate edges AND keep edges_attr aligned:
    # unique rows with indices of first occurrence
    _, unique_idx = np.unique(edges_gnn, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)  # keep a stable order

    edges_gnn = edges_gnn[unique_idx]
    edges_attr = edges_attr[unique_idx]

    return edges_gnn, edges_attr


edges_gnn, edges_attr = build_edges_gnn_from_mesh(mesh)


# Get node features
coordinates_fc: dpf.FieldsContainer = model.results.coordinates.on_all_time_freqs.eval()  # mm unit
acceleration_fc: dpf.FieldsContainer = model.results.acceleration.on_all_time_freqs.eval()  # mm/ms^2 unit
velocity_fc: dpf.FieldsContainer = model.results.velocity.on_all_time_freqs.eval()  # mm/ms unit
displacement_fc: dpf.FieldsContainer = model.results.displacement.on_all_time_freqs.eval()  # mm unit
kinetic_energy = model.results.global_kinetic_energy.eval()
internal_energy = model.results.global_internal_energy.eval()
total_energy = model.results.global_total_energy.eval()
# stress_field_container:dpf.FieldsContainer = model.results.stress.on_all_time_freqs.eval() # MPa unit
time = np.array(model.metadata.time_freq_support.time_frequencies.data)  # ms unit
# delta_t = time[1:] - time[:-1]

# Add distance between nodes as edge attributes (using initial coordinates)
coords0 = np.asarray(coordinates_fc[0].data)
edge_distance = np.linalg.norm(
    coords0[edges_gnn[:, 0]] - coords0[edges_gnn[:, 1]],
    axis=1,
)
edges_attr = np.stack([edges_attr.astype(float), edge_distance.astype(float)], axis=1)

##########
all_features = []   # list of T tensors
predict_features = []
delta_T = []
total_kinetic_energy = []
total_internal_energy = []
num_series = 3
num_steps = len(time)

# Step1: Extract node features for entire node at every time step
node_feat_series = []
for i in range(num_steps):  # Start from num_series-1 to T-1
    coor = np.asarray(coordinates_fc[i].data)
    acc = np.asarray(acceleration_fc[i].data)
    vel = np.asarray(velocity_fc[i].data)
    disp = np.asarray(displacement_fc[i].data)
    node_feat_series.append(np.concatenate([coor, acc, vel, disp], axis=1))  # (N, 12)
node_feat_series = np.stack(node_feat_series, axis=2)  # (N, 12, T)

for t in range(num_series - 1, num_steps - 1):
    # Extract node features for entire node at time t with history
    node_feat = node_feat_series[:, :, t - num_series + 1: t + 1]  # (N, 12, num_series)
    all_features.append(node_feat)

    # Predict residual
    x_t = node_feat_series[:, :, t]
    x_t_1 = node_feat_series[:, :, t + 1]
    y_residual = x_t_1 - x_t

    # Delta time
    delta_t = time[t + 1] - time[t]
    delta_T.append(delta_t)
    y_change = y_residual / delta_t
    predict_features.append(y_change)

    # Store total energies
    total_kinetic_energy.append(kinetic_energy[0].data[t])
    total_internal_energy.append(internal_energy[0].data[t])


data_path = os.path.join(root, "data", "ball_plate_gnn_data.npz")

# Convert element/node indices to zero-based for training data
element_to_nodes_zero = {
    int(eid) - 1: [int(nid) - 1 for nid in nodes]
    for eid, nodes in element_to_nodes.items()
}
element_to_material_zero = {
    int(eid) - 1: int(mat)
    for eid, mat in element_to_material.items()
}

solids = {eid: n for eid, n in element_to_nodes_zero.items() if len(n) == 8}
shells = {eid: n for eid, n in element_to_nodes_zero.items() if len(n) == 4}

np.savez_compressed(
    data_path,
    X_list=np.array(all_features),  # (num_samples, N, 12, num_series)
    Y_list=np.array(predict_features),  # (num_samples, N, 12)
    node_mass=node_mass_data_gnn,  # (N,)
    Delta_t=np.array(delta_T),
    element_id_solids=np.array(list(solids.keys())),  # (num_elements,)
    element_id_shells=np.array(list(shells.keys())),  # (num_elements,)
    element_to_nodes_solids=np.array(list(solids.values())),  # (num_elements, nodes_per_element)
    element_to_nodes_shells=np.array(list(shells.values())),  # (num_elements, nodes_per_element)
    element_materials=np.array(list(element_to_material_zero.values())),  # (num_elements,)
    edge_index=edges_gnn,  # (E, 2)
    edge_attr=edges_attr,  # (E,)
    total_internal_energy=total_internal_energy,  # (num_series,)
    total_kinetic_energy=total_kinetic_energy,  # (num_series,)
)
