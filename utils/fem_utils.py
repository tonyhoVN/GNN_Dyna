import os
import numpy as np
import matplotlib.pyplot as plt
from ansys.dpf import core as dpf
import pyvista as pv

# -----------------------------
# Helpers
# -----------------------------
def displacement_in_mesh_node_order(field, nodes):
    """
    Remap nodal displacement field data to the same node order as mesh.nodes.
    """
    ind, mask = nodes.map_scoping(field.scoping)
    if not np.all(mask):
        raise ValueError("Some field-scoped nodes are missing from mesh.nodes.")
    u = np.array(field.data, dtype=float)   # field order
    u_mesh = np.empty((nodes.n_nodes, u.shape[1]), dtype=float)
    u_mesh[ind] = u
    return u_mesh

def element_displacement_gradient(X, U):
    """
    Least-squares affine fit:
        U ≈ c + (X - Xc) @ G
    Returns G = grad(u), shape (3, 3).
    """
    Xc = X.mean(axis=0)
    Uc = U.mean(axis=0)

    dX = X - Xc # (N,3)
    dU = U - Uc # (N,3)

    # Solve dX @ G ≈ dU
    # G is the displacement gradient du/dX
    G, *_ = np.linalg.lstsq(dX, dU, rcond=None)

    print(G.shape)

    return G

def small_strain_from_grad_u(G):
    return 0.5 * (G + G.T)

def green_lagrange_from_grad_u(G):
    F = np.eye(3) + G
    return 0.5 * (F.T @ F - np.eye(3))

def strain_tensor_to_voigt(E):
    """
    Return tensorial Voigt order:
    [exx, eyy, ezz, exy, eyz, exz]
    """
    return np.array([E[0, 0], E[1, 1], E[2, 2], E[0, 1], E[1, 2], E[0, 2]], dtype=float)

def equivalent_strain_from_tensor(E):
    """
    Von-Mises-like equivalent strain based on deviatoric small-strain tensor.
    Uses tensor shear strain terms (not engineering gamma).
    """
    tr = np.trace(E)
    dev = E - np.eye(3) * tr / 3.0
    return np.sqrt((2.0 / 3.0) * np.sum(dev * dev))


if __name__ == "__main__":
    # -----------------------------
    # Load model
    # -----------------------------
    ds = dpf.DataSources()
    binout_path = os.path.abspath(os.path.join('..', r"output\20260302_133256", "d3plot"))
    ds.set_result_file_path(binout_path, "d3plot")
    model = dpf.Model(ds)

    mesh = model.metadata.meshed_region
    nodes = mesh.nodes
    elements = mesh.elements

    print(model)
    print(f"n_nodes = {nodes.n_nodes}, n_elements = {elements.n_elements}")

    # -----------------------------
    # Read nodal displacement for all time steps
    # -----------------------------
    disp_fc = model.results.displacement.on_all_time_freqs.eval()
    n_steps = len(disp_fc)

    # Mesh node coordinates in mesh order
    coords_mesh_order = np.array(nodes.coordinates_field.data, dtype=float)  # shape: (N, 3)

    # Mapping from node ID -> mesh index
    id_to_index = nodes.mapping_id_to_index

    # Precompute element -> node-index list, and centroids
    elem_node_indices = []
    elem_ids = []
    elem_centroids = []

    for elem in elements:
        node_ids = elem.node_ids  # list of node IDs
        node_idx = np.array([id_to_index[nid] for nid in node_ids], dtype=int)

        elem_node_indices.append(node_idx)
        elem_ids.append(elem.id)
        elem_centroids.append(coords_mesh_order[node_idx].mean(axis=0))

    elem_centroids = np.array(elem_centroids, dtype=float)  # shape: (E, 3)


    # -----------------------------
    # Compute element strain for every time step
    # -----------------------------
    n_elems = len(elem_node_indices)

    # Store both tensor components and scalar eqv strain
    elem_strain_voigt = np.zeros((n_steps, n_elems, 6), dtype=float)
    elem_eqv_strain = np.zeros((n_steps, n_elems), dtype=float)

    for t, field in enumerate(disp_fc):
        u_mesh = displacement_in_mesh_node_order(field, nodes)  # (N, 3)

        for e, node_idx in enumerate(elem_node_indices):
            X = coords_mesh_order[node_idx]   # undeformed nodal coordinates
            U = u_mesh[node_idx]              # nodal displacement at time t

            # Skip degenerate tiny elements if needed
            if X.shape[0] < 4:
                # For beams/very small connectivities this simple 3D fit is not reliable
                elem_strain_voigt[t, e, :] = np.nan
                elem_eqv_strain[t, e] = np.nan
                continue

            G = element_displacement_gradient(X, U)

            # Small strain tensor
            # E_small = small_strain_from_grad_u(G)

            # If you expect larger deformation, replace with:
            E_small = green_lagrange_from_grad_u(G)

            elem_strain_voigt[t, e, :] = strain_tensor_to_voigt(E_small)
            elem_eqv_strain[t, e] = equivalent_strain_from_tensor(E_small)

    print("elem_strain_voigt shape:", elem_strain_voigt.shape)   # (T, E, 6)
    print("elem_eqv_strain shape:", elem_eqv_strain.shape)       # (T, E)


    # -----------------------------
    # Manual nodal averaging of equivalent strain
    # -----------------------------
    nodal_eqv_strain = np.zeros((n_steps, nodes.n_nodes), dtype=float)
    nodal_counts = np.zeros(nodes.n_nodes, dtype=int)

    # Precompute node -> attached elements
    node_to_elems = [[] for _ in range(nodes.n_nodes)]
    for e, node_idx in enumerate(elem_node_indices):
        for ni in node_idx:
            node_to_elems[ni].append(e)

    for ni in range(nodes.n_nodes):
        nodal_counts[ni] = len(node_to_elems[ni])

    for t in range(n_steps):
        for ni in range(nodes.n_nodes):
            attached = node_to_elems[ni]
            if attached:
                nodal_eqv_strain[t, ni] = np.nanmean(elem_eqv_strain[t, attached])
            else:
                nodal_eqv_strain[t, ni] = np.nan

    print("nodal_eqv_strain shape:", nodal_eqv_strain.shape)  # (T, N)

    # Visual nodal equivalent strain
    grid = mesh.grid.copy()
    t_plot = 30

    # One scalar per node -> point_data
    grid.point_data["eqv_strain_nodal"] = nodal_eqv_strain[t_plot]

    pl = pv.Plotter()
    pl.add_mesh(grid, scalars="eqv_strain_nodal", show_edges=True)
    pl.add_text(f"Nodal averaged equivalent strain, time step {t_plot + 1}", font_size=12)
    pl.show()