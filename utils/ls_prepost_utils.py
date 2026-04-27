import re
from pathlib import Path
from typing import Dict, List, Tuple, Union
import os
from ansys.dpf.post import load_simulation
from ansys.dpf import core as dpf
from utils.ansys_utils import run_cfile_ls_prepost
from math import log1p

_MASS_LINE_RE = re.compile(
    r"Mass of (?:Solid|Shell|Beam) Ele #(?P<id>\d+)\s*=\s*"
    r"(?P<mass>[-+]?[\d.]+(?:[eE][-+]?\d+)?)"
)

REAL_EDGES = {
    "TriShell3": [
        (0,1), (1,2), (2,0)
    ],
    "QuadShell4": [
        (0,1), (1,2), (2,3), (3,0)
    ],
    "Shell4": [
        (0,1), (1,2), (2,3), (3,0)
    ],
    "Tet4": [
        (0,1), (1,2), (2,0),
        (0,3), (1,3), (2,3)
    ],
    "Hex8": [
        # Bottom square
        (0,1), (1,2), (2,3), (3,0),
        # Top square
        (4,5), (5,6), (6,7), (7,4),
        # Vertical edges
        (0,4), (1,5), (2,6), (3,7)
    ]
}

_NODE_ID_RE = re.compile(r"NODE ID=\s*(\d+)")

# MATERIAL_PROPERTIES = {
#     0: [log1p(207), log1p(0.3), log1p(0.2), log1p(2), 1, 0], # Plastic mat [E, nu, sigma_y, E_t, 1, 0]
#     1: [log1p(207), log1p(0.3), 0, 0, 0, 1], # Rigid mat [E, nu, 0, 0, 0, 1]
# }

MATERIAL_PROPERTIES = {
    0: [207, 0.3, 0.2, 2, 1, 0], # Plastic mat [E, nu, sigma_y, E_t, 1, 0]
    1: [207, 0.3, 0, 0, 0, 1], # Rigid mat [E, nu, 0, 0, 0, 1]
}

# spc_nodes = [34,35,51,52,68,69,85,86,102,103,119,120,136,137,153,154,
#             170,171,187,188,204,205,221,222,238,239,255,256] + list(range(1,19)) + list(range(272,290)) # 1-based index
# SPC_NODES = {nid - 1 for nid in spc_nodes} # 0-base index

def read_element_masses(msg_path: Union[str, Path]) -> List[Tuple[int, float]]:
    """
    Parse LS-PrePost .msg output and return (element_id, mass) pairs.
    """
    msg_path = Path(msg_path)
    if not msg_path.is_file():
        raise FileNotFoundError(f"LS-PrePost message file not found: {msg_path}")

    pairs: List[Tuple[int, float]] = []
    with msg_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = _MASS_LINE_RE.search(line)
            if not match:
                continue
            elem_id = int(match.group("id"))
            mass = float(match.group("mass"))
            pairs.append((elem_id, mass))

    return pairs

def read_element_mass_map(msg_path: Union[str, Path]) -> Dict[int, float]:
    """
    Parse LS-PrePost .msg output and return {element_id: mass}.
    """
    return dict(read_element_masses(msg_path))

def load_model_metadata(d3plot_path: str):
    ds = dpf.DataSources()
    ds.set_result_file_path(d3plot_path, "d3plot")
    model = dpf.Model(ds)
    return model

def load_mesh(d3plot_path: str):
    ds = dpf.DataSources()
    ds.set_result_file_path(d3plot_path, "d3plot")
    simulation = load_simulation(ds, "static mechanical")
    return simulation.mesh

def build_element_connectivity(mesh):
    """
    Build element to nodes connectivity from mesh with 1-based indexing.
    
    :param mesh: Mesh object from DPF
    :return: Dictionary mapping element IDs to lists of node IDs
    """
    element_to_nodes = {}
    ids = []
    for element in mesh.elements:
        element_id = element.id
        ids.append(element_id)
        element_to_nodes[element_id] = element.to_node_connectivity
    # Sort by element index
    # breakpoint()
    element_to_nodes = dict(sorted(element_to_nodes.items(), key= lambda x: x[0]))
    # breakpoint()
    return element_to_nodes

def build_element_material_map(mesh):
    materials_array = mesh.materials.array
    element_ids = list(mesh.element_ids)
    element_to_material = {}
    for idx, element_id in enumerate(element_ids):
        value = materials_array[idx]
        if hasattr(value, "item"):
            value = value.item()
        elif isinstance(value, (list, tuple)) and value:
            value = value[0]
        element_to_material[int(element_id)] = int(value)
    
    # Sort by element index
    element_to_material = dict(sorted(element_to_material.items(), key=lambda x: x[0]))
    return element_to_material

def build_element_type_map(mesh):
    element_to_type = {}
    for element in mesh.elements:
        etype = str(element.type).split(".")[-1]  # e.g. element_types.Hex8 -> "Hex8"
        element_to_type[int(element.id)] = etype

    # Sort by element index
    element_to_type = dict(sorted(element_to_type.items(), key=lambda x: x[0]))
    return element_to_type

def write_mass_cfile(output_path: str, keyword_path: str, element_to_type: Dict[int, str]):
    lines = [
        "bgstyle plain",
        f'openc keyword "{keyword_path}"',
        "",
        "measure select 1",
        "genselect target element",
        "measure type 12",
        "",
        "$# --- list of element IDs you want ---",
    ]
    for element_id, etype in element_to_type.items():
        if etype == "Hex8" or etype == "Tet4":
            lines.append(f"genselect element add solid {element_id}")
        if etype == "QuadShell4" or etype == "TriShell3" or etype == "Shell4":
            lines.append(f"genselect element add shell {element_id}")
        if etype == "Line2" or etype == "Beam3":
            lines.append(f"genselect element add beam {element_id}")
        lines.append("genselect unsel")
    lines.append("exit")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_translate_cfile(output_path: str, 
                          open_keyword_path: str, 
                          save_keyword_path: str,
                          part_id: int,
                          xyz):
    """
    Create an LS-PrePost command file that selects all nodes in a part and translates them.

    Parameters
    ----------
    output_path : str
        Where to write the .cfile (command file).
    keyword_path : str
        Path to the input keyword file (.k) that LS-PrePost opens, and also the path it saves back to.
    part_id : int
        Part ID whose nodes will be translated.
    xyz : (x, y, z)
        Translation vector.
    """
    if len(xyz) != 3:
        raise ValueError("xyz must be a 3-tuple/list like (dx, dy, dz)")

    dx, dy, dz = xyz

    lines = [
        "bgstyle plain",
        f'openc keyword "{open_keyword_path}"',
        "genselect target node",
        "genselect transfer 0",
        f"genselect node add part {part_id}",
        f"translate_model {dx} {dy} {dz}",
        "save keywordabsolute 0",
        "save keywordbylongfmt 0",
        "save keywordbyi10fmt 0",
        f'save keyword "{save_keyword_path}"',
        "exit",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def compute_node_masses(element_masses, element_to_nodes):
    """
    Compute node masses by distributing element masses equally to their nodes.
    
    :param element_masses: Dictionary mapping element IDs to their masses
    :param element_to_nodes: Dictionary mapping element IDs to lists of node IDs
    """
    node_mass = {}
    missing_elements = []
    for element_id, node_ids in element_to_nodes.items():
        mass = element_masses.get(element_id)
        if mass is None:
            missing_elements.append(element_id)
            continue
        if not node_ids:
            continue
        share = mass / len(node_ids)
        for node_id in node_ids:
            node_mass[node_id] = node_mass.get(node_id, 0.0) + share

    # sort by node_id
    node_mass = dict(sorted(node_mass.items(), key=lambda x: x[0]))
    return node_mass, missing_elements

def get_nodes_mass():
    root = os.path.dirname(os.path.abspath(__file__))
    d3plot_path = os.path.join(root, "output", "d3plot")
    keyword_path = os.path.join(root, "output", "ball_plate.k")
    cfile_path = os.path.join(root, "cfile", "element_mass_all.cfile")
    msg_path = os.path.join(root, "cfile", "lspost.msg")

    # Read mesh 
    mesh = load_mesh(d3plot_path)
    element_to_nodes = build_element_connectivity(mesh)
    element_to_material = build_element_material_map(mesh)

    #### Calculate mass of elements
    # Create ls-dyna cfile 
    write_mass_cfile(cfile_path, keyword_path, element_to_nodes)
    # Run cfile 
    run_cfile_ls_prepost(cfile_path)
    # Get mass elements
    element_masses = read_element_mass_map(msg_path)
    node_mass, missing_elements = compute_node_masses(element_masses, element_to_nodes)

    print(f"Elements: {len(element_to_nodes)}")
    print(f"Materials: {len(element_to_material)}")
    print(f"Element masses parsed: {len(element_masses)}")
    print(f"Node masses computed: {len(node_mass)}")
    print(f"type: {type(node_mass.values())}")
    if missing_elements:
        print(f"Missing element masses: {len(missing_elements)}")
    
    return node_mass, missing_elements

def translate_model(
        keyword_path: str = None, 
        output_path: str = None,
        lb: List[float] = [-50, -50, 0],
        ub: List[float] = [50, 50, 20],
    ):
    import random
    dir_file = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(dir_file)

    keyword_path = keyword_path
    save_keyword_path = output_path
    cfile_path = os.path.join(root, "cfile", "translate.cfile")

    # Random translation (units = model units)
    x = random.uniform(lb[0], ub[0])
    y = random.uniform(lb[1], ub[1])
    z = random.uniform(lb[2], ub[2])
    # z = 0.0

    write_translate_cfile(
        output_path=cfile_path,
        open_keyword_path=keyword_path,
        save_keyword_path=save_keyword_path,
        part_id=2,
        xyz=(x, y, z),
    )

    run_cfile_ls_prepost(cfile_path)

    print(f"Translated part 2 by (x={x:.2f}, y={y:.2f}, z={z:.2f})")

def write_get_nodes_on_surface_cfile(keyword_path: str, output_path: str, part_id: int):
    lines = [
        "bgstyle plain",
        f'openc keyword "{keyword_path}"',
        "ident select 1",
        "genselect target node",
        "genselect 3dsurf on",
        f"genselect node add part {part_id}",
        "ident select 0",
        "exit",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def get_nodes_on_surface(keyword_path: str, part_id: int) -> List[int]:
    dir_file = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(dir_file)
    cfile_path = os.path.join(root, "cfile", "get_nodes_on_surface.cfile")
    msg_path = os.path.join(root, "cfile", "lspost.msg")

    write_get_nodes_on_surface_cfile(
        keyword_path=keyword_path,
        output_path=cfile_path,
        part_id=part_id,
    )

    run_cfile_ls_prepost(cfile_path)

    node_ids = []
    msg_path = Path(msg_path)
    if not msg_path.is_file():
        raise FileNotFoundError(f"LS-PrePost message file not found: {msg_path}")

    with msg_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = _NODE_ID_RE.search(line)
            if match:
                node_ids.append(int(match.group(1)))

    print(f"Nodes on surface of part {part_id}: {len(node_ids)}")

    return node_ids

def write_get_spc_cfile(keyword_path: str, output_path: str, spc_set_id: int):
    """
    Create an LS-PrePost command file that selects all SPC nodes
    """
    lines = [
        "bgstyle plain",
        f'openc keyword "{keyword_path}"',
        f'ident select {spc_set_id}',
        "genselect target node",
        f"genselect node add setnode {spc_set_id}",
        "ident select 0",
        "exit"
    ]
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def get_spc_nodes(keyword_path: str, spc_set_id: int) -> List[int]:
    """
    Get all SPC nodes on the given set ID, returned as zero-based node IDs.
    """
    dir_file = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(dir_file)
    cfile_path = os.path.join(root, "cfile", "get_spc_nodes.cfile")
    msg_path = os.path.join(root, "cfile", "lspost.msg")

    # Write cfile template
    write_get_spc_cfile(
        keyword_path=keyword_path,
        output_path=cfile_path,
        spc_set_id=spc_set_id
    )

    # Run LS-prepost with cfile
    run_cfile_ls_prepost(cfile_path)

    # Get node ids with zero-based index
    node_ids = []
    msg_path = Path(msg_path)
    if not msg_path.is_file():
        raise FileNotFoundError(f"LS-PrePost message file not found: {msg_path}")

    with msg_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = _NODE_ID_RE.search(line)
            if match:
                node_ids.append(int(match.group(1)) - 1) # zero-based index

    print(f"SPC nodes in set {spc_set_id}: {len(node_ids)}")

    return node_ids

if __name__ == "__main__":
    server = dpf.start_local_server(ansys_path=r"C:\Program Files\ANSYS Inc\v242", as_global=True)
    # translate_model()
    nodes_id = get_nodes_on_surface(
        keyword_path=r"D:\Projects_code\ball_plate\output\ball_plate.k",
        part_id=2,
    )
    print(len(nodes_id))
