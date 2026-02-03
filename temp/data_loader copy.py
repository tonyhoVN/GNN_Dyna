import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional
from torch_geometric.data import Data

# Custom dataset class for loading FEM simulation data

@dataclass
class GraphData:
    """
    Generic graph container for GNNs.
    """
    x: torch.Tensor                 # Node kinetics features (N, F*T)
    y: torch.Tensor                 # Labels / targets (N, F)
    x_initial: torch.Tensor         # Initial node coordinates (N, 3)
    node_mass: torch.Tensor         # Node mass (N, )
    delta_t: torch.Tensor           # Time step information (scalar)
    edge_index: torch.Tensor        # Edge list (2, E)
    edge_attr: Optional[torch.Tensor] = None  # Edge features (E, D)
    edge_surf_index: Optional[torch.Tensor] = None
    # element_to_nodes: Optional[dict] = None  # {element_id: [node_ids, ]}
    # element_materials: Optional[dict] = None # {element_id: material_id}
    element_node_ids: Optional[torch.Tensor] = None  # (El, K) padded with -1
    element_node_mask: Optional[torch.Tensor] = None  # (El, K) bool
    element_materials: Optional[torch.Tensor] = None  # (El,)
    total_internal_energy: Optional[torch.Tensor] = None
    total_kinetic_energy: Optional[torch.Tensor] = None

    def to(self, device):
        # element_to_nodes = self.element_to_nodes
        # if element_to_nodes is not None:
        #     if isinstance(element_to_nodes, dict):
        #         element_to_nodes = {
        #             eid: torch.tensor(node_ids, dtype=torch.long, device=device)
        #             for eid, node_ids in element_to_nodes.items()
        #         }
        #     else:
        #         element_to_nodes = torch.as_tensor(element_to_nodes, dtype=torch.long, device=device)

        # element_materials = self.element_materials
        # if element_materials is not None:
        #     if isinstance(element_materials, dict):
        #         element_materials = {
        #             eid: torch.tensor(mat, dtype=torch.float, device=device)
        #             for eid, mat in element_materials.items()
        #         }
        #     else:
        #         element_materials = torch.as_tensor(element_materials, dtype=torch.float, device=device)

        return GraphData(
            x=self.x.to(device),
            y=self.y.to(device),
            x_initial=self.x_initial.to(device),
            node_mass=self.node_mass.to(device),
            delta_t=self.delta_t.to(device) if self.delta_t is not None else None,
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device) if self.edge_attr is not None else None,
            edge_surf_index=self.edge_surf_index.to(device) if self.edge_attr is not None else None,
            # element_to_nodes=element_to_nodes,
            # element_materials=element_materials,
            element_node_ids=self.element_node_ids.to(device) if self.element_node_ids is not None else None,
            element_node_mask=self.element_node_mask.to(device) if self.element_node_mask is not None else None,
            element_materials=self.element_materials.to(device) if self.element_material_ids is not None else None,
            total_internal_energy=self.total_internal_energy.to(device) if self.total_internal_energy is not None else None,
            total_kinetic_energy=self.total_kinetic_energy.to(device) if self.total_kinetic_energy is not None else None,
        )
    
class FEMDataset(Dataset):
    def __init__(self, data_path, geometry_path: str = None):
        """
        X_list: list of input feature tensors for each time step 
                list[(Nodes, features, consecutive_time_frames)]
        Y_list: list of target feature tensors for each time step
                list[(Nodes, target_features,)]

        delta_t: time interval between consecutive frames
        """
        super().__init__()
        
        # Load time series and geometry data
        kinematic_data = np.load(data_path, allow_pickle=True)
        if geometry_path is None:
            if "geometry_path" in kinematic_data:
                geometry_path = str(kinematic_data["geometry_path"])
            else:
                geometry_path = os.path.join(os.path.dirname(data_path), "geometry_shared.npz")
        if geometry_path and os.path.isfile(geometry_path):
            geometry_data = np.load(geometry_path, allow_pickle=True)

        # Node features and its prediction
        self.X_list = kinematic_data['X_list'] # list[(N, F, T)] size = time_steps
        self.Y_list = kinematic_data['Y_list'] # list[(N, F,)] size = time_steps
        
        # Time intervals
        self.Delta_t = kinematic_data['Delta_t'] # N

        # Initial coordinates 
        self.X_initial = torch.as_tensor(geometry_data["initial_coords"], dtype=torch.float)  # (N, 3)

        # total target energies if available
        self.total_internal_energy = kinematic_data['total_internal_energy'] if 'total_internal_energy' in kinematic_data else None
        self.total_kinetic_energy = kinematic_data['total_kinetic_energy'] if 'total_kinetic_energy' in kinematic_data else None

        # Node mass 
        self.node_mass = torch.as_tensor(geometry_data["node_mass"], dtype=torch.float)   # (N, )

        # Load edge connectivity and attributes for GNN
        edge_index = torch.as_tensor(geometry_data["edge_index"], dtype=torch.long)
        if edge_index.ndim == 2 and edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
            edge_index = edge_index.t().contiguous()
        self.edge_index = edge_index  # (2, E)

        self.edge_attr = None
        if "edge_attr" in geometry_data:
            self.edge_attr = torch.as_tensor(geometry_data["edge_attr"], dtype=torch.float) # (E, D)

        # Load surface connectivity
        edge_surf_index = torch.as_tensor(geometry_data["edge_index"], dtype=torch.long) 
        if edge_surf_index.ndim == 2 and edge_surf_index.shape[0] != 2 and edge_surf_index.shape[1] == 2:
            edge_surf_index = edge_surf_index.t().contiguous()
        self.edge_surf_index = edge_surf_index # (2, E_s)

        # Load element to nodes mapping
        self.element_to_nodes = None
        element_id_order = []
        if ("element_id_solids" in geometry_data
            and "element_to_nodes_solids" in geometry_data):
            element_to_nodes = {}
            for eid, nodes in zip(geometry_data["element_id_solids"], geometry_data["element_to_nodes_solids"]):
                element_to_nodes[int(eid)] = [int(nid) for nid in nodes]
                element_id_order.append(int(eid))
            for eid, nodes in zip(geometry_data["element_id_shells"], geometry_data["element_to_nodes_shells"]):
                element_to_nodes[int(eid)] = [int(nid) for nid in nodes]
                element_id_order.append(int(eid))
            self.element_to_nodes = element_to_nodes

        if self.element_to_nodes is not None:
            self.element_to_nodes = dict(sorted(self.element_to_nodes.items(), key=lambda kv: kv[0]))

        # Load element to material mapping
        self.element_materials = geometry_data["element_materials"] # (E,)


        # Precompute padded element node ids + materials for fast PhysicsNet
        self.element_node_ids = None
        self.element_node_mask = None

        if self.element_to_nodes is not None:
            element_ids = list(self.element_to_nodes.keys())
            max_nodes = max(len(nodes) for nodes in self.element_to_nodes.values()) if element_ids else 0
            if max_nodes > 0:
                elem_nodes = np.full((len(element_ids), max_nodes), -1, dtype=np.int64)
                elem_mask = np.zeros((len(element_ids), max_nodes), dtype=np.bool_)
                for i, eid in enumerate(element_ids):
                    nodes = self.element_to_nodes[eid]
                    elem_nodes[i, :len(nodes)] = nodes
                    elem_mask[i, :len(nodes)] = True
                self.element_node_ids = torch.as_tensor(elem_nodes, dtype=torch.long)
                self.element_node_mask = torch.as_tensor(elem_mask, dtype=torch.bool)
        
        # breakpoint()

    def __len__(self):
        return len(self.X_list)
    
    def __getitem__(self, idx):
        current_state = torch.as_tensor(self.X_list[idx], dtype=torch.float)  # (N, F, T)
        predict_feature = torch.as_tensor(self.Y_list[idx], dtype=torch.float)    # (N, F,)
        delta_t = torch.as_tensor(self.Delta_t[idx], dtype=torch.float)  # scalar
        total_internal_energy = None
        if self.total_internal_energy is not None:
            total_internal_energy = torch.as_tensor(self.total_internal_energy[idx], dtype=torch.float) # (2, )
        total_kinetic_energy = None
        if self.total_kinetic_energy is not None:
            total_kinetic_energy = torch.as_tensor(self.total_kinetic_energy[idx], dtype=torch.float) # (2, )
        return GraphData(
            x=current_state,
            y=predict_feature,
            x_initial=self.X_initial,
            node_mass=self.node_mass,
            delta_t=delta_t,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            edge_surf_index=self.edge_surf_index,
            element_node_ids=self.element_node_ids,
            element_node_mask=self.element_node_mask,
            element_materials=self.element_materials,
            total_internal_energy=total_internal_energy,
            total_kinetic_energy=total_kinetic_energy
        )


def simple_test():
    # Test the dataset class
    X_dummy = [torch.randn(100, 6, 5) for _ in range(10)]  # 10 time steps, 100 nodes, 6 features, 5 consecutive frames
    Y_dummy = [torch.randn(100, 6) for _ in range(10)]  # 10 time steps, 100 nodes, 3 target features, 5 consecutive frames
    Delta_t_dummy  = [0.01 for _ in range(10)]  # constant time interval
    dataset = FEMDataset(X_dummy, Y_dummy, Delta_t_dummy)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch_idx, (X_batch, Y_batch, delta_t_batch) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  X_batch shape: {X_batch.shape}")  # Expected: (B, N, F, T)
        print(f"  Y_batch shape: {Y_batch.shape}")  # Expected: (B, N, F)
        print(f"  delta_t_batch shape: {delta_t_batch.shape}")  # Expected: (B, 1)



def load_data_example():
    import os
    root = os.getcwd()
    data_path = os.path.join(root, "data", "20260122_094004_gnn_data.npz")
    dataset = FEMDataset(data_path)

    def collate_graphs(batch):
        return batch
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_graphs)

    i = 1
    for batch in loader:
        for g in batch:
            print(g.x.shape, g.y.shape, g.x_initial.shape)
            print(f'node mass shape: {g.node_mass.shape}')
            y_new = torch.cat([g.y, g.node_mass.unsqueeze(-1)], dim=-1)
            print(f'new y shape with node mass: {y_new.shape}')
            y_gpu = g.y.to('cuda')
            print(f'y on GPU shape: {y_gpu.shape}')
            print(f'y on GPU device: {y_gpu[0, 0].device}')
            print(g.edge_attr.shape)
            print(g.edge_index.numel())
        break
if __name__ == "__main__":
    load_data_example()
