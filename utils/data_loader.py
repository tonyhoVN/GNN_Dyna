import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional


# Custom dataset class for loading FEM simulation data

@dataclass
class GraphData:
    """
    Generic graph container for GNNs.
    """
    x: torch.Tensor                 # Node kinetics features (N, F*T)
    y: torch.Tensor                 # Labels / targets (N, F)
    node_mass: torch.Tensor         # Node mass (N, )
    edge_index: torch.Tensor        # Edge list (2, E)
    edge_attr: Optional[torch.Tensor] = None  # Edge features (E, D)
    delta_t: Optional[torch.Tensor] = None    # Time step information
    element_to_nodes: Optional[dict] = None  # {element_id: [node_ids, ]}
    element_materials: Optional[dict] = None # {element_id: material_id}
    total_internal_energy: Optional[torch.Tensor] = None
    total_kinetic_energy: Optional[torch.Tensor] = None

    def to(self, device):
        element_to_nodes = self.element_to_nodes
        if element_to_nodes is not None:
            if isinstance(element_to_nodes, dict):
                element_to_nodes = {
                    eid: torch.tensor(node_ids, dtype=torch.long, device=device)
                    for eid, node_ids in element_to_nodes.items()
                }
            else:
                element_to_nodes = torch.as_tensor(element_to_nodes, dtype=torch.long, device=device)

        element_materials = self.element_materials
        if element_materials is not None:
            if isinstance(element_materials, dict):
                element_materials = {
                    eid: torch.tensor(mat, dtype=torch.float, device=device)
                    for eid, mat in element_materials.items()
                }
            else:
                element_materials = torch.as_tensor(element_materials, dtype=torch.float, device=device)

        return GraphData(
            x=self.x.to(device),
            y=self.y.to(device),
            node_mass=self.node_mass.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device) if self.edge_attr is not None else None,
            delta_t=self.delta_t.to(device) if self.delta_t is not None else None,
            element_to_nodes=element_to_nodes,
            element_materials=element_materials,
            total_internal_energy=self.total_internal_energy.to(device) if self.total_internal_energy is not None else None,
            total_kinetic_energy=self.total_kinetic_energy.to(device) if self.total_kinetic_energy is not None else None,
        )
    
class FEMDataset(Dataset):
    def __init__(self, data_path):
        """
        X_list: list of input feature tensors for each time step 
                list[(Nodes, features, consecutive_time_frames)]
        Y_list: list of target feature tensors for each time step
                list[(Nodes, target_features,)]

        delta_t: time interval between consecutive frames
        """
        super().__init__()
        data = np.load(data_path, allow_pickle=True)
        self.X_list = data['X_list'] # list[(N, F, T)] size = time_steps
        self.Y_list = data['Y_list'] # list[(N, F,)] size = time_steps
        self.Delta_t = data['Delta_t'] # N
        edge_index = torch.as_tensor(data['edge_index'], dtype=torch.long)
        if edge_index.ndim == 2 and edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
            edge_index = edge_index.t().contiguous()
        self.edge_index = edge_index  # (2, E)
        self.edge_attr = None
        if 'edge_attr' in data:
            self.edge_attr = torch.as_tensor(data['edge_attr'], dtype=torch.float) # (E, D)

        self.element_to_nodes = None
        element_id_order = []
        if 'element_to_nodes' in data:
            element_to_nodes = data['element_to_nodes']
            if hasattr(element_to_nodes, "item"):
                element_to_nodes = element_to_nodes.item()
            self.element_to_nodes = element_to_nodes
        elif 'element_id_solids' in data and 'element_to_nodes_solids' in data:
            element_to_nodes = {}
            for eid, nodes in zip(data['element_id_solids'], data['element_to_nodes_solids']):
                element_to_nodes[int(eid)] = [int(nid) for nid in nodes]
                element_id_order.append(int(eid))
            for eid, nodes in zip(data['element_id_shells'], data['element_to_nodes_shells']):
                element_to_nodes[int(eid)] = [int(nid) for nid in nodes]
                element_id_order.append(int(eid))
            self.element_to_nodes = element_to_nodes

        if self.element_to_nodes is not None:
            self.element_to_nodes = dict(sorted(self.element_to_nodes.items(), key=lambda kv: kv[0]))

        self.element_materials = None
        if 'element_materials' in data:
            materials = data['element_materials']
            if isinstance(materials, np.ndarray) and element_id_order:
                if materials.shape[0] == len(element_id_order):
                    self.element_materials = {
                        int(eid): int(materials[i])
                        for i, eid in enumerate(element_id_order)
                    }
                else:
                    self.element_materials = None
            else:
                self.element_materials = materials
        self.node_mass = torch.as_tensor(data['node_mass'], dtype=torch.float)   # (N, )
        self.total_internal_energy = data['total_internal_energy'] if 'total_internal_energy' in data else None
        self.total_kinetic_energy = data['total_kinetic_energy'] if 'total_kinetic_energy' in data else None
        # breakpoint()

    def __len__(self):
        return len(self.X_list)
    
    def __getitem__(self, idx):
        current_state = torch.as_tensor(self.X_list[idx], dtype=torch.float)  # (N, F, T)
        predict_feature = torch.as_tensor(self.Y_list[idx], dtype=torch.float)    # (N, F,)
        delta_t = torch.as_tensor(self.Delta_t[idx], dtype=torch.float)  # scalar
        total_internal_energy = None
        if self.total_internal_energy is not None:
            total_internal_energy = torch.as_tensor(self.total_internal_energy[idx], dtype=torch.float)
        total_kinetic_energy = None
        if self.total_kinetic_energy is not None:
            total_kinetic_energy = torch.as_tensor(self.total_kinetic_energy[idx], dtype=torch.float)
        return GraphData(
            x=current_state,
            y=predict_feature,
            node_mass=self.node_mass,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            delta_t=delta_t,
            element_to_nodes=self.element_to_nodes,
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
    data_path = os.path.join(root, "data", "ball_plate_gnn_data.npz")
    dataset = FEMDataset(data_path)

    def collate_graphs(batch):
        return batch
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_graphs)

    i = 1
    for batch in loader:
        for g in batch:
            print(g.x.shape, g.y.shape)
            print(f'node mass shape: {g.node_mass.shape}')
            y_new = torch.cat([g.y, g.node_mass.unsqueeze(-1)], dim=-1)
            print(f'new y shape with node mass: {y_new.shape}')
            y_gpu = g.y.to('cuda')
            print(f'y on GPU shape: {y_gpu.shape}')
            print(f'y on GPU device: {y_gpu[0, 0].device}')
            print(g.edge_attr.shape)
            
        break
if __name__ == "__main__":
    load_data_example()
