import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Custom dataset class for loading FEM simulation data
class FEMDataset(Dataset):
    def __init__(self, X_list, Y_list, Delta_t):
        """
        X_list: list of input feature tensors for each time step 
                list[(Nodes, features, consecutive_time_frames)]
        Y_list: list of target feature tensors for each time step
                list[(Nodes, target_features,)]

        delta_t: time interval between consecutive frames
        """
        super().__init__()
        self.X_list = X_list # list[(N, F, T)] size = time_steps
        self.Y_list = Y_list # list[(N, F,)] size = time_steps
        self.Delta_t = Delta_t # N

    def __len__(self):
        return len(self.X_list)
    
    def __getitem__(self, idx):
        current_state = self.X_list[idx]  # (N, F, T)
        predict_feature = self.Y_list[idx]    # (N, F,)
        delta_t = self.Delta_t[idx]  # scalar
        return current_state, predict_feature, delta_t


if __name__ == "__main__":
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