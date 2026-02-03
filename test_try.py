# from ansys.dpf import core as dpf
# from ansys.dpf.post import DataFrame
# from ansys.dpf.core import examples
# import torch.nn as nn
import torch

import numpy as np
import os

from torch_geometric.data import Data
# from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from utils.data_loader import GraphData, FEMDataset
from model.GNN import EncodeDecodeGNN


# x = torch.randn((2,100,10,3))
# y = torch.randn((2,1))
# edge = torch.randint(0, 100, (2, 2, 500))
# edge_surf = torch.randint(0, 100, (2, 2, 400))
# d_t = torch.randn((2,15))

model = torch.nn.LSTM(10, 150, 5, batch_first=True)

# data_list = []
# for i in range(2):
#     data = Data(
#         x=x[i],                  # (100, 3)
#         edge_index=edge[i],      # (2, 500)
#         edge_surf_index=edge_surf[i],
#         y=y[i],                   # (1,) or scalar
#         dt=d_t[i]
#     )
#     data_list.append(data)

# dataset = ListDataset(data_list)

import os
root = os.getcwd()
data_path = os.path.join(root, "data", "20260122_094004_time_data.npz")
dataset = FEMDataset(data_path)
datalist = []

for i in range(len(dataset)):
    datalist.append(dataset[i])
    
model = EncodeDecodeGNN(node_dim=9, edge_dim=2, out_dim=9, latent_dim=24)
loader = DataLoader(dataset, batch_size=5, shuffle=True)

for batch in loader:
    print("ok")
    print(batch)
    batch.delta_t = batch.delta_t[batch.batch]

    out = model(batch)
    
    # out, (h_n, c_n) = model(batch.x.permute(0, 2, 1).contiguous())
    # print(h_n[-1].shape)
    model.loss(out, batch.y)

# server = dpf.start_local_server(ansys_path=r"C:\Program Files\ANSYS Inc\v242", as_global=True)
# ds = dpf.DataSources()
# binout_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "binout")
# print(binout_path)
# ds.set_result_file_path(binout_path, "binout")
# model = dpf.Model(ds)

# # binout = examples.download_binout_matsum()
# # ds = dpf.DataSources()
# # ds.set_result_file_path(binout, "binout")
# # model = dpf.Model(ds)

# print(model.results.global_sliding_interface_energy.eval()[0])

# def func1(x):
#     a = x.detach().cpu().numpy()
#     return a
# a = nn.Embedding(2, 16)
# x = torch.tensor([0,1]).to("cuda")
# b = func1(x)
# print(x.device, b.device)