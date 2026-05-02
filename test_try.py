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
from model.model_creation import ModelConfig, create_gnn_model
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import qmc


random.seed(42)
# x = torch.randn((2,100,10,3))
# y = torch.randn((2,1))
# edge = torch.randint(0, 100, (2, 2, 500))
# edge_surf = torch.randint(0, 100, (2, 2, 400))
# d_t = torch.randn((2,15))

# model = torch.nn.LSTM(10, 150, 5, batch_first=True)

# # data_list = []
# # for i in range(2):
# #     data = Data(
# #         x=x[i],                  # (100, 3)
# #         edge_index=edge[i],      # (2, 500)
# #         edge_surf_index=edge_surf[i],
# #         y=y[i],                   # (1,) or scalar
# #         dt=d_t[i]
# #     )
# #     data_list.append(data)

# # dataset = ListDataset(data_list)

# import os
# root = os.getcwd()
# data_path = os.path.join(root, "data", "20260122_094004_time_data.npz")
# dataset = FEMDataset(data_path)
# datalist = []

# for i in range(len(dataset)):
#     datalist.append(dataset[i])
    
# model = EncodeDecodeGNN(node_dim=9, edge_dim=2, out_dim=9, latent_dim=24)
# loader = DataLoader(dataset, batch_size=5, shuffle=True)

# for batch in loader:
#     print("ok")
#     print(batch)
#     batch.delta_t = batch.delta_t[batch.batch]

#     out = model(batch)
    
#     # out, (h_n, c_n) = model(batch.x.permute(0, 2, 1).contiguous())
#     # print(h_n[-1].shape)
#     model.loss(out, batch.y)

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


if __name__ == "__main__":
    import os
    # root = os.getcwd()
    # config_path = os.path.join(root, "config", "gnn_contact_general.json")

    # model_config = ModelConfig.from_json(config_path)
    # model = create_gnn_model(model_config)
    # print(model)
    # num_params = sum(p.numel() for p in model.parameters())
    # num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Model parameters: {num_params} (trainable: {num_trainable})")
    # parent_dir = os.path.dirname(os.path.abspath(__file__))
    # pass

    # import pyvista as pv
    # pl = pv.Plotter()
    # _ = pl.add_mesh(pv.Sphere(center=(2, 0, 0)), color='r')
    # _ = pl.add_mesh(pv.Sphere(center=(0, 2, 0)), color='g')
    # _ = pl.add_mesh(pv.Sphere(center=(0, 0, 2)), color='b')
    # _ = pl.add_axes_at_origin()
    # pl.show()

    # horizon = 8

    # t = torch.arange(horizon)

    # tau = 0.4*horizon  # more reasonable scale for H=5
    # weights = torch.exp(-t / tau)
    # weights = weights / weights.sum()
    # print(weights)

    # XY = []
    # for _ in range(50):
    #     x = random.uniform(-250, 250)
    #     y = random.uniform(-450, 450)
    #     XY.append((x, y))
    # XY = np.array(XY)
    
    # # Số điểm
    # n_samples = 30

    # # Khởi tạo LHS cho 2 chiều (x, y)
    # sampler = qmc.LatinHypercube(d=2)

    # # Sample trong [0,1]
    # sample = sampler.random(n=n_samples)

    # # Scale về khoảng mong muốn
    # l_bounds = [-250, -450]
    # u_bounds = [250, 450]

    # scaled_sample = qmc.scale(sample, l_bounds, u_bounds)

    # # Tách x, y
    # x = scaled_sample[:, 0]
    # y = scaled_sample[:, 1]

    # plt.scatter(x, y)
    # plt.show()

    horizon = 5
    t = np.linspace(0, horizon + 1, num=300)
    percent = np.linspace(0, 1, num=5)
    alpha_start = 20
    alpha_end = 2
    alphas = alpha_start - (alpha_start - alpha_end) * (percent / 0.8)

    blues = ['#1a5fa8', '#3e8ecb', '#6aabdc', '#95c7e8', '#b8d9ee']
    dashes = ['-', '--', '-.', (0,(5,2)), (0,(2,2))]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, alpha in enumerate(alphas):
        weights = np.exp(-alpha * (t / horizon))
        ax.plot(t, weights, color=blues[i], linestyle=dashes[i],
                linewidth=2.0 if i == 0 else 1.6,
                label=f'α = {alpha:.1f}')

    ax.set_xlabel('Time step $t$', fontsize=13)
    ax.set_ylabel('Weight $w(t)$', fontsize=13)
    ax.set_title('Adaptive Exponential Decay Weights', fontsize=14, fontweight='normal', pad=12)

    ax.set_xlim(0, horizon + 1)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.tick_params(labelsize=11)

    ax.grid(True, color='#e0e0e0', linewidth=0.7, linestyle='-')
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    ax.legend(fontsize=11, frameon=True, framealpha=0.9,
            edgecolor='#cccccc', loc='upper right')

    plt.tight_layout()
    plt.savefig('images/decay_weights.png', dpi=600, bbox_inches='tight')
    plt.show()