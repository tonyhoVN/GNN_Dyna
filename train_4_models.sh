#!/bin/bash
set -e  # stop if any command fails

echo "Training Model 1: MeshGraphNet NeMo"
python -m mesh_graph_net_nemo.train_mesh_graph_net --config mesh_graph_net_nemo/mesh_graph_net.json

echo "Training Model 2: Baseline"
python -m train_baseline --config base_line/base_line.json

echo "Training Model 3: Recurrent 1"
python -m train_recurrent --config config/gnn_contact_residual_recurrent_1.json

echo "Training Model 4: Recurrent 5"
python -m train_recurrent --config config/gnn_contact_residual_recurrent.json

echo "All training finished successfully."