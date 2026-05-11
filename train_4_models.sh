#!/bin/bash
set -e  # stop if any command fails

echo "Training Model 1: MeshGraphNet NeMo"
python3 -m mesh_graph_net_nemo.train_mesh_graph_net --config mesh_graph_net_nemo/mesh_graph_net.json

echo "Training Model 2: Baseline"
python3 -m baseline.train_baseline --config base_line/base_line.json

echo "Training Model 3: Recurrent 1"
python3 -m train_recurrent --config config/gnn_contact_residual_recurrent_1.json

echo "Training Model 4: Recurrent 5"
python3 -m train_recurrent --config config/gnn_contact_residual_recurrent.json

echo "All training finished successfully."

############ Validate

# echo "Validate Model 1: MeshGraphNet + 5 His"
# python -m mesh_graph_net_nemo.validate_mesh_graph_net --config mesh_graph_net_nemo/mesh_graph_net.json --pt-file save_model/hood/mesh_graph_net_20260509_003253_hood.pt --rollout-steps 200

# echo "Validate Model 2: GraphSAGE"
# python -m validate_model --config base_line/base_line.json --pt-file save_model/hood/gnn_baseline_20260510_052058.pt --rollout-steps 200

# echo "Validate Model 3: Recurrent Residual 1P"
# python -m validate_model --config config/gnn_contact_residual_recurrent_1.json --pt-file save_model/hood/recurrent_20260510_052230_h5_p1.pt --rollout-steps 200

# echo "Validate Model 4: Recurrent Residual 5P"
# python -m validate_model --config config/gnn_contact_residual_recurrent_1.json --pt-file save_model/hood/recurrent_20260510_140336_h5_p5.pt --rollout-steps 200