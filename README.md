GNN Dyna for Ball and Plate
==========
This project builds and runs an LS-DYNA "ball plate" simulation using the
PyDYNA pre-service, and includes early work on a graph neural network model
for learning from the simulation results.

What is here
------------
- `ball_plate.py`: Creates the LS-DYNA input deck using `ansys.dyna.core.pre`
  and writes `output/ball_plate.k`.
- `run_simulation.py`: Runs LS-DYNA against the generated input deck.
- `model/GNN.py`: Prototype GNN model for mesh/contact learning.
- Notebooks: Experimentation and analysis (`*.ipynb`).

Generate data manual
----------------
1. Generate the input template deck:
   `python -m ball_plate`

2. Run the solver simulation to obtain data:
   `python -m run_simulation --solver "path\to\ls-dyna.exe" --input name_k_file`

Generate data automate 
----------------
1. Run the solver to obtain data 
   `python -m create_dataset --num-runs 10 --keyword-file ball_plate.k`

2. Post-process and save dataset for training
   `python -m data_process --keyword-file ball_plate.k`


Training 
----------------
Load data and train recurrent model
   `python train_recurrent.py --config config/gnn_contact_direct_recurrent.json`
   `python train_force.py --config config/contact_gnn_force.json`

For standard baseline residual gnn
   `python train_baseline.py --config config/gnn_contact_general.json`

Test
------
`python test_meshgraphnet_prediction.py --model_path model\meshgraphnet_20260122_103348.pt --npz_path data\20260122_094307_gnn_data.npz --sample_index 30`


Rollout Animation Test
------
- Test the training model 
   `python animate_rollout_prediction.py  --config config\gnn_contact_direct_no_acc.json --model-path save_model\gnn_general_20260303_120856.pt --npz-path data\20260302_133424_time_data.npz --interval-ms 200 --start-index 20 --rollout-steps 50`

- Bench mark with base-line and 1 step prediction model 
   `python benchmark.py --benchmark-config benchmark\rollout_benchmark.json --plate-node-count 289`



Notes
-----
- The pre-service must be running before `ball_plate.py` is executed.
- Update the LS-DYNA solver path if you do not use the default student install.


### TODO ###
- Remesh plate
- Benchmark baseline GNN with simple edge feature