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

2. Pre-process for data agumented:
   `python -m `

3. Run the solver simulation to obtain data:
   `python -m run_simulation --solver "path\to\ls-dyna.exe" --input name_k_file`

Generate data automate 
----------------
`python create_dataset.py --num_runs 2`

4. Post-process and save dataset for training
   `python -m data_process`

5. Load data and train
   `python train_force.py --config config/contact_gnn_force.json`

Test
------
`python test_meshgraphnet_prediction.py --model_path model\meshgraphnet_20260122_103348.pt --npz_path data\20260122_094307_gnn_data.npz --sample_index 30`

Notes
-----
- The pre-service must be running before `ball_plate.py` is executed.
- Update the LS-DYNA solver path if you do not use the default student install.
