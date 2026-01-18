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

Typical workflow
----------------
1. Generate the input deck:
   `python ball_plate.py`
2. Run the solver:
   `python run_simulation.py --solver "path\to\ls-dyna.exe"`

Notes
-----
- The pre-service must be running before `ball_plate.py` is executed.
- Update the LS-DYNA solver path if you do not use the default student install.
