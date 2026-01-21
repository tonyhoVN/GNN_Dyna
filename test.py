from ansys.dpf import core as dpf
from ansys.dpf.post import DataFrame
from ansys.dpf.core import examples

import numpy as np
import os


server = dpf.start_local_server(ansys_path=r"C:\Program Files\ANSYS Inc\v242", as_global=True)
ds = dpf.DataSources()
binout_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "binout")
print(binout_path)
ds.set_result_file_path(binout_path, "binout")
model = dpf.Model(ds)

# binout = examples.download_binout_matsum()
# ds = dpf.DataSources()
# ds.set_result_file_path(binout, "binout")
# model = dpf.Model(ds)

print(model.results.global_sliding_interface_energy.eval()[0])