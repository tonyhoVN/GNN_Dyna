from ansys.dyna.core import Deck
from ansys.dyna.core import keywords as kwd
from ansys.dyna.core.pre import examples
import lsdyna_mesh_reader as mesh
import os
import numpy as np

# -------------------------------------------------------------------------
# 1. Load an existing mesh / keyword input
# -------------------------------------------------------------------------
path = examples.ball_plate + os.sep
Input_LatticeFile = path + "ball_plate.k"
deck = Deck(Input_LatticeFile)
deck.import_file(Input_LatticeFile)   # read existing keyword file

# -------------------------------------------------------------------------
# 2. Define Material keywords
# -------------------------------------------------------------------------

# MAT_024: Piecewise Linear Plasticity (MID=1)
mat1 = kwd.MatPiecewiseLinearPlasticity(mid=1)
mat1.ro = 1.25e-9      # density
mat1.e = 1.0           # Young’s modulus
mat1.pr = 0.35         # Poisson’s ratio
mat1.sigy = 0.05       # yield stress
mat1.etan = 0.2        # tangent modulus
deck.extend([mat1])

# MAT_RIGID example (MID=2)
mat2 = kwd.MatRigid(mid=2)
mat2.ro = 7.83e-6
mat2.e = 207.0
mat2.pr = 0.3
deck.extend([mat2])

# -------------------------------------------------------------------------
# 3. Define Sections and Parts
# -------------------------------------------------------------------------
shell_section = kwd.SectionShell(secid=1)
shell_section.elform = 2  # Belytschko-Tsay
shell_section.shrf = 5/6
shell_section.nip = 5
shell_section.t1 = 1.0
shell_section.t2 = 1.0
shell_section.t3 = 1.0
shell_section.t4 = 1.0
deck.extend([shell_section])

solid_section = kwd.SectionSolid(secid=2)
solid_section.elform = 1  # 8-pt hexahedron
deck.extend([solid_section])

# PART 1 (plate)
if deck.get_kwds_by_type('PART'):
    print("PART keywords already exist in the deck. They will be updated.")
# 1. Delete existing PARTs
part_kwds = deck.get(type="PART")
print(len(part_kwds))

# 3. Add new PARTs
import pandas as pd

parts1_df = pd.DataFrame([
    {"heading": "Plate", "pid": 1, "secid": shell_section.secid, "mid": mat1.mid,
     "eosid": 0, "hgid": 0, "grav": 0, "adpopt": 0, "tmid": 0},
])

parts2_df = pd.DataFrame([
    {"heading": "Ball",  "pid": 2, "secid": solid_section.secid, "mid": mat2.mid,
     "eosid": 0, "hgid": 0, "grav": 0, "adpopt": 0, "tmid": 0},
])

part_kwds[0].parts = parts1_df
part_kwds[1].parts = parts2_df

# 4. Verify
print([deck.get(type="PART")])  # [1, 2]

# -------------------------------------------------------------------------
# 4. Define Contact
# -------------------------------------------------------------------------
contact = kwd.ContactAutomaticSurfaceToSurface()
contact.cid = 1
contact.surfa = 1  # slave surface ID
contact.surfb = 2  # master surface ID 
contact.surfatyp = 3
contact.surfbtyp = 3

deck.extend([contact])

# -------------------------------------------------------------------------
# 5. Boundary conditions
# -------------------------------------------------------------------------
spc_nodes = [34,35,51,52,68,69,85,86,102,103,119,120,136,137,153,154,
             170,171,187,188,204,205,221,222,238,239,255,256] + list(range(1,19)) + list(range(272,290))

spc_set = kwd.SetNodeList(sid = 1, nodes = spc_nodes)
deck.extend([spc_set])

spc = kwd.BoundarySpcSet(nsid=1, dofx=1, dofy=1, dofz=1, dofrx=0, dofry=0, dofrz=0)
deck.extend([spc])

# -------------------------------------------------------------------------
# 6. Initial velocity
# -------------------------------------------------------------------------
for node_id in range(1, 1652):
    vel = kwd.InitialVelocityNode(nid=node_id, vx=0.0, vy=0.0, vz=-10.0)
    deck.extend([vel])

# -------------------------------------------------------------------------
# 7. Database output controls
# -------------------------------------------------------------------------
# ASCII databases (BINARY=2 for binout)
db_glstat = kwd.DatabaseGlstat(dt=0.5, binary=2)
db_matsum = kwd.DatabaseMatsum(dt=0.5, binary=2)
db_nodout = kwd.DatabaseNodout(dt=0.5, binary=2)
db_nodfor = kwd.DatabaseNodfor(dt=0.5, binary=2)
db_ncforc = kwd.DatabaseNcforc(dt=0.5, binary=2)
db_sleout = kwd.DatabaseSleout(dt=0.5, binary=2)
db_elout  = kwd.DatabaseElout(dt=0.5, binary=2)

# Add them all to the deck
deck.extend([
    db_glstat,
    db_matsum,
    db_nodout,
    db_nodfor,
    db_ncforc,
    db_sleout,
    db_elout,
])

# Binary (d3plot etc.)
db_bin = kwd.DatabaseBinaryD3Plot(dt=0.5)
deck.extend([db_bin])

# -------------------------------------------------------------------------
# 8. Control cards (termination time)
# -------------------------------------------------------------------------
ctrl_term = kwd.ControlTermination()
ctrl_term.endtim = 10.0
deck.extend([ctrl_term])

# -------------------------------------------------------------------------
# 9. Save deck
# -------------------------------------------------------------------------
downloadpath = os.path.join(os.getcwd(), "output")
downloadfile = os.path.join(downloadpath, "ball_plate_new.k")
deck_string = deck.write()

with open(downloadfile, 'w') as f:
    f.write(deck_string)
    print(f"Keyword deck saved to {downloadfile}")