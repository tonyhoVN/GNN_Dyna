from utils.ls_prepost_utils import translate_model
from utils.ansys_utils import run_dyna
import os 
import datetime
import argparse
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LS-DYNA simulation with configurable inputs"
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of runs to perform"
    )

    parser.add_argument(
        "--solver",
        type=str,
        default=r"C:\Program Files\LS-DYNA Suite R16.1 Student\lsdyna\ls-dyna_smp_d_R16.1_180-gd50332dbe5_winx64_ifort190_sse2_studentversion.exe",
        help="Path to LS-DYNA solver executable"
    )

    parser.add_argument(
        "--ncpu",
        type=int,
        default=4,
        help="Number of CPUs"
    )

    parser.add_argument(
        "--memory",
        type=int,
        default=20,
        help="Memory in MB"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    keyword_path = os.path.join(root, "output", "ball_plate.k")

    solver = args.solver
    ncpu = args.ncpu
    memory = args.memory
    for i in range(args.num_runs):
        # Create k model with random velocity 
        subprocess.run([sys.executable, "-m", "ball_plate"], check=True, cwd=root)

        # Translate model and run simulation
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(root, "output", f"{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        modify_keyword_path = os.path.join(run_dir, "ball_plate.k")
        translate_model(keyword_path=keyword_path, output_path=modify_keyword_path)

        # run simulation
        run_dyna(ls_solver=solver, k_file=modify_keyword_path, ncpu=ncpu, memory=memory)
