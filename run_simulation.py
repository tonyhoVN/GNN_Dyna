import os
from utils.ansys_utils import run_dyna
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LS-DYNA simulation with configurable inputs"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="ball_plate",
        help="Path to input keyword file (e.g. ball_plate)"
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

    # Default input path if --input not provided
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "output", args.input + ".k")

    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    INPUT = input_file
    SOLVER = args.solver
    NCPU = args.ncpu
    MEMORY = args.memory

    print("INPUT :", INPUT)
    print("SOLVER:", SOLVER)
    print("NCPU  :", NCPU)
    print("MEMORY:", MEMORY)

    # Run the simulation
    run_dyna(ls_solver=SOLVER, k_file=INPUT, ncpu=NCPU, memory=MEMORY)