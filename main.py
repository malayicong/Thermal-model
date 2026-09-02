"""
EBM Thermal Modeling Tool — main entry point.

Usage:
    python main.py                          # uses inputs/config.yaml
    python main.py path/to/config.yaml
"""

import sys
import os
import time

from src.config_reader import load_config
from src.grid import Grid
from src.solver import Solver
from src.output import write_csv, write_vtk, write_t_history


def main(config_path: str = "inputs/config.yaml") -> None:

    t0 = time.perf_counter()
    sim = load_config(config_path)
    print(f"Config loaded in {time.perf_counter() - t0:.3f} s\n")

    os.makedirs(sim.output.directory, exist_ok=True)

    t1 = time.perf_counter()
    grid = Grid(sim)
    print(f"Grid built in {time.perf_counter() - t1:.3f} s\n")

    solver = Solver(sim, grid)
    solver.run()

    # --- Output ---
    print()
    if sim.param.mode == 'T_history':
        write_t_history(solver.t_history_data, sim)
    else:
        write_csv(grid, sim)
        if sim.output.vtk:
            write_vtk(grid, sim)

    print(f"\nTotal wall time: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "inputs/config.yaml"
    main(cfg)
