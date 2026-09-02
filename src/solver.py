"""
Solver: time-stepping loops for all three simulation modes.

  Solidification : marches from t=0 through scan + cooling, detecting
                   melt/solidification events and recording G, V, dT/dt
                   at each point's solidification front.

  Snapshots      : evaluates the full T field at prescribed time instants.

  T_history      : records temperature vs time at user-specified probe points.
"""

from __future__ import annotations
import time as _time
import numpy as np

from .data_structs import Simdat
from .grid import Grid
from .integration import Integrator, calc_temperature
from .boundary_conditions import apply_radiation_correction


class Solver:
    """
    Runs the thermal simulation for a fully-initialised Simdat + Grid.

    Usage
    -----
        solver = Solver(sim, grid)
        result = solver.run()        # returns mode-dependent data
    """

    def __init__(self, sim: Simdat, grid: Grid) -> None:
        self.sim        = sim
        self.grid       = grid
        self.integrator = Integrator(sim)

        # T_history results stored here after _run_t_history(); shape (steps, 1+N_probes)
        self.t_history_data: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self):
        mode = self.sim.param.mode
        if mode == 'Solidification':
            self._run_solidification()
        elif mode == 'Snapshots':
            self._run_snapshots()
        elif mode == 'T_history':
            self._run_t_history()
        else:
            raise ValueError(f"Unknown simulation mode: '{mode}'")

    # ------------------------------------------------------------------
    # Solidification mode
    # ------------------------------------------------------------------

    def _run_solidification(self) -> None:
        sim  = self.sim
        grid = self.grid
        mat  = sim.material
        dom  = sim.domain
        p    = sim.param
        bcs  = dom.bcs

        dt    = p.dt
        t_max = sim.scan_end_time * 3.0   # scan + 2x cooling phase
        n_steps = int(t_max / dt) + 1

        print(f"\nSolidification | dt={dt:.2e} s | t_max={t_max:.4f} s | ~{n_steps} steps")
        t_wall = _time.perf_counter()

        t    = 0.0
        step = 0

        while t <= t_max:
            nodes = self.integrator.build_nodes(t, is_solidification=True)

            # Temperature on active (non-masked) grid points
            T_active = calc_temperature(
                nodes, grid.xa, grid.ya, grid.za, mat.T_init
            )

            # Assemble full-domain T array (inactive points stay at T_init)
            T_new = np.full(dom.pnum, mat.T_init, dtype=np.float64)
            T_new[grid.active] = T_active

            # Radiation cooling at top surface
            apply_radiation_correction(
                T_new, grid.active, grid.z, dom.zmax, bcs,
                mat.rho, mat.cp, dom.zres, dt
            )

            # Remember which points were solidified before this step
            prev_finite = np.isfinite(grid.tSol)

            # Detect liquidus crossings, update tSol / dTdt / numMelt
            grid.update_solidification(T_new, t, dt, mat.T_liq)

            # For newly solidified points: compute G and then V = |dTdt| / G
            new_sol = np.isfinite(grid.tSol) & ~prev_finite
            if np.any(new_sol):
                self._compute_gradient(T_new, new_sol)
                G_vals  = grid.G[new_sol]
                safe_G  = np.where(G_vals > 0.0, G_vals, np.nan)
                grid.V[new_sol] = np.abs(grid.dTdt[new_sol]) / safe_G

            # Progress report
            if step % p.out_freq == 0:
                T_act = T_new[grid.active]
                n_sol  = int(np.isfinite(grid.tSol).sum())
                n_melt = int((T_act >= mat.T_liq).sum())
                elapsed = _time.perf_counter() - t_wall
                print(f"  t={t:.4f} s | T_max={T_act.max():.0f} K | "
                      f"melted={n_melt} | solidified={n_sol} | {elapsed:.1f}s")

            t    += dt
            step += 1

            # Early exit once scan is done and nothing is molten
            if (t > sim.scan_end_time
                    and not np.any(grid.T[grid.active] >= mat.T_liq)):
                print(f"  No molten points at t={t:.4f} s -- stopping early.")
                break

        elapsed = _time.perf_counter() - t_wall
        n_sol   = int(np.isfinite(grid.tSol).sum())
        print(f"\nDone. {n_sol} points solidified in {elapsed:.1f} s total.")

    # ------------------------------------------------------------------
    # Snapshots mode
    # ------------------------------------------------------------------

    def _run_snapshots(self) -> None:
        sim  = self.sim
        grid = self.grid
        mat  = sim.material
        dom  = sim.domain
        p    = sim.param
        bcs  = dom.bcs

        times = p.snapshot_times
        if not times:
            print("No snapshot times defined. "
                  "Add 'times: [...]' under mode_settings.snapshots.")
            return

        print(f"\nSnapshots | {len(times)} snapshot(s): "
              f"{[f'{t:.4f}s' for t in times]}")

        for t in times:
            t0    = _time.perf_counter()
            nodes = self.integrator.build_nodes(t, is_solidification=False)
            T_active = calc_temperature(
                nodes, grid.xa, grid.ya, grid.za, mat.T_init
            )

            T_new = np.full(dom.pnum, mat.T_init, dtype=np.float64)
            T_new[grid.active] = T_active

            # Radiation: no timestep for a static snapshot (dt=0 → no correction)
            apply_radiation_correction(
                T_new, grid.active, grid.z, dom.zmax, bcs,
                mat.rho, mat.cp, dom.zres, dt=0.0
            )

            np.copyto(grid.T, T_new)

            elapsed = _time.perf_counter() - t0
            print(f"  t={t:.4f} s | T_max={T_active.max():.0f} K | "
                  f"T_min={T_active.min():.0f} K | {elapsed:.2f} s")

    # ------------------------------------------------------------------
    # T_history mode
    # ------------------------------------------------------------------

    def _run_t_history(self) -> None:
        sim  = self.sim
        grid = self.grid
        mat  = sim.material
        dom  = sim.domain
        p    = sim.param
        bcs  = dom.bcs

        if not p.probe_points:
            print("No probe points defined. "
                  "Add 'probe_points: [[x,y,z], ...]' under mode_settings.t_history.")
            return

        probes = np.array(p.probe_points, dtype=np.float64)   # (N, 3)
        px, py, pz = probes[:, 0], probes[:, 1], probes[:, 2]
        n_probes   = len(probes)

        dt    = p.dt
        t_max = sim.scan_end_time * 3.0
        t     = 0.0

        print(f"\nT_history | {n_probes} probe(s) | dt={dt:.2e} s | t_max={t_max:.4f} s")
        t_wall = _time.perf_counter()

        rows = []
        while t <= t_max:
            nodes   = self.integrator.build_nodes(t, is_solidification=False)
            T_probes = calc_temperature(nodes, px, py, pz, mat.T_init)
            rows.append([t] + list(T_probes))
            t += dt

        self.t_history_data = np.array(rows)   # shape (n_steps, 1 + n_probes)

        elapsed = _time.perf_counter() - t_wall
        print(f"  Done: {len(rows)} timesteps recorded in {elapsed:.1f} s.")

    # ------------------------------------------------------------------
    # Thermal gradient (finite differences on structured grid)
    # ------------------------------------------------------------------

    def _compute_gradient(self, T_new: np.ndarray, mask: np.ndarray) -> None:
        """
        Compute Gx, Gy, Gz, G = |grad T| at masked points using central
        finite differences (numpy.gradient) on the structured Cartesian grid.
        """
        dom  = self.sim.domain
        grid = self.grid

        T_3d = T_new.reshape(dom.xnum, dom.ynum, dom.znum)

        Gx_3d = np.gradient(T_3d, dom.xres, axis=0)
        Gy_3d = np.gradient(T_3d, dom.yres, axis=1)
        Gz_3d = np.gradient(T_3d, dom.zres, axis=2)

        Gx_f = Gx_3d.ravel()
        Gy_f = Gy_3d.ravel()
        Gz_f = Gz_3d.ravel()

        grid.Gx[mask] = Gx_f[mask]
        grid.Gy[mask] = Gy_f[mask]
        grid.Gz[mask] = Gz_f[mask]
        grid.G[mask]  = np.sqrt(Gx_f[mask]**2 + Gy_f[mask]**2 + Gz_f[mask]**2)
