"""
Grid module: coordinate arrays, result storage, and active-point masking.

The Grid holds all spatial coordinates and simulation-output arrays for
every grid point in the domain.  It is the central data container passed
between the Solver (Step 6) and Output (Step 7) modules.

Grid layout mirrors 3DThesis: p = i*(ynum*znum) + j*znum + k
  i: x index  (0 .. xnum-1)
  j: y index  (0 .. ynum-1)
  k: z index  (0 .. znum-1)
"""

from __future__ import annotations
import numpy as np
from .data_structs import Simdat, Domain


class Grid:
    """
    Structured Cartesian grid for EBM thermal simulation.

    Attributes
    ----------
    x, y, z    : 1-D float64, shape (P,) — grid point coordinates (m)
    T          : temperature field (K), updated each timestep
    G          : thermal gradient magnitude |grad T| at solidification (K/m)
    Gx, Gy, Gz : gradient components (K/m)
    V          : solidification velocity (m/s)
    dTdt       : cooling rate at solidification (K/s)
    tSol       : time of solidification (s), NaN if not yet solidified
    numMelt    : number of times each point has been melted
    active     : bool mask — False for points outside the part geometry
                 (surrounding powder/vacuum). When no geometry is set, all
                 points in the rectangular domain are active.
    """

    def __init__(self, sim: Simdat) -> None:
        dom = sim.domain
        self._build_coordinates(dom)
        self._allocate_arrays(sim.material.T_init)
        self._apply_geometry_mask(dom)
        self._print_summary(sim)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_coordinates(self, dom: Domain) -> None:
        """Create 1-D coordinate arrays and update domain grid counts."""
        xs = np.arange(dom.xmin, dom.xmax + 0.5 * dom.xres, dom.xres)
        ys = np.arange(dom.ymin, dom.ymax + 0.5 * dom.yres, dom.yres)
        zs = np.arange(dom.zmin, dom.zmax + 0.5 * dom.zres, dom.zres)

        dom.xnum = len(xs)
        dom.ynum = len(ys)
        dom.znum = len(zs)
        dom.pnum = dom.xnum * dom.ynum * dom.znum

        # meshgrid with indexing='ij' → p = i*(ynum*znum) + j*znum + k on ravel
        xg, yg, zg = np.meshgrid(xs, ys, zs, indexing='ij')
        self.x = xg.ravel()
        self.y = yg.ravel()
        self.z = zg.ravel()

    def _allocate_arrays(self, T_init: float) -> None:
        P = len(self.x)

        self.T      = np.full(P, T_init, dtype=np.float64)
        self.G      = np.zeros(P, dtype=np.float64)
        self.Gx     = np.zeros(P, dtype=np.float64)
        self.Gy     = np.zeros(P, dtype=np.float64)
        self.Gz     = np.zeros(P, dtype=np.float64)
        self.V      = np.zeros(P, dtype=np.float64)
        self.dTdt   = np.zeros(P, dtype=np.float64)
        self.tSol   = np.full(P, np.nan, dtype=np.float64)
        self.numMelt = np.zeros(P, dtype=np.int32)

        # Previous-step temperature — used for liquidus-crossing detection
        self._T_prev = np.full(P, T_init, dtype=np.float64)

        # All points active initially; geometry masking applied next
        self.active = np.ones(P, dtype=bool)

    def _apply_geometry_mask(self, dom: Domain) -> None:
        """Keep only grid points that lie inside the part geometry.
        Points outside the geometry (surrounding powder/vacuum) are inactive."""
        if dom.geometry is None:
            return
        inside = dom.geometry.is_inside(self.x, self.y, self.z)
        self.active &= inside

    # ------------------------------------------------------------------
    # Active-point views  (convenience — avoids repeated boolean indexing)
    # ------------------------------------------------------------------

    @property
    def xa(self) -> np.ndarray:
        return self.x[self.active]

    @property
    def ya(self) -> np.ndarray:
        return self.y[self.active]

    @property
    def za(self) -> np.ndarray:
        return self.z[self.active]

    # ------------------------------------------------------------------
    # Liquidus-crossing detection  (called by Solver each timestep)
    # ------------------------------------------------------------------

    def update_solidification(self, T_new: np.ndarray, t: float,
                              dt: float, T_liq: float) -> None:
        """
        Detect liquidus crossings in T_new vs _T_prev, then update
        tSol, dTdt, and numMelt.  G and V are left for the Solver to
        fill in after computing the spatial gradient at solidified points.

        Parameters
        ----------
        T_new : freshly computed temperature field, shape (P,)
        t     : current simulation time (s) — end of this timestep
        dt    : timestep size (s)
        T_liq : liquidus temperature (K)
        """
        prev = self._T_prev

        # Points that crossed from solid to liquid this step
        just_melted = self.active & (prev < T_liq) & (T_new >= T_liq)
        self.numMelt[just_melted] += 1

        # Points that crossed from liquid to solid this step (solidification)
        just_solidified = self.active & (prev >= T_liq) & (T_new < T_liq)

        if np.any(just_solidified):
            T_p = prev[just_solidified]
            T_c = T_new[just_solidified]

            # Linear interpolation for the exact crossing time
            frac = (T_p - T_liq) / (T_p - T_c)          # in [0, 1]
            self.tSol[just_solidified] = (t - dt) + frac * dt

            # Cooling rate at solidification front
            self.dTdt[just_solidified] = (T_c - T_p) / dt

        # Advance stored state
        np.copyto(self._T_prev, T_new)
        np.copyto(self.T, T_new)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset_T(self, T_init: float) -> None:
        """Reset temperature and history arrays (useful for re-runs)."""
        self.T[:]      = T_init
        self._T_prev[:] = T_init

    def _print_summary(self, sim: Simdat) -> None:
        dom = sim.domain
        active_count = int(self.active.sum())
        total = dom.pnum
        print(f"Grid: {dom.xnum} x {dom.ynum} x {dom.znum} = {total} points "
              f"({active_count} active)")
        print(f"  x: [{dom.xmin*1e3:.2f}, {dom.xmax*1e3:.2f}] mm  "
              f"res={dom.xres*1e6:.0f} um")
        print(f"  y: [{dom.ymin*1e3:.2f}, {dom.ymax*1e3:.2f}] mm  "
              f"res={dom.yres*1e6:.0f} um")
        print(f"  z: [{dom.zmin*1e3:.2f}, {dom.zmax*1e3:.2f}] mm  "
              f"res={dom.zres*1e6:.0f} um")
