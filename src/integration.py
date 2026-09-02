"""
Core integration engine: adaptive Gauss-Legendre quadrature over the scan history.

Ported from 3DThesis Calc::GaussIntegrate (C++ -> NumPy/Python).

Analytical temperature solution (Green's function, semi-infinite medium):

  T(x,y,z,t) = T_init + (3/pi)^1.5 * Q/(rho*cp)
               * INT_0^t  1/sqrt(phix*phiy*phiz)
                          * exp(-3*(dx^2/phix + dy^2/phiy + dz^2/phiz))  dtau

where:
  phi_i(tau) = a_i^2 + 12*alpha*tau      (diffusion-broadened beam half-width^2)
  dx,dy,dz   = (x,y,z) - beam_position(t - tau)
  Q          = beam power * efficiency * 2   (Goldak factor-of-2 convention)

The integral is evaluated by adaptive Gauss-Legendre quadrature:
  - order 16 for recent history (small tau, sharp contribution)
  - order halves and step doubles as tau grows (beam has diffused, less detail needed)
  - terminates when spatial or temporal cutoff is reached
"""

from __future__ import annotations
import math
import numpy as np
from typing import TYPE_CHECKING

from .data_structs import Nodes, Simdat, BoundaryConditions
from .boundary_conditions import add_geometry_image_sources

if TYPE_CHECKING:
    pass

PI = math.pi

# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature nodes and weights (orders 2, 4, 8, 16)
# Stored flat: order k uses slice [k-2 : 2*k-2]
#   order 2  -> indices  0:2
#   order 4  -> indices  2:6
#   order 8  -> indices  6:14
#   order 16 -> indices 14:30
# ---------------------------------------------------------------------------
_GL_LOCS = np.array([
    # order 2
    -0.57735027,  0.57735027,
    # order 4
    -0.86113631, -0.33998104,  0.33998104,  0.86113631,
    # order 8
    -0.96028986, -0.79666648, -0.52553241, -0.18343464,
     0.18343464,  0.52553241,  0.79666648,  0.96028986,
    # order 16
    -0.98940093, -0.94457502, -0.86563120, -0.75540441,
    -0.61787624, -0.45801678, -0.28160355, -0.09501251,
     0.09501251,  0.28160355,  0.45801678,  0.61787624,
     0.75540441,  0.86563120,  0.94457502,  0.98940093,
], dtype=np.float64)

_GL_WEIGHTS = np.array([
    # order 2
    1.0, 1.0,
    # order 4
    0.34785485, 0.65214515, 0.65214515, 0.34785485,
    # order 8
    0.10122854, 0.22238103, 0.31370665, 0.36268378,
    0.36268378, 0.31370665, 0.22238103, 0.10122854,
    # order 16
    0.02715246, 0.06225352, 0.09515851, 0.12462897,
    0.14959599, 0.16915652, 0.18260342, 0.18945061,
    0.18945061, 0.18260342, 0.16915652, 0.14959599,
    0.12462897, 0.09515851, 0.06225352, 0.02715246,
], dtype=np.float64)


def _gl_slice(order: int):
    """Return (locs, weights) arrays for the requested quadrature order."""
    s = slice(order - 2, 2 * order - 2)
    return _GL_LOCS[s], _GL_WEIGHTS[s]


# ---------------------------------------------------------------------------
# Integrator class
# ---------------------------------------------------------------------------

class Integrator:
    """
    Builds the Gauss-Legendre quadrature Nodes for a given simulation time t.

    One Integrator is created per simulation run and reused across timesteps.
    It caches the current path segment index so searches stay O(1) for
    forward-marching time.
    """

    def __init__(self, sim: Simdat) -> None:
        self.sim      = sim
        self.beam     = sim.beam
        self.material = sim.material
        self.settings = sim.settings
        self.path     = sim.path

        # Dimensional prefactor: beta = (3/pi)^1.5 * Q / (rho * cp)
        self.beta = (3.0 / PI) ** 1.5 * self.beam.q / (
            self.material.rho * self.material.cp
        )

        # Cached segment index for efficient forward-in-time search
        self._seg = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_nodes(self, t: float, is_solidification: bool = False) -> Nodes:
        """
        Build quadrature Nodes for temperature evaluation at time t.
        After this call, pass the Nodes to calc_temperature().
        """
        nodes = Nodes()
        self._gauss_integrate(nodes, t, is_solidification)
        if self.sim.domain.bcs.use_image_BCs:
            _add_image_sources(nodes, self.sim.domain.bcs, self.sim.domain.zmax)
        if self.sim.domain.geometry is not None:
            add_geometry_image_sources(
                nodes,
                self.sim.domain.geometry,
                self.sim.domain.bcs.BC_reflections,
            )
        return nodes

    # ------------------------------------------------------------------
    # Core adaptive quadrature
    # ------------------------------------------------------------------

    def _gauss_integrate(self, nodes: Nodes, t: float,
                         is_solidification: bool) -> None:
        path    = self.path
        beam    = self.beam
        alpha   = self.material.alpha
        beta    = self.beta

        seg = self._find_seg(t, is_solidification)

        t0 = self._t0_calc(t)

        # Initial quadrature step = non-dimensional diffusion time
        # Finer initial step in solidification mode improves melt-pool resolution.
        # Scale by (depth_z / width_x) so the ratio is dimensionless.
        step_start  = beam.nond_dt
        if is_solidification:
            step_start *= beam.depth_z / beam.width_x   # dimensionless ratio ~0.5
        curStep_max = step_start
        curOrder     = 16

        # -- Instantaneous node at t=0 (dtau=0) ---------------------------
        # Records beam position/shape at the exact current time.
        # dtau=0 means it contributes zero to T but is used in gradient
        # calculations during solidification (see solver.py Step 6).
        if t <= path[-1].seg_time:
            xb, yb, zb, qmod = self._beam_loc(t, seg)
            if qmod > 0.0:
                nodes.append(xb, yb, zb,
                             beam.width_x ** 2,
                             beam.width_y ** 2,
                             beam.depth_z ** 2,
                             0.0, qmod * beta)

        # -- Set up backward integration ------------------------------------
        t2   = t
        tpp  = 0.0   # cumulative time integrated backwards from t

        # If scan has already ended, jump to the end of the scan path
        if t > path[-1].seg_time:
            tpp += t - path[-1].seg_time
            t2   = path[-1].seg_time

        # Advance coarseness level to match already-elapsed tpp
        while tpp >= 2.0 * curStep_max - step_start:
            curStep_max *= 2.0
            if curOrder > 2:
                curOrder //= 2

        # -- Main backward integration loop --------------------------------
        tflag = True
        while tflag:
            ref_time    = self._ref_time(tpp, seg)
            curStep_use = min(ref_time, curStep_max)
            t1          = t2 - curStep_use

            next_time  = path[seg - 1].seg_time
            switch_seg = False

            if t1 <= next_time:
                t1 = next_time
                if next_time > t0:
                    switch_seg = True   # reached segment boundary, keep going
                else:
                    tflag = False       # reached cutoff time t0, stop

            # Add Gauss-Legendre quadrature points for interval [t1, t2]
            locs, weights = _gl_slice(curOrder)
            half_dt = 0.5 * (t2 - t1)
            t_mid   = 0.5 * (t2 + t1)

            for loc, w in zip(locs, weights):
                tp  = t_mid + half_dt * loc
                tau = t - tp
                ct  = 12.0 * alpha * tau

                phix = beam.width_x ** 2 + ct
                phiy = beam.width_y ** 2 + ct
                phiz = beam.depth_z ** 2 + ct

                xb, yb, zb, qmod = self._beam_loc(tp, seg)
                qmod_eff = qmod * beta
                dt_node  = half_dt * w

                if qmod_eff > 0.0 and dt_node > 0.0:
                    nodes.append(xb, yb, zb, phix, phiy, phiz, dt_node, qmod_eff)

            if switch_seg:
                seg -= 1

            tpp += t2 - t1
            t2   = t1

            # Step doubling + order halving as history grows older
            while tpp >= 2.0 * curStep_max - step_start:
                curStep_max *= 2.0
                if curOrder > 2:
                    curOrder //= 2

        # Update cached segment pointer (only for forward time-stepping)
        if not is_solidification:
            self._seg = seg

    # ------------------------------------------------------------------
    # Helper: path segment search
    # ------------------------------------------------------------------

    def _find_seg(self, t: float, is_solidification: bool) -> int:
        """Return index of the path segment active at time t."""
        path = self.path
        seg  = self._seg
        while seg + 1 < len(path) and t > path[seg].seg_time:
            seg += 1
        while seg - 1 > 0 and t < path[seg - 1].seg_time:
            seg -= 1
        if not is_solidification:
            self._seg = seg
        return seg

    # ------------------------------------------------------------------
    # Helper: beam position at time tp
    # ------------------------------------------------------------------

    def _beam_loc(self, tp: float, seg: int):
        """
        Return (xb, yb, zb, power_mod) for the beam at time tp within
        segment seg.  Returns power_mod=0 for positions outside r_max.
        Mirrors 3DThesis Util::GetBeamLoc.
        """
        path = self.path
        s    = path[seg]

        if s.mode == 1:                     # spot dwell
            xb, yb, zb = s.x, s.y, s.z
        else:                               # line scan
            prev    = path[seg - 1]
            dt_seg  = s.seg_time - prev.seg_time
            frac    = (tp - prev.seg_time) / dt_seg
            xb = prev.x + frac * (s.x - prev.x)
            yb = prev.y + frac * (s.y - prev.y)
            zb = prev.z + frac * (s.z - prev.z)

        if _in_rmax(xb, yb, self.sim.domain, self.settings):
            return xb, yb, zb, s.power_mod
        return xb, yb, zb, 0.0

    # ------------------------------------------------------------------
    # Helper: integration step reference time for current segment
    # ------------------------------------------------------------------

    def _ref_time(self, tpp: float, seg: int) -> float:
        """
        Maximum allowed integration step for the current segment.
        Mirrors 3DThesis Util::GetRefTime.
        """
        path = self.path
        beam = self.beam
        s    = path[seg]
        spp  = max(tpp / beam.nond_dt, 0.0)

        if s.mode == 0:                     # line scan
            # Step bounded by diffusion distance along track
            # sqrt(log(sqrt(2))) * width / speed
            t0 = 0.58870501125 * beam.width_x / s.param
            return t0 * math.sqrt(12.0 * spp + 1.0)
        else:                               # spot dwell
            dt = s.seg_time - path[seg - 1].seg_time
            return max(dt, 1.0e-9)

    # ------------------------------------------------------------------
    # Helper: temporal cutoff t0
    # ------------------------------------------------------------------

    def _t0_calc(self, t: float) -> float:
        """
        Compute the earliest time to integrate to (contribution cutoff).
        Integration is skipped for tau > (t - t0) when the beam's thermal
        contribution has become negligible.
        Mirrors 3DThesis Util::t0calc.

        Tunable via settings.t_hist and settings.p_hist.
        """
        beam = self.beam
        mat  = self.material
        s    = self.settings

        # Criterion 1: beam peak decays to t_hist fraction of its maximum
        t_hist_inv = 1.0 / s.t_hist
        t_hist_t   = beam.nond_dt / 12.0 * (t_hist_inv ** (2.0 / 3.0) - 1.0)

        # Criterion 2: beam contribution < p_hist * (T_liq - T_init)
        beta       = (3.0 / PI) ** 1.5 * beam.q / (mat.rho * mat.cp)
        temp_diff  = mat.T_liq - mat.T_init
        x          = temp_diff * s.p_hist
        y          = 432.0 * t * (x * x) * (mat.alpha ** 3) / (beta * beta)
        p_hist_t   = t / ((1.0 + math.sqrt(y)) ** 2) if y > 0 else 0.0

        t0 = t - max(t_hist_t, p_hist_t)
        return max(t0, 0.0)


# ---------------------------------------------------------------------------
# Temperature kernel  (vectorised over all grid points)
# ---------------------------------------------------------------------------

def calc_temperature(nodes: Nodes,
                     x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     T_init: float,
                     chunk: int = 8_000) -> np.ndarray:
    """
    Compute temperature at every grid point simultaneously using NumPy
    broadcasting.  This is the vectorised equivalent of 3DThesis Grid::Calc_T.

    T(p) = T_init + SUM_nodes [ dtau_i * exp(-3*(dx^2/phix + dy^2/phiy +
                                               dz^2/phiz) + expmod_i) ]

    Parameters
    ----------
    nodes   : quadrature nodes from Integrator.build_nodes()
    x,y,z   : 1-D arrays of grid point coordinates, shape (P,)
    T_init  : initial / preheat temperature (K)
    chunk   : number of grid points processed per batch (controls peak memory)
              Default 8000 keeps peak memory under ~200 MB for N~500 nodes.

    Returns
    -------
    T : temperature array, shape (P,)
    """
    P = len(x)
    T = np.full(P, T_init, dtype=np.float64)

    if nodes.size == 0:
        return T

    xb, yb, zb, phix, phiy, phiz, dtau, expmod = nodes.as_arrays()

    for start in range(0, P, chunk):
        end = min(start + chunk, P)

        # shape (chunk, N) via broadcasting
        dx = x[start:end, None] - xb[None, :]
        dy = y[start:end, None] - yb[None, :]
        dz = z[start:end, None] - zb[None, :]

        # Gaussian kernel: shape (chunk, N)
        kernel = np.exp(
            -3.0 * (dx * dx / phix + dy * dy / phiy + dz * dz / phiz)
            + expmod
        )

        # Weighted sum: shape (chunk,)
        T[start:end] += kernel @ dtau

    return T


# ---------------------------------------------------------------------------
# Method-of-images boundary conditions
# ---------------------------------------------------------------------------

def _add_image_sources(nodes: Nodes, bcs: BoundaryConditions,
                       domain_zmax: float = 0.0) -> None:
    """
    Add mirror image sources to enforce adiabatic (zero-flux) boundary
    conditions on flat domain walls.

    Each active BC plane reflects all existing nodes across it.  The process
    is repeated for `bcs.BC_reflections` generations, producing higher-order
    corrections for points very close to walls.

    Mirrors 3DThesis Calc::AddBCs.
    """
    if nodes.size == 0:
        return

    INF = float('inf')
    planes = {
        'xmin': bcs.BC_xmin, 'xmax': bcs.BC_xmax,
        'ymin': bcs.BC_ymin, 'ymax': bcs.BC_ymax,
        'zmin': bcs.BC_zmin,
    }
    active = {k: v for k, v in planes.items() if v != INF}
    if not active:
        return

    # Encode reflection history as integer coord tuples (same scheme as 3DThesis)
    # coords = (xRef, yRef, zRef); (0,0,0) = original
    all_coords  = [(0, 0, 0)]
    check_coords = [(0, 0, 0)]

    for _ in range(bcs.BC_reflections):
        new_coords = []
        for c in check_coords:
            xr, yr, zr = c
            candidates = []
            if 'xmin' in active: candidates.append((-1 - xr, yr, zr))
            if 'xmax' in active: candidates.append(( 1 - xr, yr, zr))
            if 'ymin' in active: candidates.append((xr, -1 - yr, zr))
            if 'ymax' in active: candidates.append((xr,  1 - yr, zr))
            if 'zmin' in active: candidates.append((xr, yr, -1 - zr))
            for nc in candidates:
                if nc not in all_coords:
                    all_coords.append(nc)
                    new_coords.append(nc)
        check_coords = new_coords
        if not check_coords:
            break

    # Convert coord indices to actual position offsets and create image nodes
    xb_orig = np.array(nodes.xb)
    yb_orig = np.array(nodes.yb)
    zb_orig = np.array(nodes.zb)

    xmin = active.get('xmin', 0.0); xmax = active.get('xmax', 0.0)
    ymin = active.get('ymin', 0.0); ymax = active.get('ymax', 0.0)
    zmin = active.get('zmin', 0.0)
    # domain_zmax is the top surface (z=0); used in the reflection formula for
    # even-order z reflections.  Must be finite to avoid 0*inf = nan in numpy.
    zmax = domain_zmax

    orig_nodes = Nodes()
    orig_nodes.xb     = list(xb_orig)
    orig_nodes.yb     = list(yb_orig)
    orig_nodes.zb     = list(zb_orig)
    orig_nodes.phix   = list(nodes.phix)
    orig_nodes.phiy   = list(nodes.phiy)
    orig_nodes.phiz   = list(nodes.phiz)
    orig_nodes.dtau   = list(nodes.dtau)
    orig_nodes.expmod = list(nodes.expmod)
    nodes.clear()

    for (xr, yr, zr) in all_coords:
        nx = abs(xr % 2)
        ny = abs(yr % 2)
        nz = abs(zr % 2)

        # Reflected coordinates (mirrors 3DThesis formula)
        xb_ref = (xr + nx) * xmax - (xr - nx) * xmin + (1 - 2*nx) * xb_orig
        yb_ref = (yr + ny) * ymax - (yr - ny) * ymin + (1 - 2*ny) * yb_orig
        zb_ref = (zr + nz) * zmax - (zr - nz) * zmin + (1 - 2*nz) * zb_orig

        img = Nodes()
        img.xb     = list(xb_ref)
        img.yb     = list(yb_ref)
        img.zb     = list(zb_ref)
        img.phix   = list(orig_nodes.phix)
        img.phiy   = list(orig_nodes.phiy)
        img.phiz   = list(orig_nodes.phiz)
        img.dtau   = list(orig_nodes.dtau)
        img.expmod = list(orig_nodes.expmod)
        nodes.extend(img)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _in_rmax(xb: float, yb: float, domain, settings) -> bool:
    """Return True if beam position (xb, yb) is within the domain + r_max buffer."""
    r = settings.r_max
    return (domain.xmin - r <= xb <= domain.xmax + r and
            domain.ymin - r <= yb <= domain.ymax + r)
