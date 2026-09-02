"""
Boundary condition corrections beyond the analytical method of images.

Two corrections are implemented here:

  1. Top-surface radiation (vacuum environment, EBM-specific)
     Applied as an explicit Euler temperature correction to the top-surface
     grid layer after each temperature evaluation.
     q_rad = eps * sigma * (T^4 - T_env^4)  [W/m^2]

  2. Geometry-wall adiabatic BCs (flat-face reflection)
     For geometry primitives that expose flat faces (cubic, wall), adds
     mirror image sources via the general plane-reflection formula.
     Curved surfaces (cylinder) are not reflected — for typical EBM scan
     tracks the melt pool is far enough from cylinder walls that the error
     is small.  This is noted in output when such a geometry is active.
"""

from __future__ import annotations
import numpy as np
from .data_structs import BoundaryConditions, Nodes, STEFAN_BOLTZMANN


# ---------------------------------------------------------------------------
# 1. Top-surface radiation correction
# ---------------------------------------------------------------------------

def apply_radiation_correction(T: np.ndarray,
                                active: np.ndarray,
                                z: np.ndarray,
                                zmax: float,
                                bcs: BoundaryConditions,
                                rho: float,
                                cp: float,
                                dz: float,
                                dt: float) -> None:
    """
    Apply radiation cooling to the top-surface grid layer, in place.

    The correction is an explicit Euler step:
        dT = -q_rad * dt / (rho * cp * dz/2)
    where dz/2 is the half-cell depth at the surface.

    Parameters
    ----------
    T      : temperature field, shape (P,) — modified in place
    active : bool mask, shape (P,)
    z      : z-coordinates, shape (P,)
    zmax   : top surface z-coordinate (m)
    bcs    : BoundaryConditions object (emissivity and T_env from bcs.top)
    rho    : density (kg/m^3)
    cp     : specific heat (J/kg/K)
    dz     : z grid spacing (m)
    dt     : timestep (s)
    """
    if bcs.top.bc_type != 'radiation':
        return

    eps   = bcs.top.emissivity
    T_env = bcs.top.T_env
    sig   = STEFAN_BOLTZMANN

    # Identify top-surface active points (within half a cell of zmax)
    at_top = active & (np.abs(z - zmax) < 0.5 * dz)
    if not np.any(at_top):
        return

    T_s   = T[at_top]
    q_rad = eps * sig * (T_s**4 - T_env**4)   # W/m^2, positive = leaving surface

    # Explicit Euler temperature decrement for surface half-cell
    dT = -q_rad * dt / (rho * cp * (dz / 2.0))
    T[at_top] += dT


# ---------------------------------------------------------------------------
# 2. Geometry-wall image sources
# ---------------------------------------------------------------------------

def add_geometry_image_sources(nodes: Nodes,
                                geometry,
                                n_reflections: int = 1) -> None:
    """
    Add method-of-images sources for flat geometry walls (adiabatic BC).

    For each flat face exposed by the geometry (via get_image_planes()), all
    existing integration nodes are reflected across the face plane.  This
    enforces dT/dn = 0 at the face, matching the analytical method already
    used for the domain walls in integration.py.

    Curved surfaces (e.g. cylinder) return an empty plane list and are
    silently skipped.

    Reflection formula (general plane with outward normal n̂ through point Q):
        P' = P - 2 * ((P - Q) · n̂) * n̂

    Parameters
    ----------
    nodes        : Nodes object — new image nodes are appended in place
    geometry     : Geometry instance (from geometry.py)
    n_reflections: number of image generations (1 is sufficient for most cases)
    """
    if nodes.size == 0 or geometry is None:
        return

    planes = geometry.get_image_planes()
    if not planes:
        return

    for _ in range(n_reflections):
        # Snapshot the current node count so each generation only reflects
        # the nodes that existed at the start of this generation.
        N = nodes.size
        xb = np.array(nodes.xb[:N])
        yb = np.array(nodes.yb[:N])
        zb = np.array(nodes.zb[:N])
        phix   = np.array(nodes.phix[:N])
        phiy   = np.array(nodes.phiy[:N])
        phiz   = np.array(nodes.phiz[:N])
        dtau   = np.array(nodes.dtau[:N])
        expmod = np.array(nodes.expmod[:N])

        for normal, point in planes:
            nx, ny, nz = normal
            qx, qy, qz = point

            # Signed distance of each source from the plane
            dot = (xb - qx)*nx + (yb - qy)*ny + (zb - qz)*nz

            xb_ref = xb - 2.0 * dot * nx
            yb_ref = yb - 2.0 * dot * ny
            zb_ref = zb - 2.0 * dot * nz

            img = Nodes()
            img.xb     = list(xb_ref)
            img.yb     = list(yb_ref)
            img.zb     = list(zb_ref)
            img.phix   = list(phix)
            img.phiy   = list(phiy)
            img.phiz   = list(phiz)
            img.dtau   = list(dtau)
            img.expmod = list(expmod)
            nodes.extend(img)
