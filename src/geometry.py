"""
Geometry primitives for masking grid points and providing surface normals
for geometry-aware boundary conditions.

Supported types: cubic, wall, cylinder.
All primitives share a common ABC so more types can be added later.

is_inside(x, y, z)       -- True for points inside the solid body
surface_normal(x, y, z)  -- outward unit normal at a boundary point
get_image_planes()        -- list of (normal_vec, plane_point) for flat faces;
                             used by boundary_conditions to add image sources
                             (returns [] for curved surfaces like cylinder)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Geometry(ABC):
    """Abstract base for all geometry primitives."""

    @abstractmethod
    def is_inside(self,
                  x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return bool array: True where the point is inside the solid."""

    @abstractmethod
    def surface_normal(self,
                       x: np.ndarray, y: np.ndarray,
                       z: np.ndarray) -> np.ndarray:
        """
        Return outward unit normal at each point, shape (..., 3).
        For interior points the normal points toward the nearest face.
        """

    def get_image_planes(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Return a list of (normal_vec, plane_point) pairs for each flat face
        of this geometry.  Used by the boundary-condition module to construct
        method-of-images sources for adiabatic wall BCs.

        Returns an empty list for curved surfaces where flat-plane
        reflection is not a valid approximation.
        """
        return []

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line human-readable description (for print summaries)."""


# ---------------------------------------------------------------------------
# Cubic (box)
# ---------------------------------------------------------------------------

class CubicGeometry(Geometry):
    """
    Axis-aligned rectangular solid.

    Parameters
    ----------
    x_range : [xmin, xmax]  (m)
    y_range : [ymin, ymax]  (m)
    z_range : [zmin, zmax]  (m)
    """

    def __init__(self, x_range, y_range, z_range):
        self.x0, self.x1 = float(x_range[0]), float(x_range[1])
        self.y0, self.y1 = float(y_range[0]), float(y_range[1])
        self.z0, self.z1 = float(z_range[0]), float(z_range[1])

    def is_inside(self, x, y, z):
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        return ((x >= self.x0) & (x <= self.x1) &
                (y >= self.y0) & (y <= self.y1) &
                (z >= self.z0) & (z <= self.z1))

    def surface_normal(self, x, y, z):
        x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
        shape = np.broadcast(x, y, z).shape

        # Distance from each of the 6 faces (positive = inside)
        d = np.stack([
            x - self.x0,        # dist from x_min face (inward +x)
            self.x1 - x,        # dist from x_max face (inward -x)
            y - self.y0,
            self.y1 - y,
            z - self.z0,
            self.z1 - z,
        ], axis=-1)

        # Outward normals for each face
        face_normals = np.array([
            [-1.,  0.,  0.],   # x_min outward = -x
            [ 1.,  0.,  0.],   # x_max outward = +x
            [ 0., -1.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0., -1.],
            [ 0.,  0.,  1.],
        ])

        # Closest face = smallest non-negative distance (or smallest absolute)
        closest = np.argmin(np.abs(d), axis=-1)
        return face_normals[closest]

    def get_image_planes(self):
        return [
            (np.array([-1., 0., 0.]), np.array([self.x0, 0., 0.])),
            (np.array([ 1., 0., 0.]), np.array([self.x1, 0., 0.])),
            (np.array([ 0.,-1., 0.]), np.array([0., self.y0, 0.])),
            (np.array([ 0., 1., 0.]), np.array([0., self.y1, 0.])),
            (np.array([ 0., 0.,-1.]), np.array([0., 0., self.z0])),
            (np.array([ 0., 0., 1.]), np.array([0., 0., self.z1])),
        ]

    @property
    def description(self):
        return (f"cubic  "
                f"x=[{self.x0*1e3:.2f},{self.x1*1e3:.2f}] mm  "
                f"y=[{self.y0*1e3:.2f},{self.y1*1e3:.2f}] mm  "
                f"z=[{self.z0*1e3:.2f},{self.z1*1e3:.2f}] mm")


# ---------------------------------------------------------------------------
# Wall  (thin slab along one axis)
# ---------------------------------------------------------------------------

class WallGeometry(Geometry):
    """
    Thin flat slab oriented perpendicular to one coordinate axis.

    Parameters
    ----------
    normal    : 'x', 'y', or 'z' — axis the wall is perpendicular to
    center    : [cx, cy, cz]      — geometric centre of the wall (m)
    thickness : wall thickness (m)
    extents   : [len1, len2]      — size in the two in-plane directions (m)
    """

    def __init__(self, normal: str, center, thickness: float, extents):
        self.axis = normal.lower()
        if self.axis not in ('x', 'y', 'z'):
            raise ValueError(f"Wall normal must be 'x', 'y', or 'z'; got '{normal}'")
        self.cx = float(center[0])
        self.cy = float(center[1])
        self.cz = float(center[2])
        self.t  = float(thickness)
        self.e0 = float(extents[0])
        self.e1 = float(extents[1])

    def is_inside(self, x, y, z):
        x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
        t2 = self.t / 2.0
        ax = self.axis
        if ax == 'x':
            in_norm = np.abs(x - self.cx) <= t2
            in_p1   = np.abs(y - self.cy) <= self.e0 / 2.0
            in_p2   = np.abs(z - self.cz) <= self.e1 / 2.0
        elif ax == 'y':
            in_norm = np.abs(y - self.cy) <= t2
            in_p1   = np.abs(x - self.cx) <= self.e0 / 2.0
            in_p2   = np.abs(z - self.cz) <= self.e1 / 2.0
        else:
            in_norm = np.abs(z - self.cz) <= t2
            in_p1   = np.abs(x - self.cx) <= self.e0 / 2.0
            in_p2   = np.abs(y - self.cy) <= self.e1 / 2.0
        return in_norm & in_p1 & in_p2

    def surface_normal(self, x, y, z):
        x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
        shape = np.broadcast(x, y, z).shape
        normals = np.zeros(shape + (3,))
        ax = self.axis
        if ax == 'x':
            normals[..., 0] = np.where(x >= self.cx, 1., -1.)
        elif ax == 'y':
            normals[..., 1] = np.where(y >= self.cy, 1., -1.)
        else:
            normals[..., 2] = np.where(z >= self.cz, 1., -1.)
        return normals

    def get_image_planes(self):
        ax = self.axis
        if ax == 'x':
            pos0 = self.cx - self.t / 2.0
            pos1 = self.cx + self.t / 2.0
            return [
                (np.array([-1., 0., 0.]), np.array([pos0, 0., 0.])),
                (np.array([ 1., 0., 0.]), np.array([pos1, 0., 0.])),
            ]
        elif ax == 'y':
            pos0 = self.cy - self.t / 2.0
            pos1 = self.cy + self.t / 2.0
            return [
                (np.array([0., -1., 0.]), np.array([0., pos0, 0.])),
                (np.array([0.,  1., 0.]), np.array([0., pos1, 0.])),
            ]
        else:
            pos0 = self.cz - self.t / 2.0
            pos1 = self.cz + self.t / 2.0
            return [
                (np.array([0., 0., -1.]), np.array([0., 0., pos0])),
                (np.array([0., 0.,  1.]), np.array([0., 0., pos1])),
            ]

    @property
    def description(self):
        return (f"wall  normal={self.axis}  "
                f"center=[{self.cx*1e3:.2f},{self.cy*1e3:.2f},{self.cz*1e3:.2f}] mm  "
                f"t={self.t*1e3:.2f} mm  extents=[{self.e0*1e3:.2f},{self.e1*1e3:.2f}] mm")


# ---------------------------------------------------------------------------
# Cylinder  (axis parallel to z)
# ---------------------------------------------------------------------------

class CylinderGeometry(Geometry):
    """
    Vertical cylinder with axis parallel to z.

    Parameters
    ----------
    center  : [cx, cy]       — centre in the x-y plane (m)
    radius  : cylinder radius (m)
    z_range : [zmin, zmax]   — axial extent (m); None = unbounded in z
    """

    def __init__(self, center, radius: float, z_range=None):
        self.cx = float(center[0])
        self.cy = float(center[1])
        self.r  = float(radius)
        self.z_range: Optional[Tuple[float, float]] = (
            (float(z_range[0]), float(z_range[1])) if z_range is not None else None
        )

    def is_inside(self, x, y, z):
        x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
        in_r = (x - self.cx)**2 + (y - self.cy)**2 <= self.r**2
        if self.z_range is not None:
            in_z = (z >= self.z_range[0]) & (z <= self.z_range[1])
            return in_r & in_z
        return in_r

    def surface_normal(self, x, y, z):
        x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
        shape = np.broadcast(x, y, z).shape
        normals = np.zeros(shape + (3,))
        dx = x - self.cx
        dy = y - self.cy
        r  = np.sqrt(dx**2 + dy**2)
        safe_r = np.where(r > 0., r, 1.)
        normals[..., 0] = dx / safe_r
        normals[..., 1] = dy / safe_r
        return normals

    # Cylinder has no flat faces — image-source reflections are not supported.
    # Points near a cylinder wall can still be adiabatic in practice if the
    # melt pool is far from the curved wall (as is typical for EBM track widths).
    def get_image_planes(self):
        return []

    @property
    def description(self):
        z_str = (f"z=[{self.z_range[0]*1e3:.2f},{self.z_range[1]*1e3:.2f}] mm"
                 if self.z_range else "z=unbounded")
        return (f"cylinder  "
                f"center=[{self.cx*1e3:.2f},{self.cy*1e3:.2f}] mm  "
                f"r={self.r*1e3:.2f} mm  {z_str}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_geometry(cfg: dict) -> Optional[Geometry]:
    """
    Parse a geometry config dict (from config.yaml) → Geometry object or None.

    Supported types: cubic, wall, cylinder.
    Returns None for type=null/none/empty.
    """
    if not cfg:
        return None

    gtype = str(cfg.get('type', 'null')).lower()
    if gtype in ('null', 'none', ''):
        return None

    if gtype == 'cubic':
        return CubicGeometry(cfg['x'], cfg['y'], cfg['z'])

    if gtype == 'wall':
        return WallGeometry(
            normal    = cfg['normal'],
            center    = cfg['center'],
            thickness = float(cfg['thickness']),
            extents   = cfg['extents'],
        )

    if gtype == 'cylinder':
        return CylinderGeometry(
            center  = cfg['center'],
            radius  = float(cfg['radius']),
            z_range = cfg.get('z_range', None),
        )

    raise ValueError(
        f"Unknown geometry type: '{gtype}'. Supported: cubic, wall, cylinder"
    )
