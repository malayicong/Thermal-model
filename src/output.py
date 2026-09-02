"""
Output module: CSV and VTK writing.

CSV  — one row per active grid point, space-separated, compatible with
       3DThesis post-processing scripts.

VTK  — VTK Rectilinear Grid (.vtr) for ParaView 3-D visualization.
       Inactive (geometry-masked) points are written as NaN so the
       structured grid remains complete and ParaView can threshold them out.

T_history — CSV time series at probe points.
"""

from __future__ import annotations
import os
import numpy as np
from .data_structs import Simdat
from .grid import Grid


# ---------------------------------------------------------------------------
# Field catalogue
# ---------------------------------------------------------------------------

def _get_field(name: str, grid: Grid) -> np.ndarray | None:
    """Return the 1-D array for a named output field, or None if unavailable."""
    if name == 'x':     return grid.x
    if name == 'y':     return grid.y
    if name == 'z':     return grid.z
    if name == 'T':     return grid.T
    if name == 'G':     return grid.G
    if name == 'Gx':    return grid.Gx
    if name == 'Gy':    return grid.Gy
    if name == 'Gz':    return grid.Gz
    if name == 'V':     return grid.V
    if name == 'dTdt':  return grid.dTdt
    if name == 'tSol':  return grid.tSol
    if name == 'numMelt': return grid.numMelt.astype(np.float64)
    if name == 'depth': return -grid.z   # distance below top surface (m)
    return None


def _field_header(name: str) -> str:
    """Header label for the CSV column."""
    labels = {
        'x': 'x_m', 'y': 'y_m', 'z': 'z_m',
        'T': 'T_K', 'G': 'G_Kpm', 'Gx': 'Gx_Kpm', 'Gy': 'Gy_Kpm', 'Gz': 'Gz_Kpm',
        'V': 'V_mps', 'dTdt': 'dTdt_Kps',
        'tSol': 'tSol_s', 'numMelt': 'numMelt', 'depth': 'depth_m',
    }
    return labels.get(name, name)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(grid: Grid, sim: Simdat, suffix: str = '') -> None:
    """
    Write active grid points to a space-separated CSV file.

    Only active (non-masked) points are written.  Fields written are
    determined by sim.output.fields in config.yaml.

    Parameters
    ----------
    grid   : Grid object (after solver has run)
    sim    : Simdat (contains output config)
    suffix : optional filename suffix, e.g. '_t0002' for snapshot files
    """
    out  = sim.output
    mask = grid.active

    fields = [f for f in out.fields if _get_field(f, grid) is not None]
    if not fields:
        print("CSV: no recognised output fields specified.")
        return

    fname = os.path.join(out.directory, f"{out.name}{suffix}.csv")
    os.makedirs(out.directory, exist_ok=True)

    header = ' '.join(_field_header(f) for f in fields)
    data   = np.column_stack([_get_field(f, grid)[mask] for f in fields])

    np.savetxt(fname, data, header=header, comments='', fmt='%.6e')
    print(f"CSV written: {fname}  ({mask.sum()} points, fields: {fields})")


# ---------------------------------------------------------------------------
# VTK output  (VTK Rectilinear Grid — .vtr, opens directly in ParaView)
# ---------------------------------------------------------------------------

def write_vtk(grid: Grid, sim: Simdat, suffix: str = '') -> None:
    """
    Write the full structured grid to a VTK Rectilinear Grid (.vtr) file.

    The .vtr format stores a regular Cartesian grid with per-point field data.
    Open the file in ParaView: File > Open > select *.vtr.

    Inactive (geometry-masked) points are written as NaN — in ParaView use
    Filters > Threshold to remove them (set range to exclude NaN).

    Parameters
    ----------
    grid   : Grid object (after solver has run)
    sim    : Simdat
    suffix : optional filename suffix
    """
    try:
        from pyevtk.hl import gridToVTK
    except ImportError:
        print("VTK output skipped: 'pyevtk' not installed.  "
              "Run: pip install pyevtk")
        return

    out = sim.output
    dom = sim.domain

    # Unique 1-D coordinate arrays for each axis
    xs = np.unique(grid.x).astype(np.float64)   # shape (xnum,)
    ys = np.unique(grid.y).astype(np.float64)
    zs = np.unique(grid.z).astype(np.float64)

    # Mask: inactive points become NaN in VTK output
    mask_3d = grid.active.reshape(dom.xnum, dom.ynum, dom.znum)

    def to_vtk_array(arr: np.ndarray) -> np.ndarray:
        """Reshape to (xnum, ynum, znum), NaN-fill inactive, return C-contiguous float64."""
        a = arr.reshape(dom.xnum, dom.ynum, dom.znum).astype(np.float64).copy()
        a[~mask_3d] = np.nan
        return np.ascontiguousarray(a)

    # Build point-data dict from requested fields (skip x/y/z — VTK stores coords)
    scalar_fields = [f for f in out.fields
                     if f not in ('x', 'y', 'z') and _get_field(f, grid) is not None]

    point_data = {f: to_vtk_array(_get_field(f, grid)) for f in scalar_fields}

    if not point_data:
        print("VTK: no scalar fields to write (x/y/z are stored as coordinates).")
        return

    path = os.path.join(out.directory, f"{out.name}{suffix}")
    os.makedirs(out.directory, exist_ok=True)

    gridToVTK(path, xs, ys, zs, pointData=point_data)

    vtr_file = path + '.vtr'
    print(f"VTK written: {vtr_file}  "
          f"(grid {dom.xnum}x{dom.ynum}x{dom.znum}, fields: {scalar_fields})")


# ---------------------------------------------------------------------------
# T_history output
# ---------------------------------------------------------------------------

def write_t_history(t_history_data: np.ndarray,
                    sim: Simdat) -> None:
    """
    Write T_history probe-point data to CSV.

    Columns: time_s, T_probe0_K, T_probe1_K, ...
    Rows:    one per timestep.

    Parameters
    ----------
    t_history_data : shape (n_steps, 1 + n_probes) — from Solver.t_history_data
    sim            : Simdat (contains probe_points and output config)
    """
    if t_history_data is None or len(t_history_data) == 0:
        return

    out    = sim.output
    probes = sim.param.probe_points
    n      = len(probes)

    header_parts = ['time_s'] + [f'T_probe{i}_K' for i in range(n)]
    header       = ' '.join(header_parts)

    fname = os.path.join(out.directory, f"{out.name}_T_history.csv")
    os.makedirs(out.directory, exist_ok=True)

    np.savetxt(fname, t_history_data, header=header, comments='', fmt='%.6e')

    probe_strs = [f"[{p[0]*1e3:.2f},{p[1]*1e3:.2f},{p[2]*1e3:.2f}]mm" for p in probes]
    print(f"T_history written: {fname}  "
          f"({len(t_history_data)} steps, probes: {probe_strs})")
