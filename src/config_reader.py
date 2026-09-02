"""
Config reader: loads config.yaml + path.csv into a Simdat object.

Two-file input system:
  config.yaml  — all simulation parameters
  path.csv     — scan path (tabular data, referenced from config.yaml)
"""

from __future__ import annotations
import csv
import math
import os
import warnings
from pathlib import Path
from typing import List, Optional

import yaml

from .data_structs import (
    Simdat, Material, Beam, PathSegment, Domain,
    BoundaryConditions, BCFace, SimParams, Settings, OutputConfig,
)
from .geometry import make_geometry

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Simdat:
    """
    Parse config.yaml (and the path.csv it references) into a Simdat object.
    All derived quantities (alpha, beam.q, grid counts, r_max, ...) are
    computed here so downstream modules receive a fully-initialised Simdat.
    """
    config_path = Path(config_path).resolve()
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    sim = Simdat()
    sim.name = cfg.get('simulation', {}).get('name', 'EBM_sim')
    sim.output.name = sim.name

    # --- Material ----------------------------------------------------------
    sim.material = _parse_material(cfg.get('material', {}))

    # --- Beam --------------------------------------------------------------
    sim.beam = _parse_beam(cfg.get('beam', {}), sim.material)

    # --- Scan path ---------------------------------------------------------
    path_cfg = cfg.get('path', [])
    if isinstance(path_cfg, list):
        sim.path = _parse_inline_path(path_cfg)
    else:
        path_abs = config_path.parent / path_cfg.get('file', 'path.csv')
        sim.path = _read_path_csv(str(path_abs))
    sim.scan_end_time = sim.path[-1].seg_time if len(sim.path) > 1 else 0.0

    # --- Domain (needs path bounds if bounds not explicit) -----------------
    sim.domain = _parse_domain(cfg.get('domain', {}),
                               cfg.get('boundary_conditions', {}),
                               sim.path)

    # --- Simulation mode ---------------------------------------------------
    sim_cfg = cfg.get('simulation', {})
    sim.param = _parse_mode(sim_cfg.get('mode', 'Solidification'),
                            cfg.get('mode_settings', {}),
                            sim.scan_end_time)

    # --- Compute settings --------------------------------------------------
    sim.settings = _parse_settings(cfg.get('compute', {}),
                                   sim.beam, sim.material)

    # --- Output ------------------------------------------------------------
    sim.output = _parse_output(cfg.get('output', {}), sim.name)

    _print_summary(sim)
    return sim


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------

def _parse_material(mat: dict) -> Material:
    m = Material(
        T_init = mat.get('T_init', 923.0),
        T_liq  = mat.get('T_liq',  1928.0),
        k      = mat.get('k',      21.9),
        cp     = mat.get('cp',     670.0),
        rho    = mat.get('rho',    4420.0),
    )
    cet = mat.get('cet', {})
    m.cet_N0 = cet.get('N0', float('inf'))
    m.cet_n  = cet.get('n',  float('inf'))
    m.cet_a  = cet.get('a',  float('inf'))
    # alpha is computed in Material.__post_init__
    return m


# ---------------------------------------------------------------------------
# Beam
# ---------------------------------------------------------------------------

def _parse_beam(bm: dict, mat: Material) -> Beam:
    b = Beam(
        power      = bm.get('power',      1000.0),
        efficiency = bm.get('efficiency', 0.90),
        width_x    = bm.get('width_x',    300e-6),
        width_y    = bm.get('width_y',    300e-6),
        depth_z    = bm.get('depth_z',    150e-6),
    )
    # Factor-of-2 from Goldak double-ellipsoid convention (same as 3DThesis SetBeamPower)
    b.q = b.power * b.efficiency * 2.0
    # Non-dimensional diffusion time: time for beam to diffuse one beam-width
    b.nond_dt = b.width_x ** 2 / mat.alpha
    return b


# ---------------------------------------------------------------------------
# Scan path
# ---------------------------------------------------------------------------

def _parse_inline_path(rows: list) -> List[PathSegment]:
    """
    Parse path embedded in config.yaml as a list of lists:
      [[mode, x_mm, y_mm, z_mm, power_mod, param], ...]
    """
    segs: List[PathSegment] = [
        PathSegment(mode=1, x=0.0, y=0.0, z=0.0,
                    power_mod=0.0, param=0.0, seg_time=0.0)
    ]
    for row in rows:
        segs.append(PathSegment(
            mode      = int(row[0]),
            x         = float(row[1]) * 1e-3,
            y         = float(row[2]) * 1e-3,
            z         = float(row[3]) * 1e-3,
            power_mod = float(row[4]),
            param     = float(row[5]),
        ))
    return _compute_seg_times(segs)


def _read_path_csv(path_file: str) -> List[PathSegment]:
    """Fallback: read path from an external CSV file for long/generated paths."""
    segs: List[PathSegment] = [
        PathSegment(mode=1, x=0.0, y=0.0, z=0.0,
                    power_mod=0.0, param=0.0, seg_time=0.0)
    ]
    with open(path_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            segs.append(PathSegment(
                mode      = int(row['mode']),
                x         = float(row['x_mm'])    * 1e-3,
                y         = float(row['y_mm'])    * 1e-3,
                z         = float(row['z_mm'])    * 1e-3,
                power_mod = float(row['power_mod']),
                param     = float(row['param']),
            ))
    return _compute_seg_times(segs)


def _compute_seg_times(segs: List[PathSegment]) -> List[PathSegment]:
    """Fill seg_time on each segment (cumulative end time)."""
    for i in range(1, len(segs)):
        prev, seg = segs[i - 1], segs[i]
        if seg.mode == 1:
            seg.seg_time = prev.seg_time + seg.param
        else:
            dx = seg.x - prev.x
            dy = seg.y - prev.y
            dz = seg.z - prev.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if seg.param <= 0.0:
                raise ValueError(f"Segment {i}: line scan speed must be > 0.")
            seg.seg_time = prev.seg_time + dist / seg.param
    return segs


# ---------------------------------------------------------------------------
# Domain + BCs
# ---------------------------------------------------------------------------

def _parse_domain(dom: dict, bc_cfg: dict,
                  path: List[PathSegment]) -> Domain:
    d = Domain()

    # Resolve bounds: explicit config values take priority; otherwise auto from path
    x_range = dom.get('x', None)
    y_range = dom.get('y', None)
    z_range = dom.get('z', None)

    if x_range is None or y_range is None or z_range is None:
        auto = _path_bounds(path)
    else:
        auto = None

    # Explicit float() cast: PyYAML only recognises scientific notation as float
    # when there is a decimal point (e.g. 5.0e-3).  Without one (e.g. 5e-3)
    # the value is returned as a string.  The cast makes both forms safe.
    if x_range:
        d.xmin, d.xmax = float(x_range[0]), float(x_range[1])
    else:
        d.xmin, d.xmax = auto['xmin'] - 500e-6, auto['xmax'] + 500e-6

    if y_range:
        d.ymin, d.ymax = float(y_range[0]), float(y_range[1])
    else:
        d.ymin, d.ymax = auto['ymin'] - 500e-6, auto['ymax'] + 500e-6

    if z_range:
        d.zmin, d.zmax = float(z_range[0]), float(z_range[1])
    else:
        d.zmin, d.zmax = auto['zmin'] - 250e-6, auto['zmax']

    # Resolution — can be uniform or per-axis
    default_res = float(dom.get('resolution', 50e-6))
    d.xres = float(dom.get('xres', default_res))
    d.yres = float(dom.get('yres', default_res))
    d.zres = float(dom.get('zres', default_res))

    # Grid counts
    d.xnum = max(1, round((d.xmax - d.xmin) / d.xres) + 1)
    d.ynum = max(1, round((d.ymax - d.ymin) / d.yres) + 1)
    d.znum = max(1, round((d.zmax - d.zmin) / d.zres) + 1)
    d.pnum = d.xnum * d.ynum * d.znum

    # Geometry definition (parsed in Step 5; stored as None until geometry.py is added)
    geo_cfg = dom.get('geometry', {})
    d.geometry = make_geometry(geo_cfg) if geo_cfg else None

    # Boundary conditions
    d.bcs = _parse_bcs(bc_cfg, d)

    return d


def _path_bounds(path: List[PathSegment]) -> dict:
    """Scan path bounding box (only active segments with power > 0)."""
    active = [s for s in path if s.power_mod > 0]
    if not active:
        active = path
    xs = [s.x for s in active]; ys = [s.y for s in active]; zs = [s.z for s in active]
    return dict(xmin=min(xs), xmax=max(xs),
                ymin=min(ys), ymax=max(ys),
                zmin=min(zs), zmax=max(zs))


def _parse_bcs(bc_cfg: dict, d: Domain) -> BoundaryConditions:
    bcs = BoundaryConditions()

    # --- Top surface (z = zmax, powder-bed / vacuum interface) ---
    top = bc_cfg.get('top_surface', {})
    bcs.top = BCFace(
        bc_type    = top.get('type',       'radiation'),
        emissivity = top.get('emissivity', 0.70),
        T_env      = top.get('T_env',      923.0),
    )
    # The analytical Green's function for a semi-infinite medium already
    # satisfies ∂T/∂z = 0 at z = zmax (the natural Neumann BC).
    # Radiation at the top surface is applied as an iterative flux correction
    # in the BC module (Step 4); no image source is added for z = zmax.

    # --- Side walls (x and y faces) ---
    sides = bc_cfg.get('side_walls', {})
    bcs.sides = BCFace(
        bc_type    = sides.get('type', 'adiabatic'),
        emissivity = sides.get('emissivity', 0.70),
        T_env      = sides.get('T_env', 923.0),
    )
    if bcs.sides.bc_type == 'adiabatic':
        bcs.BC_xmin = d.xmin
        bcs.BC_xmax = d.xmax
        bcs.BC_ymin = d.ymin
        bcs.BC_ymax = d.ymax
        bcs.use_image_BCs = True

    # --- Bottom (z = zmin, substrate / build plate) ---
    bot = bc_cfg.get('bottom', {})
    bcs.bottom = BCFace(
        bc_type    = bot.get('type',   'fixed_T'),
        emissivity = bot.get('emissivity', 0.70),
        T_env      = bot.get('value',   923.0),
    )
    if bcs.bottom.bc_type == 'adiabatic':
        bcs.BC_zmin = d.zmin
        bcs.use_image_BCs = True
    elif bcs.bottom.bc_type == 'fixed_T':
        # Fixed-T at the bottom cannot be enforced analytically with method of
        # images (images enforce Neumann, not Dirichlet).  We treat the domain
        # as semi-infinite here and issue a warning.  For thin substrates,
        # the FDM correction layer in Step 4 can be used.
        warnings.warn(
            "Bottom BC 'fixed_T' is approximated as semi-infinite in the "
            "analytical solver.  Use a deep domain (zmin << 0) to minimise error.",
            UserWarning, stacklevel=3,
        )

    bcs.BC_reflections = bc_cfg.get('reflections', 3)
    return bcs


# ---------------------------------------------------------------------------
# Simulation mode
# ---------------------------------------------------------------------------

def _parse_mode(mode: str, mode_cfg: dict, scan_end_time: float) -> SimParams:
    p = SimParams(mode=mode)

    if mode == 'Solidification':
        sol = mode_cfg.get('solidification', {})
        p.dt           = sol.get('timestep',          1e-4)
        p.tracking     = sol.get('tracking',          'Volume')
        p.out_freq     = sol.get('output_frequency',  100)
        p.secondary    = bool(sol.get('secondary',    False))
        p.sol_tol      = sol.get('sol_tolerance',     1e-3)
        p.sol_max_iter = sol.get('sol_max_iter',      10)

    elif mode == 'Snapshots':
        snap = mode_cfg.get('snapshots', {})
        raw = snap.get('times', [])
        # Allow scan-fraction syntax: '50%' → 0.5 * scan_end_time
        p.snapshot_times = _parse_times(raw, scan_end_time)

    elif mode == 'T_history':
        th = mode_cfg.get('t_history', {})
        p.dt           = th.get('timestep', 1e-4)
        p.probe_points = th.get('probe_points', [])

    else:
        raise ValueError(f"Unknown simulation mode: '{mode}'")

    return p


def _parse_times(raw, scan_end_time: float) -> List[float]:
    times = []
    for v in raw:
        if isinstance(v, str) and v.endswith('%'):
            times.append(float(v[:-1]) / 100.0 * scan_end_time)
        else:
            times.append(float(v))
    return sorted(times)


# ---------------------------------------------------------------------------
# Compute settings + auto r_max
# ---------------------------------------------------------------------------

def _parse_settings(comp: dict, beam: Beam, mat: Material) -> Settings:
    s = Settings(
        threads     = comp.get('threads',     4),
        compression = bool(comp.get('compression', True)),
        r_max       = comp.get('r_max',       -1.0),
        t_hist      = comp.get('t_hist',      1e-9),
        p_hist      = comp.get('p_hist',      1e-2),
    )

    if s.r_max < 0.0:
        s.r_max = _auto_rmax(beam, mat, s)

    return s


def _auto_rmax(beam: Beam, mat: Material, s: Settings) -> float:
    """
    Estimate maximum radius beyond which the beam contribution is negligible.
    Mirrors 3DThesis Util::Calc_RMax logic.
    """
    t_hist_inv = 1.0 / s.t_hist
    if t_hist_inv < math.exp(1.5):
        r1 = beam.width_x * math.sqrt(math.log(t_hist_inv) / 3.0)
    else:
        r1 = beam.width_x * t_hist_inv ** (1.0 / 3.0) / math.sqrt(2.0 * math.e)

    import numpy as np
    beta = (3.0 / math.pi) ** 1.5 * beam.q / (mat.rho * mat.cp)
    temp_diff = mat.T_liq - mat.T_init
    x = temp_diff * s.p_hist
    ratio = beta / (x * beam.width_x ** 3)
    if ratio < math.exp(1.5):
        r2 = beam.width_x * math.sqrt(math.log(ratio) / 3.0) if ratio > 1.0 else 0.0
    else:
        r2 = ratio ** (1.0 / 3.0) / math.sqrt(2.0 * math.e)

    return max(r1, r2)


# ---------------------------------------------------------------------------
# Output config
# ---------------------------------------------------------------------------

def _parse_output(out: dict, sim_name: str) -> OutputConfig:
    return OutputConfig(
        directory = out.get('directory', 'Data'),
        fields    = out.get('fields',    ['x', 'y', 'z', 'T']),
        vtk       = bool(out.get('vtk', True)),
        name      = sim_name,
    )


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------

def _print_summary(sim: Simdat) -> None:
    d = sim.domain
    b = sim.beam
    m = sim.material
    print("=" * 55)
    print(f"  EBM Thermal Solver - {sim.name}")
    print("=" * 55)
    print(f"  Mode          : {sim.param.mode}")
    print(f"  Material      : T_init={m.T_init} K | T_liq={m.T_liq} K | a={m.alpha:.2e} m2/s")
    print(f"  Beam          : P={b.power} W | eff={b.efficiency} | w={b.width_x*1e6:.0f} um")
    print(f"  Domain        : [{d.xmin*1e3:.1f},{d.xmax*1e3:.1f}] x "
          f"[{d.ymin*1e3:.1f},{d.ymax*1e3:.1f}] x "
          f"[{d.zmin*1e3:.1f},{d.zmax*1e3:.1f}] mm")
    print(f"  Grid          : {d.xnum} x {d.ynum} x {d.znum} = {d.pnum:,} points")
    print(f"  BCs           : top={d.bcs.top.bc_type} | sides={d.bcs.sides.bc_type} | bottom={d.bcs.bottom.bc_type}")
    geo_desc = d.geometry.description if d.geometry is not None else 'none (semi-infinite)'
    print(f"  Geometry      : {geo_desc}")
    print(f"  Scan end time : {sim.scan_end_time:.4f} s")
    print("=" * 55)
