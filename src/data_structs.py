"""
Data structures for EBM thermal modeling tool.
Mirrors 3DThesis DataStructs.h but adapted for Python + NumPy.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

PI = np.pi
STEFAN_BOLTZMANN = 5.670374419e-8  # W/m^2/K^4


# ---------------------------------------------------------------------------
# Integration quadrature nodes  (equivalent to 3DThesis Nodes struct)
# ---------------------------------------------------------------------------

@dataclass
class Nodes:
    """
    Stores Gauss-Legendre quadrature nodes accumulated over the scan history.
    Lists are used for fast appending during integration; call `as_arrays()`
    before the vectorised temperature kernel.
    """
    xb:     List[float] = field(default_factory=list)
    yb:     List[float] = field(default_factory=list)
    zb:     List[float] = field(default_factory=list)
    phix:   List[float] = field(default_factory=list)
    phiy:   List[float] = field(default_factory=list)
    phiz:   List[float] = field(default_factory=list)
    dtau:   List[float] = field(default_factory=list)
    # expmod = log(qmod) - 0.5*log(phix*phiy*phiz)  (pre-computed for speed)
    expmod: List[float] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.xb)

    def append(self, xb: float, yb: float, zb: float,
                phix: float, phiy: float, phiz: float,
                dtau: float, qmod: float) -> None:
        self.xb.append(xb)
        self.yb.append(yb)
        self.zb.append(zb)
        self.phix.append(phix)
        self.phiy.append(phiy)
        self.phiz.append(phiz)
        self.dtau.append(dtau)
        self.expmod.append(np.log(qmod) - 0.5 * np.log(phix * phiy * phiz))

    def extend(self, other: Nodes) -> None:
        self.xb     += other.xb
        self.yb     += other.yb
        self.zb     += other.zb
        self.phix   += other.phix
        self.phiy   += other.phiy
        self.phiz   += other.phiz
        self.dtau   += other.dtau
        self.expmod += other.expmod

    def clear(self) -> None:
        self.xb.clear(); self.yb.clear(); self.zb.clear()
        self.phix.clear(); self.phiy.clear(); self.phiz.clear()
        self.dtau.clear(); self.expmod.clear()

    def as_arrays(self):
        """Return all fields as 1-D numpy arrays for vectorised computation."""
        return (
            np.array(self.xb,     dtype=np.float64),
            np.array(self.yb,     dtype=np.float64),
            np.array(self.zb,     dtype=np.float64),
            np.array(self.phix,   dtype=np.float64),
            np.array(self.phiy,   dtype=np.float64),
            np.array(self.phiz,   dtype=np.float64),
            np.array(self.dtau,   dtype=np.float64),
            np.array(self.expmod, dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# Scan-path segment  (equivalent to 3DThesis path_seg)
# ---------------------------------------------------------------------------

@dataclass
class PathSegment:
    mode:       int   = 1     # 1 = spot dwell, 0 = line scan
    x:          float = 0.0   # m
    y:          float = 0.0   # m
    z:          float = 0.0   # m
    power_mod:  float = 0.0   # power multiplier [0–1]
    param:      float = 0.0   # spot: dwell time (s) | line: speed (m/s)
    seg_time:   float = 0.0   # cumulative end time (s), filled by config_reader


# ---------------------------------------------------------------------------
# Material properties
# ---------------------------------------------------------------------------

@dataclass
class Material:
    T_init: float = 923.0    # K  preheat / initial temperature
    T_liq:  float = 1928.0   # K  liquidus temperature
    k:      float = 21.9     # W/m/K  thermal conductivity
    cp:     float = 670.0    # J/kg/K specific heat
    rho:    float = 4420.0   # kg/m^3 density
    alpha:  float = 0.0      # m^2/s  thermal diffusivity (computed)
    # CET columnar-to-equiaxed transition parameters (optional)
    cet_N0: float = float('inf')
    cet_n:  float = float('inf')
    cet_a:  float = float('inf')

    def __post_init__(self):
        self.alpha = self.k / (self.rho * self.cp)


# ---------------------------------------------------------------------------
# Beam / heat source
# ---------------------------------------------------------------------------

@dataclass
class Beam:
    power:      float = 1000.0   # W   nominal beam power
    efficiency: float = 0.90     # –   absorption efficiency (EBM: ~0.9)
    width_x:    float = 300e-6   # m   beam half-width in x  (ax in 3DThesis)
    width_y:    float = 300e-6   # m   beam half-width in y  (ay)
    depth_z:    float = 150e-6   # m   beam penetration depth (az)
    # Derived — filled by config_reader after material is loaded
    q:       float = 0.0  # effective power = power * efficiency * 2  (factor 2 from Goldak)
    nond_dt: float = 0.0  # non-dimensional diffusion time = width_x^2 / alpha


# ---------------------------------------------------------------------------
# Boundary condition descriptors
# ---------------------------------------------------------------------------

@dataclass
class BCFace:
    """Thermal BC for one domain face."""
    bc_type:    str   = 'adiabatic'  # 'adiabatic' | 'radiation' | 'fixed_T' | 'semi_infinite'
    emissivity: float = 0.70         # –    used when bc_type == 'radiation'
    T_env:      float = 923.0        # K    radiation target / fixed temperature


@dataclass
class BoundaryConditions:
    top:    BCFace = field(default_factory=lambda: BCFace(bc_type='radiation'))
    sides:  BCFace = field(default_factory=lambda: BCFace(bc_type='adiabatic'))
    bottom: BCFace = field(default_factory=lambda: BCFace(bc_type='fixed_T', T_env=923.0))

    # Flat-wall image planes for method-of-images (set by config_reader)
    # DBL_MAX / inf means "not active"
    BC_xmin: float = float('inf')
    BC_xmax: float = float('inf')
    BC_ymin: float = float('inf')
    BC_ymax: float = float('inf')
    BC_zmin: float = float('inf')  # bottom adiabatic wall
    BC_reflections: int = 3        # number of image generations

    use_image_BCs: bool = False    # True when any image plane is active


# ---------------------------------------------------------------------------
# Simulation domain
# ---------------------------------------------------------------------------

@dataclass
class Domain:
    xmin: float = float('-inf')
    xmax: float = float('inf')
    ymin: float = float('-inf')
    ymax: float = float('inf')
    zmin: float = float('-inf')
    zmax: float = 0.0             # top surface of build

    # Grid resolution (may differ per axis)
    xres: float = 50e-6
    yres: float = 50e-6
    zres: float = 50e-6

    # Grid point counts (computed from bounds + resolution)
    xnum: int = 0
    ynum: int = 0
    znum: int = 0
    pnum: int = 0                 # total = xnum * ynum * znum

    bcs: BoundaryConditions = field(default_factory=BoundaryConditions)

    # Optional geometry definition (Step 5).
    # Holds a Geometry object once parsed; None = semi-infinite rectangular domain.
    # Supported: cubic, wall, cylinder. STL reserved for future extension.
    geometry: Optional[object] = None


# ---------------------------------------------------------------------------
# Simulation-mode parameters
# ---------------------------------------------------------------------------

@dataclass
class SimParams:
    mode: str = 'Solidification'  # 'Snapshots' | 'Solidification' | 'T_history'

    # Solidification mode
    dt:          float = 1e-4   # s   time step
    tracking:    str   = 'Volume'  # 'None' | 'Volume' | 'Surface'
    out_freq:    int   = 100    # output every N time steps
    secondary:   bool  = False  # compute secondary solidification params (H vector)
    sol_tol:     float = 1e-3   # relative tolerance for liquidus root-finding
    sol_max_iter: int  = 10     # max Newton iterations

    # Snapshots mode
    snapshot_times: List[float] = field(default_factory=list)

    # T_history mode
    probe_points: List[List[float]] = field(default_factory=list)  # [[x,y,z], ...]


# ---------------------------------------------------------------------------
# Solver settings
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    threads:     int   = 4
    compression: bool  = True   # path compression (as in 3DThesis)
    r_max:       float = -1.0   # cutoff radius; <0 means auto-compute
    t_hist:      float = 1e-9   # peak-T cutoff ratio  (see 3DThesis Util::Calc_RMax)
    p_hist:      float = 1e-2   # solidus-to-initial-T cutoff ratio


# ---------------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------------

@dataclass
class OutputConfig:
    directory: str        = 'Data'
    fields:    List[str]  = field(default_factory=lambda: ['x', 'y', 'z', 'T'])
    vtk:       bool       = True
    name:      str        = 'EBM_sim'


# ---------------------------------------------------------------------------
# Top-level simulation container  (equivalent to 3DThesis Simdat)
# ---------------------------------------------------------------------------

@dataclass
class Simdat:
    name:           str          = 'EBM_sim'
    material:       Material     = field(default_factory=Material)
    beam:           Beam         = field(default_factory=Beam)
    path:           List[PathSegment] = field(default_factory=list)
    domain:         Domain       = field(default_factory=Domain)
    param:          SimParams    = field(default_factory=SimParams)
    settings:       Settings     = field(default_factory=Settings)
    output:         OutputConfig = field(default_factory=OutputConfig)
    scan_end_time:  float        = 0.0   # time when beam finishes all segments
