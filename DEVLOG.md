# EBM Thermal Modeling Tool — Development Log

## Project Goal
Build a Python-based thermal modeling tool for **Electron Beam Melting (EBM)** that:
- Extends the analytical Green's function approach from the 3DThesis reference tool
- Adds EBM-specific physics (high preheat, vacuum radiation BC, higher beam efficiency)
- Supports **geometry-specific boundary conditions** (arbitrary part surfaces via STL)
- Uses a simplified, unified input system (1–2 files instead of 3DThesis's 6–8 files)

---

## Reference Material

| Item | Path / Source |
|------|---------------|
| Reference tool | `\\ug.kth.se\...\3DThesis-master\` |
| Reference paper | Yang et al. (2025) — *Computational efficient process simulation of geometrically complex parts in metal AM* |
| Output directory | `C:\3DThesis-proj\claudecodeGenerated\` |

---

## Conversation Log

### Session 1 — 2026-04-22

#### User request
> "Help me write a tool for thermal modeling during electron beam melting. Reference tool is in '3DThesis-master'. I want it to also work when the user has a specific modeling geometry (boundary conditions should be considered carefully). Put scripts in 'claudecodeGenerated'. For boundary condition, the attached paper is for reference. Please give a plan first."

#### What was read / explored
- All source files in `3DThesis-master/src/` and `3DThesis-master/include/`
- All test input files (`Beam.txt`, `Domain.txt`, `IN718.txt`, `Mode.txt`, `Path.txt`, `Settings.txt`)
- Reference paper: Yang et al. (2025)

#### Key findings from 3DThesis
- Uses **analytical Green's function** (Nguyen et al., Welding Journal 1999) for a semi-infinite homogeneous medium
- Heat source: **3D ellipsoidal Gaussian** (Goldak-type), parameterized by `ax`, `ay`, `az`
- Integration method: **adaptive Gauss-Legendre quadrature** (orders 2/4/8/16) over scan history
- Boundary conditions: **method of images** (mirror heat sources) for flat adiabatic walls only
- Modes: `Snapshots` (T field at given times), `Solidification` (tracks melt pool, computes G, V, dT/dt, CET)
- Parallelized with OpenMP; path compression for efficiency
- Temperature kernel: `T = T_init + ∫ β/√(φx·φy·φz) · exp(−3(Δx²/φx + Δy²/φy + Δz²/φz)) dτ`
  where `φi = ai² + 12α·τ` (diffusion-broadened beam width)

---

### Session 1 — Planning decisions

#### Overall architecture decision
Port the analytical core to **Python + NumPy** (vectorized over grid points), extend with EBM physics and geometry-aware BCs. Rationale:
- Python is more accessible for thesis/research users
- NumPy vectorization compensates for the Python overhead vs C++
- Easier to extend with STL geometry, VTK output, and iterative BC corrections

#### Input file simplification
**User request:**
> "I wish I can modify the input parameters in only one or two files."

**Decision:** Reduce 3DThesis's 6–8 separate input files to **two files**:
1. `config.yaml` — all simulation parameters (beam, material, domain, BCs, mode, settings, output)
2. `path.csv` — scan path data only (tabular, too large to embed for real scans)

YAML chosen over JSON/INI because: human-readable, supports inline comments, good for nested structures, standard in scientific Python.

---

## Implementation Plan

### Step 1 — Project scaffold & data structures
- Create directory structure
- Python dataclasses mirroring 3DThesis structs (`Material`, `Beam`, `Domain`, `SimParams`, `Settings`, `Nodes`, `path_seg`)
- YAML config reader (`config.yaml` → all param structs)
- CSV path reader

### Step 2 — Core analytical integration engine (`integration.py`)
- Port `Calc::GaussIntegrate` to NumPy-vectorized Python
- Adaptive Gauss-Legendre quadrature (orders 2/4/8/16)
- Cutoff radius logic (`r_max`, `t0calc`)
- Vectorized over all grid points simultaneously via NumPy broadcasting (key performance difference from 3DThesis per-point loop)

#### Tunable parameters and their effect on accuracy / speed

All parameters below live in the `compute:` or `boundary_conditions:` block of `config.yaml`.

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `t_hist` | `compute` | `1e-9` | Temporal cutoff: stops integrating when beam peak has decayed to this fraction of its current maximum. **Decrease** → integrate longer history → more accurate cooling-phase temperatures, slower. **Increase** (e.g. `1e-6`) → faster; accuracy loss mainly for points far from the melt pool during cooling. |
| `p_hist` | `compute` | `1e-2` | Stops integrating when the beam's thermal contribution falls below `p_hist × (T_liq − T_init)`. **Decrease** → more accurate near-preheat temperatures; needed if dT/dt far from the track matters. **Increase** → faster; safe if only melt-pool geometry is of interest. |
| `r_max` | `compute` | auto | Spatial cutoff radius: beam positions outside `domain ± r_max` contribute zero. Auto-computed from beam width and cutoff ratios. **Set manually** (positive value in metres) to enlarge if temperatures at domain edges look incorrect. This is the primary speed control. |
| `compression` | `compute` | `true` | Path-segment compression: merges spatially close historical segments into a single averaged source. **`true`** → faster for long or repetitive scan paths; slight accuracy loss in complex serpentine patterns. **`false`** → maximum accuracy; use for short paths or when dT/dt at track reversals is critical. |
| `reflections` | `boundary_conditions` | `3` | Number of image-source generations for adiabatic walls. `1` → single reflection, adequate for points > ~2 beam widths from a wall. `3` → sufficient for most cases. `5+` → only needed for thin features or points pressed against a wall. Cost: each generation can up to double the node count per active wall. |

#### Fixed internal parameters (not yet exposed in config)

| Parameter | Value | How to change | Effect if changed |
|-----------|-------|---------------|-------------------|
| Max quadrature order | 16 | Edit `_gauss_integrate`: set `curOrder = 8` initially | Order 16 → most accurate near melt pool; order 8 acceptable for snapshot T fields; order 4 only for coarse scoping runs. |
| Chunk size in `calc_temperature` | 8000 points | Edit `chunk` arg | Larger → fewer Python loop iterations (faster) but higher peak RAM. Reduce if memory errors occur on fine grids. |
| Step-doubling schedule | doubles every `nond_dt` of elapsed history | Edit the `while tpp >= 2*curStep_max - step_start` logic | Slower doubling → more nodes, more accurate at large tau (cooling); faster doubling → fewer nodes, faster but less accurate. |

### Step 3 — EBM heat source & material files
- Single Gaussian beam (EBM: beam diameter ~0.2–1 mm, efficiency ~0.90)
- High preheat T_0: ~600–950 °C (EBM-specific, critical for thermal gradients)
- Material files: **Ti-6Al-4V** and **IN718** with EBM-appropriate values
- Temperature kernel in `grid.py`: `Calc_T` equivalent

### Step 4 — Boundary condition module (`boundary_conditions.py`)
Three BC types:

| Boundary | Type | Method |
|----------|------|--------|
| Top surface (z = z_top, powder bed) | Radiation (vacuum) | Iterative virtual source correction: `q_rad = ε·σ·(T⁴ − T_env⁴)` |
| Side/bottom flat walls | Adiabatic (insulated) | Method of images (identical to 3DThesis) |
| Geometry walls (STL surfaces) | Adiabatic or prescribed flux | Extended method of images: local planar patch reflection |

Key references for BCs:
- Goldak & Akhlaghi, *Computational Welding Mechanics* (2005)
- Denlinger et al., *J. Manuf. Sci. Eng.* 139(7) (2017) — radiation BC in AM
- Yang et al. (2025) — efficient geometry-aware BC treatment

### Step 5 — Geometry module (`geometry.py`)

**Decision (2026-04-23):** User requested simple built-in primitives only (no STL for now).
Three supported types: `box`, `wall`, `cylinder`.
STL and CSG kept as future extension points via a base class.

Each primitive implements two methods:
- `is_inside(x, y, z)` — analytical point-in-solid test (used to mask grid points)
- `surface_normal(x, y, z)` — outward unit normal at a boundary point (used for BC direction)

**Primitives:**

| Type | Key parameters | `is_inside` test |
|------|---------------|-----------------|
| `cubic` | `x`, `y`, `z` ranges | all three interval checks |
| `wall` | `normal` axis, `pos`, `thickness`, in-plane extents | thin slab along one axis |
| `cylinder` | `center [x,y]`, `radius`, `z` range | radial + z interval check |

**Config syntax:**
```yaml
geometry:
  type: cylinder
  center: [2.5e-3, 1.0e-3]
  radius: 2.0e-3
  # z range from domain bounds

geometry:
  type: cubic
  x: [0, 5e-3]
  y: [0, 1e-3]
  z: [-3e-3, 0]

geometry:
  type: wall
  normal: y              # wall faces in y-direction
  thickness: 0.5e-3      # m
  center: [2.5e-3, 1.0e-3, -1.5e-3]
  extents: [5e-3, 3e-3]  # [x-length, z-height]
```

**Extensibility:** a `Geometry` base class with abstract `is_inside` / `surface_normal`
methods makes it trivial to add `sphere`, `cone`, STL, or CSG later.

### Step 6 — Solver modes (`solver.py`)
- `Snapshot`: T field at specified times
- `Solidification`: stepping loop, detects liquidus crossing, computes G / V / dT/dt
- `T_history`: temperature vs time at probe points

### Step 7 — Output module (`output.py`)
- CSV (same column format as 3DThesis — compatible with existing post-processing)
- VTK `.vts` structured grid for ParaView 3D visualization

### Step 8 — Example input files
- `config.yaml` with EBM defaults for Ti-6Al-4V
- `path.csv` with a simple raster scan example
- Notes on geometry file format

---

## Status Log

| Step | Status | Notes |
|------|--------|-------|
| Planning | ✅ Done | Plan approved by user |
| Input file design | ✅ Done | YAML + CSV approach approved |
| DEVLOG.md | ✅ Done | This file |
| Step 1 — Scaffold | ✅ Done | Verified: `python main.py` parses config, prints summary, exits cleanly |
| Step 2 — Integration | ✅ Done | Verified: correct node counts, T in [923, 3335] K, no NaNs across 3 time points |
| Step 3 — Grid module | ✅ Done | `src/grid.py`: coord arrays, result arrays (T/G/V/dTdt/tSol/numMelt), active mask, liquidus-crossing detection; `inputs/config_IN718.yaml` added |
| Step 5 — Geometry | ✅ Done | `src/geometry.py`: `CubicGeometry`, `WallGeometry`, `CylinderGeometry`; ABC base; `make_geometry()` factory; `get_image_planes()` for flat-face reflections |
| Step 4 — Boundary conditions | ✅ Done | `src/boundary_conditions.py`: `apply_radiation_correction()` (explicit Euler surface flux); `add_geometry_image_sources()` (plane-reflection for flat faces); integrated into `Integrator.build_nodes` |
| Step 6 — Solver | ✅ Done | `src/solver.py`: Solidification (G/V/dTdt, early-exit), Snapshots, T_history; `_compute_gradient` via `np.gradient`; radiation BC hooked per timestep |
| Step 7 — Output | ✅ Done | `src/output.py`: `write_csv` (space-delimited, active points only), `write_vtk` (pyevtk `.vtr` rectilinear grid, NaN for masked points), `write_t_history`; all wired in `main.py` |
| Step 8 — Input examples | ✅ Done | `config.yaml` geometry syntax fixed (`type: null`), geometry comment block added; `config_cylinder_example.yaml` added (r=2 mm cylinder, Ti-6Al-4V, 74% active points); `config_IN718.yaml` syntax fixed |

---

## Notes & Decisions (ongoing)

- All spatial units: **meters** (same as 3DThesis internal representation; path.csv uses mm with auto-conversion)
- All temperatures: **Kelvin**
- Grid indexing: `p = i*(ynum*znum) + j*znum + k` (same as 3DThesis `ijk_to_p`)
- `zmax` = top surface of build (z = 0 convention, negative z = into substrate, same as 3DThesis)
- **Geometry semantics** (corrected 2026-04-27): `geometry` in config defines the **part shape**. Points **inside** the geometry are active (heated material). Points outside (surrounding powder/vacuum) are excluded. The geometry surface has an adiabatic BC.
- **VTK output**: `.vtr` rectilinear grid format (pyevtk). Open in ParaView: File → Open → select `*.vtr`, color by field `T`. Inactive (outside-geometry) points are written as NaN — use Filters → Threshold to remove them.
- **integration.py `is_solidification` flag**: only controls segment-pointer caching and initial quadrature step size. The step size scales as `nond_dt * (depth_z / width_x)` (dimensionless ratio) for solidification mode, giving finer temporal resolution near the melt pool.

---

## Input file quick reference

| File | Purpose |
|------|---------|
| `inputs/config.yaml` | Ti-6Al-4V, no geometry, Solidification mode |
| `inputs/config_IN718.yaml` | IN718, no geometry, Solidification mode |
| `inputs/config_cylinder_example.yaml` | Ti-6Al-4V, r=2 mm cylinder part, Solidification mode |

**To run:**
```
python main.py                              # uses config.yaml
python main.py inputs/config_IN718.yaml
python main.py inputs/config_cylinder_example.yaml
```

**Output** (in `Data/` by default):
- `<name>.csv` — space-delimited, active points, all requested fields
- `<name>.vtr` — open in ParaView for 3D visualization
