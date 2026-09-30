"""
Assemble a full T(t) profile for TC-PRISMA / DICTRA from
    (1) the simulated layer spikes (tools/vibenite290_layer_history.py),
    (2) the isothermal hold at the build temperature for the rest of the build,
    (3) an assumed cool-down after the build,
    (4) an optional post heat treatment.

The profile starts when the material point solidifies for the last time,
i.e. in the pass of the deepest layer that still remelts it.  Use the Scheil
result at that point as the initial condition in Prisma/DICTRA.

Output: Data/vibenite290/profile_<scenario>_h<height>mm.csv
        columns "Time [s]", "Temperature [C]"  (Prisma table format)

Example:
    python tools/vibenite290_prisma_profile.py B_power --height 5 \
        --layer-time 60 --ht "1180:5:10,40:0:-50,560:60:5,40:0:-2"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.vibenite290_layer_history import LAYER_T, OUT_DIR  # noqa: E402

PART_HEIGHT = 25.0     # mm
T_ROOM      = 25.0     # C


def load_history(scenario: str):
    f = OUT_DIR / f"{scenario}_T_history.csv"
    if not f.exists():
        sys.exit(f"{f} not found — run tools/vibenite290_layer_history.py "
                 f"{scenario} first.")
    with open(f) as fh:
        header = fh.readline()
    depths = np.array([float(h.split("_z")[1][:-2]) for h in header.split()[1:]])
    data = np.loadtxt(f, skiprows=1)
    return data[:, 0], data[:, 1:] - 273.15, depths * 1e-3


def column(times, temps, depths, n):
    j = int(np.argmin(np.abs(depths - n * LAYER_T)))
    return times, temps[:, j]


def thin(t, T, dT=5.0):
    """Keep points where T changed by > dT K, plus local extrema and ends."""
    keep = [0]
    for i in range(1, len(T) - 1):
        extremum = (T[i] - T[i - 1]) * (T[i + 1] - T[i]) < 0
        if extremum or abs(T[i] - T[keep[-1]]) >= dT:
            keep.append(i)
    keep.append(len(T) - 1)
    return t[keep], T[keep]


def build_profile(a) -> tuple[np.ndarray, np.ndarray, dict]:
    times, temps, depths = load_history(a.scenario)
    T_b = a.t_build
    n_max = int(round(depths.max() / LAYER_T))

    peaks = [column(times, temps, depths, n)[1].max() for n in range(n_max + 1)]
    remelt = [n for n in range(n_max + 1) if peaks[n] >= a.t_liq]
    n0 = max(remelt) if remelt else 0

    pts_t, pts_T = [], []
    clock = 0.0          # time since the point's last solidification

    # (1a) cooling after last solidification (pass n0); adjacent hatch
    # lines can remelt the point, so take the LAST drop below t_start
    t, T = column(times, temps, depths, n0)
    i0 = int(np.nonzero(T > a.t_start)[0][-1]) + 1
    seg_t, seg_T = thin(t[i0:] - t[i0], T[i0:])
    pts_t += list(seg_t); pts_T += list(seg_T)
    t_zero_layer = t[i0]  # sim time of solidification, relative to its layer

    # (1b) spikes of later layers, each one layer time apart
    n_spikes = 0
    for n in range(n0 + 1, n_max + 1):
        if peaks[n] - T_b < a.spike_min:
            break
        t, T = column(times, temps, depths, n)
        start = (n - n0) * a.layer_time - t_zero_layer
        # return to build temperature before the next spike
        pts_t.append(start); pts_T.append(T_b)
        seg_t, seg_T = thin(t, T)
        pts_t += list(start + seg_t); pts_T += list(seg_T)
        n_spikes += 1
    # the simulation window ends before deep points are back at T_build
    clock = pts_t[-1] + a.relax_time
    pts_t.append(clock); pts_T.append(T_b)

    # (2) hold at build temperature until the build finishes
    n_layers = int(round(PART_HEIGHT / (LAYER_T * 1e3)))
    n_here = int(round(a.height / (LAYER_T * 1e3)))
    t_build_end = (n_layers - n_here) * a.layer_time - t_zero_layer
    if t_build_end > clock:
        clock = t_build_end
        pts_t.append(clock); pts_T.append(T_b)

    # (3) cool-down: Newtonian decay T_room + (T_b - T_room) exp(-t/tau)
    tau = a.cool_tau_h * 3600.0
    t_c = np.linspace(0.0, a.cool_time_h * 3600.0, 60)[1:]
    pts_t += list(clock + t_c)
    pts_T += list(T_ROOM + (T_b - T_ROOM) * np.exp(-t_c / tau))
    clock = pts_t[-1]
    t_as_built = clock

    # (4) heat treatment: "T_C:hold_min:rate_C_per_s,..." (rate sign ignored)
    for seg in filter(None, (a.ht or "").split(",")):
        T_set, hold_min, rate = (float(x) for x in seg.split(":"))
        ramp = abs(T_set - pts_T[-1]) / max(abs(rate), 1e-9)
        clock += ramp; pts_t.append(clock); pts_T.append(T_set)
        if hold_min > 0:
            clock += hold_min * 60.0; pts_t.append(clock); pts_T.append(T_set)

    info = dict(n_remelt=n0, n_spikes=n_spikes, hold_h=(t_build_end) / 3600,
                t_as_built_h=t_as_built / 3600, n_here=n_here, n_layers=n_layers)
    return np.array(pts_t), np.array(pts_T), info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario", help="ref | A_speed | B_power | C_offset")
    p.add_argument("--height", type=float, required=True,
                   help="sample height above the start plate [mm]")
    p.add_argument("--layer-time", type=float, default=60.0,
                   help="time per layer incl. raking/preheat [s] (est. 40-80)")
    p.add_argument("--t-build", type=float, default=850.0, help="[C]")
    p.add_argument("--t-liq", type=float, default=1400.0,
                   help="liquidus used to detect remelting [C] (model value)")
    p.add_argument("--t-start", type=float, default=1400.0,
                   help="profile start temperature on cooling [C]; "
                        "set to the Scheil solidus if Prisma starts after Scheil")
    p.add_argument("--spike-min", type=float, default=10.0,
                   help="ignore spikes smaller than this above T_build [K]")
    p.add_argument("--relax-time", type=float, default=10.0,
                   help="time to relax back to T_build after the last "
                        "simulated spike [s]")
    p.add_argument("--cool-tau-h", type=float, default=3.3,
                   help="cool-down time constant [h] (ASSUMED: ~100 C after 8 h)")
    p.add_argument("--cool-time-h", type=float, default=12.0,
                   help="length of the cool-down segment [h]")
    p.add_argument("--ht", default="",
                   help='heat treatment "T_C:hold_min:rate_C_per_s,..."')
    a = p.parse_args()

    t, T, info = build_profile(a)
    out = OUT_DIR / f"profile_{a.scenario}_h{a.height:g}mm.csv"
    np.savetxt(out, np.column_stack([t, T]), delimiter=",", fmt="%.6g",
               header="Time [s],Temperature [C]", comments="")

    print(f"{a.scenario}, sample at {a.height:g} mm "
          f"(layer {info['n_here']}/{info['n_layers']}):")
    print(f"  last solidification in pass of layer +{info['n_remelt']}, "
          f"then {info['n_spikes']} sub-liquidus spike(s)")
    print(f"  hold at {a.t_build:.0f} C until build end: "
          f"{info['hold_h']:.2f} h (layer time {a.layer_time:g} s)")
    print(f"  as-built state reached at t = {info['t_as_built_h']:.2f} h")
    print(f"  {len(t)} points -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
