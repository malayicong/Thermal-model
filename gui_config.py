#!/usr/bin/env python3
"""GUI configuration tool for the EBM Thermal Solver.

Usage:
    python gui_config.py                      # opens inputs/config.yaml
    python gui_config.py inputs/myconfig.yaml # opens specific config
"""
import sys
import os
import yaml
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QComboBox, QCheckBox, QPushButton, QTextEdit, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QFrame, QSizePolicy, QAbstractItemView, QFileDialog, QSplitter,
)
from PyQt5.QtCore import Qt, pyqtSignal, QProcess
from PyQt5.QtGui import QFont, QColor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

ROOT           = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(ROOT, 'inputs', 'config.yaml')
MAIN_SCRIPT    = os.path.join(ROOT, 'main.py')

# ─────────────────────────────── helpers ─────────────────────────────────────

def _f(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d

# Custom YAML float representer: guarantees decimal point so PyYAML parses
# the saved file back as float (e.g. 5.0e-3, not 5e-3 which would be a string)
def _float_rep(dumper, value):
    s = f'{value:.10g}'
    if 'e' in s:
        m, e = s.split('e')
        if '.' not in m:
            m += '.0'
        s = f'{m}e{e}'
    elif '.' not in s:
        s += '.0'
    return dumper.represent_scalar('tag:yaml.org,2002:float', s)

yaml.add_representer(float, _float_rep)

def _label(text, w=120):
    lb = QLabel(text)
    lb.setFixedWidth(w)
    lb.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return lb

def _hrow(label_text, *widgets):
    row = QWidget()
    hl  = QHBoxLayout(row)
    hl.setContentsMargins(0, 1, 0, 1)
    hl.setSpacing(6)
    hl.addWidget(_label(label_text))
    for w in widgets:
        hl.addWidget(w)
    return row

def _dbl(v=0.0, lo=-1e9, hi=1e9, dec=6, step=0.1):
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setDecimals(dec)
    sb.setSingleStep(step)
    sb.setValue(v)
    return sb

def _spin(v=1, lo=0, hi=9999):
    sb = QSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(v)
    return sb

def _combo(*opts):
    cb = QComboBox()
    cb.addItems(opts)
    return cb

def _section(title):
    lb = QLabel(title)
    lb.setStyleSheet(
        'font-weight: 500; font-size: 12px; color: #5F5E5A;'
        'margin-top: 8px; margin-bottom: 2px;'
    )
    return lb

# ─────────────────────────── BC colour map ───────────────────────────────────

_BC_FC = {
    'radiation':     '#B5D4F4',
    'adiabatic':     '#FAC775',
    'fixed_T':       '#C0DD97',
    'semi_infinite': '#D3D1C7',
}
_BC_TC = {
    'radiation':     '#185FA5',
    'adiabatic':     '#854F0B',
    'fixed_T':       '#3B6D11',
    'semi_infinite': '#5F5E5A',
}

# ─────────────────────────── Matplotlib canvas ───────────────────────────────

class PreviewCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(3.5, 5.0), dpi=96)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fig.patch.set_facecolor('#fafaf8')

    # ── public entry points ──────────────────────────────────────────────────

    def show_geometry(self, cfg):
        self.fig.clear()
        dom   = cfg.get('domain') or {}
        geo   = dom.get('geometry') or {}
        gtype = geo.get('type') if geo else None

        x   = dom.get('x') or [-1e-3, 6e-3]
        y   = dom.get('y') or [-1e-3, 2e-3]
        z   = dom.get('z') or [-1e-3, 0.0]
        xmn, xmx = _f(x[0]) * 1e3, _f(x[1]) * 1e3
        ymn, ymx = _f(y[0]) * 1e3, _f(y[1]) * 1e3
        zmn, zmx = _f(z[0]) * 1e3, _f(z[1]) * 1e3

        ax1 = self.fig.add_subplot(2, 1, 1)
        ax2 = self.fig.add_subplot(2, 1, 2)

        # ── top view x-y ────────────────────────────────────────────────────
        ax1.set_facecolor('#f8f8f6')
        ax1.add_patch(mpatches.Rectangle(
            (xmn, ymn), xmx - xmn, ymx - ymn,
            lw=1, ec='#888780', fc='#f0ede6', ls='--'))

        if gtype == 'cylinder':
            cx = _f((geo.get('center') or [0, 0])[0]) * 1e3
            cy = _f((geo.get('center') or [0, 0])[1]) * 1e3
            r  = _f(geo.get('radius', 2e-3)) * 1e3
            ax1.add_patch(mpatches.Circle(
                (cx, cy), r, lw=1.5, ec='#185FA5', fc='#ddeeff'))
            ax1.plot(cx, cy, '+', c='#185FA5', ms=8, mew=1.2)
            ax1.annotate(f'r={r:.2g} mm', (cx + r * 0.55, cy + r * 0.55),
                         fontsize=7, color='#185FA5')
        elif gtype == 'cubic':
            gx = geo.get('x') or [xmn / 1e3, xmx / 1e3]
            gy = geo.get('y') or [ymn / 1e3, ymx / 1e3]
            ax1.add_patch(mpatches.Rectangle(
                (_f(gx[0]) * 1e3, _f(gy[0]) * 1e3),
                (_f(gx[1]) - _f(gx[0])) * 1e3,
                (_f(gy[1]) - _f(gy[0])) * 1e3,
                lw=1.5, ec='#185FA5', fc='#ddeeff'))

        ax1.set_xlim(xmn - .15 * (xmx - xmn), xmx + .15 * (xmx - xmn))
        ax1.set_ylim(ymn - .15 * (ymx - ymn), ymx + .15 * (ymx - ymn))
        ax1.set_aspect('equal', 'datalim')
        self._decorate(ax1, 'x (mm)', 'y (mm)', 'Top view (x–y)')

        # ── side view x-z ────────────────────────────────────────────────────
        ax2.set_facecolor('#f8f8f6')
        ax2.add_patch(mpatches.Rectangle(
            (xmn, zmn), xmx - xmn, zmx - zmn,
            lw=1, ec='#888780', fc='#f0ede6', ls='--'))

        if gtype == 'cylinder':
            cx = _f((geo.get('center') or [0, 0])[0]) * 1e3
            r  = _f(geo.get('radius', 2e-3)) * 1e3
            zr = geo.get('z_range')
            gz0 = _f(zr[0]) * 1e3 if zr else zmn
            gz1 = _f(zr[1]) * 1e3 if zr else zmx
            ax2.add_patch(mpatches.Rectangle(
                (cx - r, gz0), 2 * r, gz1 - gz0,
                lw=1.5, ec='#185FA5', fc='#ddeeff'))
            ax2.text(cx, (gz0 + gz1) / 2, 'cylinder',
                     ha='center', va='center', fontsize=8, color='#185FA5')
        elif gtype == 'cubic':
            gx = geo.get('x') or [xmn / 1e3, xmx / 1e3]
            gz = geo.get('z') or [zmn / 1e3, zmx / 1e3]
            ax2.add_patch(mpatches.Rectangle(
                (_f(gx[0]) * 1e3, _f(gz[0]) * 1e3),
                (_f(gx[1]) - _f(gx[0])) * 1e3,
                (_f(gz[1]) - _f(gz[0])) * 1e3,
                lw=1.5, ec='#185FA5', fc='#ddeeff'))

        ax2.set_xlim(xmn - .15 * (xmx - xmn), xmx + .15 * (xmx - xmn))
        ax2.set_ylim(zmn - .2 * abs(zmx - zmn), zmx + .2 * abs(zmx - zmn))
        self._decorate(ax2, 'x (mm)', 'z (mm)', 'Side view (x–z)')

        self.fig.tight_layout(pad=1.0)
        self.draw()

    def show_bc(self, cfg):
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.set_facecolor('#f8f8f6')

        dom   = cfg.get('domain') or {}
        bcs   = cfg.get('boundary_conditions') or {}
        geo   = dom.get('geometry') or {}
        gtype = geo.get('type') if geo else None

        x   = dom.get('x') or [-1e-3, 6e-3]
        z   = dom.get('z') or [-1e-3, 0.0]
        xmn, xmx = _f(x[0]) * 1e3, _f(x[1]) * 1e3
        zmn, zmx = _f(z[0]) * 1e3, _f(z[1]) * 1e3
        W = xmx - xmn
        H = abs(zmx - zmn)

        # domain interior
        ax.add_patch(mpatches.Rectangle(
            (xmn, zmn), W, zmx - zmn, lw=0, fc='#f0ede6'))

        # geometry part + side-wall adiabatic patches
        if gtype == 'cylinder':
            cx = _f((geo.get('center') or [0, 0])[0]) * 1e3
            r  = _f(geo.get('radius', 2e-3)) * 1e3
            zr = geo.get('z_range')
            gz0 = _f(zr[0]) * 1e3 if zr else zmn
            gz1 = _f(zr[1]) * 1e3 if zr else zmx
            ax.add_patch(mpatches.Rectangle(
                (cx - r, gz0), 2 * r, gz1 - gz0,
                lw=1.5, ec='#185FA5', fc='#ddeeff', zorder=2))
            ax.text(cx, (gz0 + gz1) / 2, 'part\n(cylinder)',
                    ha='center', va='center', fontsize=8, color='#185FA5', zorder=3)
            # geometry wall patches
            tw = W * 0.028
            for xs in [cx - r - tw, cx + r]:
                ax.add_patch(mpatches.Rectangle(
                    (xs, gz0), tw, gz1 - gz0,
                    lw=0, fc=_BC_FC['adiabatic'], zorder=4))
            kw = dict(ha='center', va='center', fontsize=6.5,
                      color=_BC_TC['adiabatic'], rotation=90, zorder=5)
            ax.text(cx - r - tw * 2.2, (gz0 + gz1) / 2, 'adiab.\n(geo)', **kw)
            ax.text(cx + r + tw * 2.2, (gz0 + gz1) / 2, 'adiab.\n(geo)', **kw)

        elif gtype == 'cubic':
            gx = geo.get('x') or [xmn / 1e3, xmx / 1e3]
            gz = geo.get('z') or [zmn / 1e3, zmx / 1e3]
            gx0, gx1 = _f(gx[0]) * 1e3, _f(gx[1]) * 1e3
            gz0, gz1 = _f(gz[0]) * 1e3, _f(gz[1]) * 1e3
            ax.add_patch(mpatches.Rectangle(
                (gx0, gz0), gx1 - gx0, gz1 - gz0,
                lw=1.5, ec='#185FA5', fc='#ddeeff', zorder=2))
            ax.text((gx0 + gx1) / 2, (gz0 + gz1) / 2, 'part\n(cubic)',
                    ha='center', va='center', fontsize=8, color='#185FA5', zorder=3)

        th_h = H * 0.05
        tw_w = W * 0.05

        # top BC
        top  = bcs.get('top_surface') or {}
        tt   = top.get('type', 'radiation')
        ax.add_patch(mpatches.Rectangle(
            (xmn, zmx), W, th_h, lw=0,
            fc=_BC_FC.get(tt, '#D3D1C7'), zorder=5))
        lbl = tt
        if tt == 'radiation':
            lbl = (f"radiation  "
                   f"ε={top.get('emissivity', '')}  "
                   f"T_env={top.get('T_env', '')} K")
        ax.text((xmn + xmx) / 2, zmx + th_h * 1.8, lbl,
                ha='center', va='bottom', fontsize=7.5,
                color=_BC_TC.get(tt, '#5F5E5A'))

        # bottom BC
        bot  = bcs.get('bottom') or {}
        bt   = bot.get('type', 'fixed_T')
        ax.add_patch(mpatches.Rectangle(
            (xmn, zmn - th_h), W, th_h, lw=0,
            fc=_BC_FC.get(bt, '#D3D1C7'), zorder=5))
        lbl = bt
        if bt == 'fixed_T':
            lbl = f"fixed_T  {bot.get('value', '')} K"
        ax.text((xmn + xmx) / 2, zmn - th_h * 2.4, lbl,
                ha='center', va='top', fontsize=7.5,
                color=_BC_TC.get(bt, '#5F5E5A'))

        # side BCs
        side = bcs.get('side_walls') or {}
        st   = side.get('type', 'adiabatic')
        for xs in [xmn - tw_w, xmx]:
            ax.add_patch(mpatches.Rectangle(
                (xs, zmn), tw_w, zmx - zmn, lw=0,
                fc=_BC_FC.get(st, '#D3D1C7'), zorder=5))
        ax.text(xmn - tw_w * 2.4, (zmn + zmx) / 2, st,
                ha='right', va='center', fontsize=7.5,
                color=_BC_TC.get(st, '#5F5E5A'), rotation=90)
        ax.text(xmx + tw_w * 2.4, (zmn + zmx) / 2, st,
                ha='left',  va='center', fontsize=7.5,
                color=_BC_TC.get(st, '#5F5E5A'), rotation=90)

        # domain outline
        ax.add_patch(mpatches.Rectangle(
            (xmn, zmn), W, zmx - zmn,
            lw=0.8, ec='#888780', fc='none', zorder=6))

        refl = bcs.get('reflections', 3)
        ax.text(xmn + W * 0.03, zmx - H * 0.08,
                f'reflections = {refl}', fontsize=7.5, color='#888780')

        ax.set_xlim(xmn - W * 0.34, xmx + W * 0.34)
        ax.set_ylim(zmn - H * 0.44, zmx + H * 0.40)
        self._decorate(ax, 'x (mm)', 'z (mm)', 'Boundary conditions (x–z)')

        leg = [mpatches.Patch(fc=_BC_FC[t], ec=_BC_TC[t], lw=0.5, label=t)
               for t in _BC_FC]
        ax.legend(handles=leg, loc='lower right', fontsize=7,
                  framealpha=0.88, edgecolor='#cccccc')

        self.fig.tight_layout(pad=1.0)
        self.draw()

    def show_placeholder(self, msg=''):
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.set_facecolor('#f8f8f6')
        ax.text(0.5, 0.5, msg or 'Switch to\nDomain & geometry\nor\nBoundary conditions\nto see preview',
                ha='center', va='center', fontsize=9, color='#B4B2A9',
                transform=ax.transAxes, linespacing=1.6)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        self.draw()

    def _decorate(self, ax, xlabel, ylabel, title):
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='normal', pad=4)
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.3, alpha=0.45)


# ─────────────────────────────── Tab: Simulation ─────────────────────────────

class SimTab(QWidget):
    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        vl = QVBoxLayout(self)
        vl.setAlignment(Qt.AlignTop)
        vl.setSpacing(4)

        self.name = QLineEdit('EBM_example')
        self.mode = _combo('Solidification', 'Snapshots', 'T_history')
        vl.addWidget(_hrow('Name', self.name))
        vl.addWidget(_hrow('Mode', self.mode))

        vl.addWidget(_section('Solidification settings'))
        self.dt       = _dbl(1e-4, 1e-7, 1.0, 7, 1e-5)
        self.out_freq = _spin(50, 1, 10000)
        vl.addWidget(_hrow('Timestep (s)', self.dt))
        vl.addWidget(_hrow('Output every N steps', self.out_freq))

        vl.addWidget(_section('Snapshots settings'))
        self.snap_times = QLineEdit('1.0e-3, 5.0e-3, 10.0e-3')
        vl.addWidget(_hrow('Times (s, comma-sep)', self.snap_times))

        vl.addStretch()
        for w in [self.name, self.snap_times]:
            w.textChanged.connect(self.changed)
        for w in [self.mode, ]:
            w.currentIndexChanged.connect(self.changed)
        for w in [self.dt, self.out_freq]:
            w.valueChanged.connect(self.changed)

    def load(self, cfg):
        s = cfg.get('simulation') or {}
        self.name.setText(s.get('name', ''))
        modes = ['Solidification', 'Snapshots', 'T_history']
        m = s.get('mode', 'Solidification')
        self.mode.setCurrentIndex(modes.index(m) if m in modes else 0)
        sol = (cfg.get('mode_settings') or {}).get('solidification') or {}
        self.dt.setValue(_f(sol.get('timestep', 1e-4)))
        self.out_freq.setValue(int(_f(sol.get('output_frequency', 50))))
        snap = (cfg.get('mode_settings') or {}).get('snapshots') or {}
        times = snap.get('times') or [1e-3, 5e-3, 10e-3]
        self.snap_times.setText(', '.join(str(t) for t in times))

    def save(self, cfg):
        cfg['simulation'] = {
            'name': self.name.text(),
            'mode': self.mode.currentText(),
        }
        try:
            times = [float(t.strip()) for t in self.snap_times.text().split(',') if t.strip()]
        except ValueError:
            times = [1e-3, 5e-3, 10e-3]
        cfg.setdefault('mode_settings', {})
        cfg['mode_settings']['solidification'] = {
            'timestep':         self.dt.value(),
            'tracking':         'Volume',
            'output_frequency': self.out_freq.value(),
            'secondary':        False,
        }
        cfg['mode_settings']['snapshots'] = {'times': times}


# ─────────────────────────────── Tab: Material ───────────────────────────────

class MatTab(QWidget):
    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        vl = QVBoxLayout(self)
        vl.setAlignment(Qt.AlignTop)
        vl.setSpacing(4)
        self.T_init = _dbl(923.0,  0, 5000, 1, 10)
        self.T_liq  = _dbl(1928.0, 0, 5000, 1, 10)
        self.k      = _dbl(21.9,   0, 2000, 3, 1)
        self.cp     = _dbl(670.0,  0, 10000, 1, 10)
        self.rho    = _dbl(4420.0, 0, 30000, 1, 100)
        for label, w in [
            ('T_init (K)',  self.T_init),
            ('T_liq (K)',   self.T_liq),
            ('k (W/m·K)',   self.k),
            ('cp (J/kg·K)', self.cp),
            ('rho (kg/m³)', self.rho),
        ]:
            vl.addWidget(_hrow(label, w))
            w.valueChanged.connect(self.changed)
        vl.addStretch()

    def load(self, cfg):
        m = cfg.get('material') or {}
        self.T_init.setValue(_f(m.get('T_init', 923.0)))
        self.T_liq.setValue(_f(m.get('T_liq',  1928.0)))
        self.k.setValue(_f(m.get('k',   21.9)))
        self.cp.setValue(_f(m.get('cp',  670.0)))
        self.rho.setValue(_f(m.get('rho', 4420.0)))

    def save(self, cfg):
        cfg['material'] = {
            'T_init': self.T_init.value(),
            'T_liq':  self.T_liq.value(),
            'k':      self.k.value(),
            'cp':     self.cp.value(),
            'rho':    self.rho.value(),
        }


# ─────────────────────────────── Tab: Beam ───────────────────────────────────

class BeamTab(QWidget):
    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        vl = QVBoxLayout(self)
        vl.setAlignment(Qt.AlignTop)
        vl.setSpacing(4)
        self.power      = _spin(1000, 1, 100000)
        self.efficiency = _dbl(0.90, 0.0, 1.0, 2, 0.01)
        self.width_x    = _dbl(300.0, 1, 10000, 1, 10)   # µm in UI
        self.width_y    = _dbl(300.0, 1, 10000, 1, 10)
        self.depth_z    = _dbl(150.0, 1, 10000, 1, 10)
        for label, w in [
            ('Power (W)',      self.power),
            ('Efficiency',     self.efficiency),
            ('width_x (µm)',   self.width_x),
            ('width_y (µm)',   self.width_y),
            ('depth_z (µm)',   self.depth_z),
        ]:
            vl.addWidget(_hrow(label, w))
            w.valueChanged.connect(self.changed)
        vl.addStretch()

    def load(self, cfg):
        b = cfg.get('beam') or {}
        self.power.setValue(int(_f(b.get('power', 1000))))
        self.efficiency.setValue(_f(b.get('efficiency', 0.90)))
        self.width_x.setValue(_f(b.get('width_x', 300e-6)) * 1e6)
        self.width_y.setValue(_f(b.get('width_y', 300e-6)) * 1e6)
        self.depth_z.setValue(_f(b.get('depth_z', 150e-6)) * 1e6)

    def save(self, cfg):
        cfg['beam'] = {
            'power':      self.power.value(),
            'efficiency': round(self.efficiency.value(), 4),
            'width_x':    self.width_x.value() * 1e-6,
            'width_y':    self.width_y.value() * 1e-6,
            'depth_z':    self.depth_z.value() * 1e-6,
        }


# ─────────────────────────────── Tab: Scan path ──────────────────────────────

class PathTab(QWidget):
    changed = pyqtSignal()
    HEADERS = ['Mode', 'X (mm)', 'Y (mm)', 'Z (mm)', 'Power mod', 'Param']

    def __init__(self):
        super().__init__()
        vl = QVBoxLayout(self)
        info = QLabel(
            'mode 0 = line scan (param = speed m/s)   '
            'mode 1 = spot dwell (param = time s)'
        )
        info.setStyleSheet('color: #888780; font-size: 11px;')
        info.setWordWrap(True)
        vl.addWidget(info)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        vl.addWidget(self.table)
        hl = QHBoxLayout()
        self.btn_add = QPushButton('+ Add row')
        self.btn_del = QPushButton('Remove selected row')
        hl.addWidget(self.btn_add)
        hl.addWidget(self.btn_del)
        hl.addStretch()
        vl.addLayout(hl)
        self.btn_add.clicked.connect(lambda: self._add_row())
        self.btn_del.clicked.connect(self._del_row)
        self.table.itemChanged.connect(self.changed)

    def _add_row(self, data=None):
        row = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.insertRow(row)
        cb = _combo('0 — line scan', '1 — spot dwell')
        if data:
            cb.setCurrentIndex(int(_f(data[0])))
        self.table.setCellWidget(row, 0, cb)
        cb.currentIndexChanged.connect(self.changed)
        defaults = data if data else [0, 0.0, 0.0, 0.0, 1.0, 1.0]
        for col, val in enumerate(defaults[1:], 1):
            it = QTableWidgetItem(f'{val:.6g}')
            it.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, col, it)
        self.table.blockSignals(False)

    def _del_row(self):
        rows = sorted({i.row() for i in self.table.selectedItems()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        if not rows and self.table.rowCount() > 0:
            self.table.removeRow(self.table.rowCount() - 1)
        self.changed.emit()

    def load(self, cfg):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        path = cfg.get('path') or []
        if isinstance(path, list):
            for seg in path:
                if isinstance(seg, list) and len(seg) >= 6:
                    self._add_row(seg)
        self.table.blockSignals(False)

    def save(self, cfg):
        rows = []
        for r in range(self.table.rowCount()):
            cb   = self.table.cellWidget(r, 0)
            mode = cb.currentIndex() if cb else 0
            vals = [mode]
            for c in range(1, 6):
                it = self.table.item(r, c)
                vals.append(_f(it.text() if it else '0'))
            rows.append(vals)
        cfg['path'] = rows


# ─────────────────────────────── Tab: Domain & geometry ──────────────────────

class DomainTab(QWidget):
    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        vl = QVBoxLayout(self)
        vl.setAlignment(Qt.AlignTop)
        vl.setSpacing(4)

        vl.addWidget(_section('Domain bounds (mm)'))
        self.xmin = _dbl(-1.0, -1e4, 1e4, 3, 0.1)
        self.xmax = _dbl(6.0,  -1e4, 1e4, 3, 0.1)
        self.ymin = _dbl(-1.0, -1e4, 1e4, 3, 0.1)
        self.ymax = _dbl(2.0,  -1e4, 1e4, 3, 0.1)
        self.zmin = _dbl(-1.0, -1e4, 1e4, 3, 0.1)
        self.zmax = _dbl(0.0,  -1e4, 1e4, 3, 0.1)
        self.res  = _dbl(50.0, 1, 10000, 1, 10)   # µm
        for label, a, b in [
            ('x range (mm)', self.xmin, self.xmax),
            ('y range (mm)', self.ymin, self.ymax),
            ('z range (mm)', self.zmin, self.zmax),
        ]:
            arrow = QLabel('→')
            arrow.setFixedWidth(14)
            arrow.setAlignment(Qt.AlignCenter)
            vl.addWidget(_hrow(label, a, arrow, b))
        vl.addWidget(_hrow('Resolution (µm)', self.res))

        vl.addWidget(_section('Geometry (part shape)'))
        self.geo_type = _combo('null — no geometry', 'cylinder', 'cubic', 'wall')
        vl.addWidget(_hrow('Type', self.geo_type))

        # cylinder fields
        self.cyl_cx = _dbl(0.0, -1e4, 1e4, 3, 0.1)
        self.cyl_cy = _dbl(0.0, -1e4, 1e4, 3, 0.1)
        self.cyl_r  = _dbl(2.0, 0,    1e4, 3, 0.1)
        self.cyl_widget = QWidget()
        cv = QVBoxLayout(self.cyl_widget)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(4)
        cv.addWidget(_hrow('Center x (mm)', self.cyl_cx))
        cv.addWidget(_hrow('Center y (mm)', self.cyl_cy))
        cv.addWidget(_hrow('Radius (mm)',   self.cyl_r))
        vl.addWidget(self.cyl_widget)

        # cubic fields
        self.cub_x0 = _dbl(0.0,  -1e4, 1e4, 3, 0.1)
        self.cub_x1 = _dbl(5.0,  -1e4, 1e4, 3, 0.1)
        self.cub_y0 = _dbl(0.0,  -1e4, 1e4, 3, 0.1)
        self.cub_y1 = _dbl(1.0,  -1e4, 1e4, 3, 0.1)
        self.cub_z0 = _dbl(-3.0, -1e4, 1e4, 3, 0.1)
        self.cub_z1 = _dbl(0.0,  -1e4, 1e4, 3, 0.1)
        self.cub_widget = QWidget()
        cbv = QVBoxLayout(self.cub_widget)
        cbv.setContentsMargins(0, 0, 0, 0)
        cbv.setSpacing(4)
        for label, a, b in [
            ('geo x range (mm)', self.cub_x0, self.cub_x1),
            ('geo y range (mm)', self.cub_y0, self.cub_y1),
            ('geo z range (mm)', self.cub_z0, self.cub_z1),
        ]:
            arrow = QLabel('→')
            arrow.setFixedWidth(14)
            arrow.setAlignment(Qt.AlignCenter)
            cbv.addWidget(_hrow(label, a, arrow, b))
        vl.addWidget(self.cub_widget)

        self.cyl_widget.hide()
        self.cub_widget.hide()
        vl.addStretch()

        self.geo_type.currentIndexChanged.connect(self._on_geo_type)
        self.geo_type.currentIndexChanged.connect(self.changed)
        for w in [self.xmin, self.xmax, self.ymin, self.ymax,
                  self.zmin, self.zmax, self.res,
                  self.cyl_cx, self.cyl_cy, self.cyl_r,
                  self.cub_x0, self.cub_x1, self.cub_y0,
                  self.cub_y1, self.cub_z0, self.cub_z1]:
            w.valueChanged.connect(self.changed)

    def _on_geo_type(self, idx):
        self.cyl_widget.hide()
        self.cub_widget.hide()
        if idx == 1:
            self.cyl_widget.show()
        elif idx == 2:
            self.cub_widget.show()

    def load(self, cfg):
        dom = cfg.get('domain') or {}
        x = dom.get('x') or [-1e-3, 6e-3]
        y = dom.get('y') or [-1e-3, 2e-3]
        z = dom.get('z') or [-1e-3, 0.0]
        self.xmin.setValue(_f(x[0]) * 1e3)
        self.xmax.setValue(_f(x[1]) * 1e3)
        self.ymin.setValue(_f(y[0]) * 1e3)
        self.ymax.setValue(_f(y[1]) * 1e3)
        self.zmin.setValue(_f(z[0]) * 1e3)
        self.zmax.setValue(_f(z[1]) * 1e3)
        res = dom.get('resolution') or 50e-6
        self.res.setValue(_f(res) * 1e6)

        geo   = dom.get('geometry') or {}
        gtype = geo.get('type') if geo else None
        type_map = {None: 0, 'null': 0, 'cylinder': 1, 'cubic': 2, 'wall': 3}
        self.geo_type.setCurrentIndex(type_map.get(gtype, 0))

        if gtype == 'cylinder':
            c = geo.get('center') or [0, 0]
            self.cyl_cx.setValue(_f(c[0]) * 1e3)
            self.cyl_cy.setValue(_f(c[1]) * 1e3)
            self.cyl_r.setValue(_f(geo.get('radius', 2e-3)) * 1e3)
        elif gtype == 'cubic':
            gx = geo.get('x') or [0, 5e-3]
            gy = geo.get('y') or [0, 1e-3]
            gz = geo.get('z') or [-3e-3, 0]
            self.cub_x0.setValue(_f(gx[0]) * 1e3)
            self.cub_x1.setValue(_f(gx[1]) * 1e3)
            self.cub_y0.setValue(_f(gy[0]) * 1e3)
            self.cub_y1.setValue(_f(gy[1]) * 1e3)
            self.cub_z0.setValue(_f(gz[0]) * 1e3)
            self.cub_z1.setValue(_f(gz[1]) * 1e3)

    def save(self, cfg):
        dom = {
            'x':          [self.xmin.value() * 1e-3, self.xmax.value() * 1e-3],
            'y':          [self.ymin.value() * 1e-3, self.ymax.value() * 1e-3],
            'z':          [self.zmin.value() * 1e-3, self.zmax.value() * 1e-3],
            'resolution': self.res.value() * 1e-6,
        }
        gt = self.geo_type.currentText()
        if gt.startswith('null'):
            dom['geometry'] = {'type': None}
        elif gt == 'cylinder':
            dom['geometry'] = {
                'type':   'cylinder',
                'center': [self.cyl_cx.value() * 1e-3,
                           self.cyl_cy.value() * 1e-3],
                'radius':  self.cyl_r.value() * 1e-3,
            }
        elif gt == 'cubic':
            dom['geometry'] = {
                'type': 'cubic',
                'x': [self.cub_x0.value() * 1e-3, self.cub_x1.value() * 1e-3],
                'y': [self.cub_y0.value() * 1e-3, self.cub_y1.value() * 1e-3],
                'z': [self.cub_z0.value() * 1e-3, self.cub_z1.value() * 1e-3],
            }
        elif gt == 'wall':
            dom['geometry'] = {'type': 'wall'}
        cfg['domain'] = dom


# ─────────────────────────────── Tab: Boundary conditions ────────────────────

class BCTab(QWidget):
    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        vl = QVBoxLayout(self)
        vl.setAlignment(Qt.AlignTop)
        vl.setSpacing(4)

        vl.addWidget(_section('Top surface'))
        self.top_type = _combo('radiation', 'adiabatic')
        self.top_eps  = _dbl(0.70, 0, 1, 2, 0.01)
        self.top_tenv = _dbl(923.0, 0, 5000, 1, 10)
        vl.addWidget(_hrow('Type',        self.top_type))
        self.top_eps_row  = _hrow('Emissivity', self.top_eps)
        self.top_tenv_row = _hrow('T_env (K)',  self.top_tenv)
        vl.addWidget(self.top_eps_row)
        vl.addWidget(self.top_tenv_row)

        vl.addWidget(_section('Side walls'))
        self.side_type = _combo('adiabatic', 'radiation')
        vl.addWidget(_hrow('Type', self.side_type))

        vl.addWidget(_section('Bottom'))
        self.bot_type = _combo('fixed_T', 'adiabatic', 'semi_infinite')
        self.bot_val  = _dbl(923.0, 0, 5000, 1, 10)
        vl.addWidget(_hrow('Type',     self.bot_type))
        self.bot_val_row = _hrow('Value (K)', self.bot_val)
        vl.addWidget(self.bot_val_row)

        vl.addWidget(_section('Image sources'))
        self.reflections = _spin(3, 0, 10)
        vl.addWidget(_hrow('Reflections', self.reflections))
        vl.addStretch()

        self.top_type.currentIndexChanged.connect(self._on_top)
        self.bot_type.currentIndexChanged.connect(self._on_bot)
        for w in [self.top_type, self.side_type, self.bot_type]:
            w.currentIndexChanged.connect(self.changed)
        for w in [self.top_eps, self.top_tenv, self.bot_val]:
            w.valueChanged.connect(self.changed)
        self.reflections.valueChanged.connect(self.changed)

    def _on_top(self, idx):
        show = (idx == 0)
        self.top_eps_row.setVisible(show)
        self.top_tenv_row.setVisible(show)

    def _on_bot(self, idx):
        self.bot_val_row.setVisible(idx == 0)

    def load(self, cfg):
        bcs = cfg.get('boundary_conditions') or {}

        top = bcs.get('top_surface') or {}
        tt  = top.get('type', 'radiation')
        self.top_type.setCurrentIndex(0 if tt == 'radiation' else 1)
        self.top_eps.setValue(_f(top.get('emissivity', 0.70)))
        self.top_tenv.setValue(_f(top.get('T_env', 923.0)))

        side = bcs.get('side_walls') or {}
        st   = side.get('type', 'adiabatic')
        self.side_type.setCurrentIndex(0 if st == 'adiabatic' else 1)

        bot = bcs.get('bottom') or {}
        bt  = bot.get('type', 'fixed_T')
        btypes = ['fixed_T', 'adiabatic', 'semi_infinite']
        self.bot_type.setCurrentIndex(btypes.index(bt) if bt in btypes else 0)
        self.bot_val.setValue(_f(bot.get('value', 923.0)))

        self.reflections.setValue(int(_f(bcs.get('reflections', 3))))
        self._on_top(self.top_type.currentIndex())
        self._on_bot(self.bot_type.currentIndex())

    def save(self, cfg):
        tt  = self.top_type.currentText()
        top = {'type': tt}
        if tt == 'radiation':
            top['emissivity'] = round(self.top_eps.value(), 4)
            top['T_env']      = self.top_tenv.value()
        bt  = self.bot_type.currentText()
        bot = {'type': bt}
        if bt == 'fixed_T':
            bot['value'] = self.bot_val.value()
        cfg['boundary_conditions'] = {
            'top_surface': top,
            'side_walls':  {'type': self.side_type.currentText()},
            'bottom':      bot,
            'reflections': self.reflections.value(),
        }


# ─────────────────────────────── Tab: Output & compute ───────────────────────

class OutputTab(QWidget):
    changed = pyqtSignal()
    ALL_FIELDS = ['x', 'y', 'z', 'T', 'G', 'Gx', 'Gy', 'Gz',
                  'V', 'dTdt', 'tSol', 'numMelt', 'depth']
    DEFAULT_ON = {'x', 'y', 'z', 'T', 'G', 'V', 'dTdt'}

    def __init__(self):
        super().__init__()
        vl = QVBoxLayout(self)
        vl.setAlignment(Qt.AlignTop)
        vl.setSpacing(4)

        vl.addWidget(_section('Output fields'))
        self.checks = {}
        grid_w = QWidget()
        gl = QGridLayout(grid_w)
        gl.setSpacing(4)
        for i, f in enumerate(self.ALL_FIELDS):
            cb = QCheckBox(f)
            cb.setChecked(f in self.DEFAULT_ON)
            cb.toggled.connect(self.changed)
            self.checks[f] = cb
            gl.addWidget(cb, i // 4, i % 4)
        vl.addWidget(grid_w)

        vl.addWidget(_section('Format'))
        self.vtk = QCheckBox('Write VTK (.vtr) for ParaView')
        self.vtk.setChecked(True)
        self.vtk.toggled.connect(self.changed)
        vl.addWidget(self.vtk)
        self.outdir = QLineEdit('Data')
        btn_browse  = QPushButton('Browse…')
        btn_browse.clicked.connect(self._browse)
        vl.addWidget(_hrow('Directory', self.outdir, btn_browse))

        vl.addWidget(_section('Compute'))
        self.threads = _spin(4, 1, 64)
        self.compress = QCheckBox('Path compression')
        self.compress.setChecked(True)
        self.t_hist = QLineEdit('1.0e-9')
        self.p_hist = QLineEdit('1.0e-2')
        vl.addWidget(_hrow('Threads',  self.threads))
        vl.addWidget(_hrow('',         self.compress))
        vl.addWidget(_hrow('t_hist',   self.t_hist))
        vl.addWidget(_hrow('p_hist',   self.p_hist))
        vl.addStretch()

        self.outdir.textChanged.connect(self.changed)
        self.threads.valueChanged.connect(self.changed)
        self.compress.toggled.connect(self.changed)
        self.t_hist.textChanged.connect(self.changed)
        self.p_hist.textChanged.connect(self.changed)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, 'Select output directory', ROOT)
        if d:
            self.outdir.setText(d)

    def load(self, cfg):
        out = cfg.get('output') or {}
        fields = out.get('fields') or list(self.DEFAULT_ON)
        for f, cb in self.checks.items():
            cb.setChecked(f in fields)
        self.vtk.setChecked(bool(out.get('vtk', True)))
        self.outdir.setText(str(out.get('directory', 'Data')))
        cmp = cfg.get('compute') or {}
        self.threads.setValue(int(_f(cmp.get('threads', 4))))
        self.compress.setChecked(bool(cmp.get('compression', True)))
        self.t_hist.setText(str(cmp.get('t_hist', '1.0e-9')))
        self.p_hist.setText(str(cmp.get('p_hist', '1.0e-2')))

    def save(self, cfg):
        cfg['output'] = {
            'directory': self.outdir.text(),
            'fields':    [f for f, cb in self.checks.items() if cb.isChecked()],
            'vtk':       self.vtk.isChecked(),
        }
        cfg['compute'] = {
            'threads':     self.threads.value(),
            'compression': self.compress.isChecked(),
            'r_max':       -1.0,
            't_hist':      _f(self.t_hist.text(), 1e-9),
            'p_hist':      _f(self.p_hist.text(), 1e-2),
        }


# ─────────────────────────────── Summary panel ───────────────────────────────

class SummaryPanel(QWidget):
    def __init__(self):
        super().__init__()
        vl = QVBoxLayout(self)
        vl.setContentsMargins(6, 6, 6, 4)
        vl.setSpacing(2)
        vl.addWidget(_section('Summary'))
        self._vals = {}
        for key, label in [
            ('mode',     'Mode'),
            ('grid',     'Grid'),
            ('points',   'Total pts'),
            ('active',   'Active'),
            ('geometry', 'Geometry'),
            ('est_time', 'Est. time'),
        ]:
            row = QWidget()
            hl  = QHBoxLayout(row)
            hl.setContentsMargins(0, 1, 0, 1)
            hl.setSpacing(4)
            hl.addWidget(QLabel(f'<b>{label}</b>'))
            hl.addStretch()
            val = QLabel('—')
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._vals[key] = val
            hl.addWidget(val)
            vl.addWidget(row)
        vl.addStretch()

    def refresh(self, cfg):
        dom   = cfg.get('domain') or {}
        geo   = dom.get('geometry') or {}
        gtype = geo.get('type') if geo else None

        x   = dom.get('x') or [-1e-3, 6e-3]
        y   = dom.get('y') or [-1e-3, 2e-3]
        z   = dom.get('z') or [-1e-3, 0.0]
        res = _f(dom.get('resolution') or 50e-6)
        nx  = max(1, round(abs(_f(x[1]) - _f(x[0])) / res)) + 1
        ny  = max(1, round(abs(_f(y[1]) - _f(y[0])) / res)) + 1
        nz  = max(1, round(abs(_f(z[1]) - _f(z[0])) / res)) + 1
        n   = nx * ny * nz

        self._vals['mode'].setText(
            (cfg.get('simulation') or {}).get('mode', '—'))
        self._vals['grid'].setText(f'{nx} x {ny} x {nz}')
        self._vals['points'].setText(f'{n:,}')
        self._vals['geometry'].setText(str(gtype) if gtype else 'none')

        if gtype == 'cylinder':
            cx  = _f((geo.get('center') or [0, 0])[0])
            cy  = _f((geo.get('center') or [0, 0])[1])
            r   = _f(geo.get('radius', 2e-3))
            xmn, xmx = _f(x[0]), _f(x[1])
            ymn, ymx = _f(y[0]), _f(y[1])
            dom_area  = (xmx - xmn) * (ymx - ymn)
            cyl_area  = 3.14159 * r ** 2
            frac      = min(1.0, cyl_area / dom_area) if dom_area > 0 else 1.0
            self._vals['active'].setText(f'{frac * 100:.0f}%')
        else:
            self._vals['active'].setText('100%')

        # very rough run-time estimate
        nseg = len(cfg.get('path') or [])
        refl = _f((cfg.get('boundary_conditions') or {}).get('reflections', 3))
        dt   = _f(((cfg.get('mode_settings') or {}).get('solidification') or {})
                  .get('timestep', 1e-4))
        est  = n * nseg * max(1, 2 ** int(refl)) * dt * 6e-4  # rough constant
        if est < 60:
            est_str = f'~{est:.0f} s'
        elif est < 3600:
            est_str = f'~{est / 60:.0f} min'
        else:
            est_str = f'~{est / 3600:.1f} h'
        self._vals['est_time'].setText(est_str)


# ─────────────────────────────── Main window ─────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, config_path=None):
        super().__init__()
        self.setWindowTitle('EBM Thermal Solver — Config')
        self.resize(950, 720)
        self.config_path = config_path or DEFAULT_CONFIG
        self.process     = None

        cw = QWidget()
        self.setCentralWidget(cw)
        mvl = QVBoxLayout(cw)
        mvl.setContentsMargins(8, 8, 8, 8)
        mvl.setSpacing(6)

        # ── top splitter: tabs | right panel ────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)

        # Left: tab widget
        self.tabs = QTabWidget()
        self.t_sim    = SimTab()
        self.t_mat    = MatTab()
        self.t_beam   = BeamTab()
        self.t_path   = PathTab()
        self.t_domain = DomainTab()
        self.t_bc     = BCTab()
        self.t_output = OutputTab()
        for title, tab in [
            ('Simulation',          self.t_sim),
            ('Material',            self.t_mat),
            ('Beam',                self.t_beam),
            ('Scan path',           self.t_path),
            ('Domain & geometry',   self.t_domain),
            ('Boundary conditions', self.t_bc),
            ('Output & compute',    self.t_output),
        ]:
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setWidget(tab)
            sa.setFrameShape(QFrame.NoFrame)
            self.tabs.addTab(sa, title)

        splitter.addWidget(self.tabs)

        # Right: summary + canvas
        right = QWidget()
        right.setMinimumWidth(240)
        right.setMaximumWidth(320)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 0, 0, 0)
        rl.setSpacing(4)
        self.summary = SummaryPanel()
        self.canvas  = PreviewCanvas()
        rl.addWidget(self.summary)
        rl.addWidget(self.canvas, stretch=1)
        splitter.addWidget(right)
        splitter.setSizes([620, 280])
        mvl.addWidget(splitter, stretch=1)

        # ── bottom: log + buttons ────────────────────────────────────────────
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(130)
        self.log.setFont(QFont('Courier New', 9))
        mvl.addWidget(self.log)

        btn_bar = QWidget()
        bl = QHBoxLayout(btn_bar)
        bl.setContentsMargins(0, 0, 0, 0)
        self.btn_load = QPushButton('Load config…')
        self.btn_save = QPushButton('Save config…')
        self.btn_run  = QPushButton('Run simulation  ▶')
        self.btn_run.setStyleSheet(
            'background-color: #378ADD; color: white; font-weight: 500;'
            'padding: 4px 16px; border-radius: 4px;'
        )
        bl.addWidget(self.btn_load)
        bl.addWidget(self.btn_save)
        bl.addStretch()
        bl.addWidget(self.btn_run)
        mvl.addWidget(btn_bar)

        # ── signals ──────────────────────────────────────────────────────────
        self.tabs.currentChanged.connect(self._refresh_preview)
        for tab in [self.t_sim, self.t_mat, self.t_beam, self.t_path,
                    self.t_domain, self.t_bc, self.t_output]:
            tab.changed.connect(self._on_changed)
        self.btn_load.clicked.connect(self._load_dialog)
        self.btn_save.clicked.connect(self._save_dialog)
        self.btn_run.clicked.connect(self._run)

        # ── load initial config ───────────────────────────────────────────────
        self._load_config(self.config_path)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _all_tabs(self):
        return [self.t_sim, self.t_mat, self.t_beam, self.t_path,
                self.t_domain, self.t_bc, self.t_output]

    def _collect(self):
        cfg = {}
        for tab in self._all_tabs():
            tab.save(cfg)
        return cfg

    def _on_changed(self):
        cfg = self._collect()
        self.summary.refresh(cfg)
        self._refresh_preview()

    def _refresh_preview(self):
        cfg = self._collect()
        idx = self.tabs.currentIndex()
        if idx == 4:    # Domain & geometry
            self.canvas.show_geometry(cfg)
        elif idx == 5:  # Boundary conditions
            self.canvas.show_bc(cfg)
        else:
            self.canvas.show_placeholder()

    def _load_config(self, path):
        if not os.path.isfile(path):
            self._log(f'Config not found: {path}', error=True)
            return
        try:
            with open(path, encoding='utf-8') as fh:
                cfg = yaml.safe_load(fh) or {}
            for tab in self._all_tabs():
                tab.load(cfg)
            self.config_path = path
            self._log(f'Loaded: {path}')
            self._on_changed()
        except Exception as exc:
            self._log(f'Error loading config: {exc}', error=True)

    def _save_config(self, path):
        cfg = self._collect()
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                yaml.dump(cfg, fh, default_flow_style=False,
                          allow_unicode=True, sort_keys=False)
            self.config_path = path
            self._log(f'Saved: {path}')
        except Exception as exc:
            self._log(f'Error saving: {exc}', error=True)

    def _load_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load config', os.path.join(ROOT, 'inputs'), 'YAML (*.yaml *.yml)')
        if path:
            self._load_config(path)

    def _save_dialog(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save config', self.config_path, 'YAML (*.yaml *.yml)')
        if path:
            self._save_config(path)

    def _run(self):
        if self.process and self.process.state() != QProcess.NotRunning:
            self.process.kill()
            self._log('Run stopped by user.', error=True)
            self._reset_run_btn()
            return
        self._save_config(self.config_path)
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.readyReadStandardError.connect(self._on_stderr)
        self.process.finished.connect(self._on_finished)
        self.process.start(sys.executable, [MAIN_SCRIPT, self.config_path])
        self._log(f'--- python main.py {self.config_path} ---')
        self.btn_run.setText('Stop  ■')
        self.btn_run.setStyleSheet(
            'background-color: #E24B4A; color: white; font-weight: 500;'
            'padding: 4px 16px; border-radius: 4px;'
        )

    def _on_stdout(self):
        data = self.process.readAllStandardOutput().data().decode('utf-8', 'replace')
        for line in data.splitlines():
            self.log.append(line)

    def _on_stderr(self):
        data = self.process.readAllStandardError().data().decode('utf-8', 'replace')
        for line in data.splitlines():
            self.log.append(f'<span style="color:#E24B4A">{line}</span>')

    def _on_finished(self, code, _status):
        self._log(f'--- Finished (exit code {code}) ---')
        self._reset_run_btn()

    def _reset_run_btn(self):
        self.btn_run.setText('Run simulation  ▶')
        self.btn_run.setStyleSheet(
            'background-color: #378ADD; color: white; font-weight: 500;'
            'padding: 4px 16px; border-radius: 4px;'
        )

    def _log(self, msg, error=False):
        if error:
            self.log.append(f'<span style="color:#E24B4A">{msg}</span>')
        else:
            self.log.append(msg)


# ─────────────────────────────── entry point ─────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    config = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    win = MainWindow(config)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
