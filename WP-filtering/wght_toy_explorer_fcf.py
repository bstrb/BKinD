#!/usr/bin/env python3
"""
WGHT toy explorer (FCF-backed).
Loads a SHELXL .fcf (hklf 4) to get Fo^2, Fc^2, sigma(Fo^2), hkl, and cell (if present);
computes s = sin(theta)/lambda directly from hkl and cell;
plots q(s), weights, relative weights, and chi terms for one-or-more WGHT parameter sets.
Parameters per curve: a, b, c, f. Sigma/Fc come from the loaded file.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Dict
import shlex
import sys
import os

# Ensure Qt can start on macOS/WSL without xcb. Prefer cocoa on macOS if not set.
if "QT_QPA_PLATFORM" not in os.environ:
    if sys.platform == "darwin":
        os.environ["QT_QPA_PLATFORM"] = "cocoa"
    else:
        # Fallback to offscreen for headless cases; can be overridden by user env
        # os.environ["QT_QPA_PLATFORM"] = "offscreen"
        pass
import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QColor, QBrush
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

# -----------------
# CIF/FCF helpers
# -----------------
def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lowmap = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        c2 = cand.lower()
        if c2 in lowmap:
            return lowmap[c2]
    return None

def _read_cell_from_cif(lines: List[str]) -> Dict[str, float]:
    tags = {
        "_cell_length_a": "a",
        "_cell_length_b": "b",
        "_cell_length_c": "c",
        "_cell_angle_alpha": "alpha",
        "_cell_angle_beta": "beta",
        "_cell_angle_gamma": "gamma",
    }
    cell: Dict[str, float] = {}
    for ln in lines:
        s = ln.strip()
        if not s.startswith("_cell_"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        key = parts[0]
        if key in tags:
            try:
                cell[tags[key]] = float(parts[1])
            except Exception:
                continue
    return cell


def _unit_cell_reciprocal_metric(cell: Dict[str, float]) -> Tuple[float, float, float, float, float, float]:
    a = float(cell["a"]); b = float(cell["b"]); c = float(cell["c"])
    alpha = np.deg2rad(float(cell["alpha"])); beta = np.deg2rad(float(cell["beta"])); gamma = np.deg2rad(float(cell["gamma"]))
    cos_a = np.cos(alpha); cos_b = np.cos(beta); cos_g = np.cos(gamma)
    sin_g = np.sin(gamma)
    V = a * b * c * np.sqrt(1 - cos_a**2 - cos_b**2 - cos_g**2 + 2 * cos_a * cos_b * cos_g)
    astar = b * c * sin_g / V
    # use conventional reciprocal metrics
    astar = (b * c * np.sin(alpha)) / V
    bstar = (a * c * np.sin(beta)) / V
    cstar = (a * b * np.sin(gamma)) / V
    cos_alpha_star = (cos_b * cos_g - cos_a) / (np.sin(beta) * np.sin(gamma))
    cos_beta_star = (cos_a * cos_g - cos_b) / (np.sin(alpha) * np.sin(gamma))
    cos_gamma_star = (cos_a * cos_b - cos_g) / (np.sin(alpha) * np.sin(beta))
    return astar, bstar, cstar, cos_alpha_star, cos_beta_star, cos_gamma_star


def _sin_theta_over_lambda(hkl: np.ndarray, cell: Dict[str, float]) -> np.ndarray:
    h = hkl[:, 0]; k = hkl[:, 1]; l = hkl[:, 2]
    astar, bstar, cstar, cos_a, cos_b, cos_g = _unit_cell_reciprocal_metric(cell)
    # reciprocal metric tensor G*
    G11 = astar**2
    G22 = bstar**2
    G33 = cstar**2
    G12 = astar * bstar * cos_g
    G13 = astar * cstar * cos_b
    G23 = bstar * cstar * cos_a
    s2 = (G11 * h * h + G22 * k * k + G33 * l * l
          + 2 * G12 * h * k + 2 * G13 * h * l + 2 * G23 * k * l)
    s = np.sqrt(np.maximum(s2, 0.0))
    return 0.5 * s  # s = sin(theta)/lambda = |h*| / 2


def load_fcf(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[Dict[str, float]]]:
    text = path.read_text(errors="replace").splitlines()
    cell = _read_cell_from_cif(text)
    n = len(text)
    i = 0
    while i < n:
        if text[i].strip().lower() != "loop_":
            i += 1; continue
        cols: List[str] = []
        j = i + 1
        while j < n:
            t = text[j].strip()
            if not t:
                j += 1; continue
            if t.startswith("_"):
                cols.append(t.split()[0])
                j += 1; continue
            break
        if not cols or not any(c.lower().startswith("_refln_") for c in cols):
            i = j; continue
        rows: List[List[str]] = []
        k = j
        while k < n:
            raw = text[k].strip()
            if not raw or raw.startswith("#"):
                k += 1; continue
            low = raw.lower()
            if low == "loop_" or low.startswith("data_") or low == "stop_" or raw.startswith("_"):
                break
            toks = shlex.split(raw)
            kk = k
            while len(toks) < len(cols) and kk + 1 < n:
                kk += 1
                nxt = text[kk].strip()
                if not nxt or nxt.startswith("#"):
                    continue
                nxtlow = nxt.lower()
                if nxtlow == "loop_" or nxtlow.startswith("data_") or nxtlow == "stop_" or nxt.startswith("_"):
                    break
                toks += shlex.split(nxt)
            if len(toks) >= len(cols):
                rows.append(toks[:len(cols)])
            k = kk + 1
        # parse
        col_h = _pick_col(cols, ["_refln_index_h", "_refln_h"])
        col_k = _pick_col(cols, ["_refln_index_k", "_refln_k"])
        col_l = _pick_col(cols, ["_refln_index_l", "_refln_l"])
        col_Fo2 = _pick_col(cols, ["_refln_F_squared_meas", "_refln_F_squared_meas_au"])
        col_sig = _pick_col(cols, ["_refln_F_squared_sigma", "_refln_F_squared_sigma_au"])
        col_Fc2 = _pick_col(cols, ["_refln_F_squared_calc", "_refln_F_squared_calc_au"])
        need = [col_h, col_k, col_l, col_Fo2, col_sig, col_Fc2]
        if any(c is None for c in need):
            raise ValueError("Missing hkl/Fo^2/Fc^2/sigma columns in FCF")
        idx = {c: cols.index(c) for c in need}
        Fo2 = np.zeros(len(rows), float)
        Fc2 = np.zeros(len(rows), float)
        sig = np.zeros(len(rows), float)
        hkl = np.zeros((len(rows), 3), int)
        for ii, r in enumerate(rows):
            try:
                hkl[ii, 0] = int(float(r[idx[col_h]]))
                hkl[ii, 1] = int(float(r[idx[col_k]]))
                hkl[ii, 2] = int(float(r[idx[col_l]]))
                Fo2[ii] = float(r[idx[col_Fo2]])
                Fc2[ii] = float(r[idx[col_Fc2]])
                sig[ii] = float(r[idx[col_sig]])
            except Exception:
                Fo2[ii] = np.nan; Fc2[ii] = np.nan; sig[ii] = np.nan; hkl[ii, :] = 0
        mask = np.isfinite(Fo2) & np.isfinite(Fc2) & np.isfinite(sig) & (sig > 0)
        Fo2 = Fo2[mask]; Fc2 = Fc2[mask]; sig = sig[mask]; hkl = hkl[mask]
        return Fo2, Fc2, sig, (hkl if hkl.size else None), (cell if cell else None)
    raise ValueError("No _refln_ loop found in FCF")

# -----------------
# Core math helpers
# -----------------
def weights_wght(Fo2: np.ndarray, Fc2: np.ndarray, sigma2: np.ndarray, s: np.ndarray,
                 a: float, b: float, c: float, f: float) -> np.ndarray:
    # SHELXL-like q(c): q = 1, exp(c*s^2), or 1-exp(c*s^2)
    c = float(c)
    if c == 0.0:
        q = 1.0
    elif c > 0.0:
        q = np.exp(c * s * s)
    else:
        q = 1.0 - np.exp(c * s * s)
    P = float(f) * np.maximum(Fo2, 0.1) + (1.0 - float(f)) * np.maximum(Fc2, 0.1)
    denom = sigma2 + (float(a) * P) ** 2 + float(b) * P  # d=e=0
    return q / np.maximum(denom, 1e-30)


def bin_logx_median(x: np.ndarray, y: np.ndarray, nbins: int = 80):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[ok]; y = y[ok]
    if x.size == 0:
        return np.array([]), np.array([])
    xmin = np.min(x); xmax = np.max(x)
    if xmin <= 0 or xmax <= 0 or xmin == xmax:
        return np.array([]), np.array([])
    edges = np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)
    xc = np.sqrt(edges[:-1] * edges[1:])
    yy = np.full_like(xc, np.nan, float)
    for i in range(len(xc)):
        if i == len(xc) - 1:
            m = (x >= edges[i]) & (x <= edges[i+1])
        else:
            m = (x >= edges[i]) & (x < edges[i+1])
        if np.any(m):
            yy[i] = np.nanmedian(y[m])
        elif i > 0 and np.any((x >= edges[i-1]) & (x < edges[i])):
            # carry forward last value if gap, to avoid dropping the tail entirely
            yy[i] = yy[i-1]
    ok2 = np.isfinite(yy)
    return xc[ok2], yy[ok2]


def plt_cm(n: int):
    import matplotlib.pyplot as plt
    return plt.cm.tab20(np.linspace(0, 1, max(n, 1)))


class FcfWghtExplorer(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("WGHT toy explorer (FCF)")

        self.Fo2: Optional[np.ndarray] = None
        self.Fc2: Optional[np.ndarray] = None
        self.sigFo2: Optional[np.ndarray] = None
        self.hkl: Optional[np.ndarray] = None
        self.cell: Optional[Dict[str, float]] = None

        self.lambda_presets = [
            ("Electron 300 kV", 0.01968),
            ("Electron 200 kV", 0.02508),
            ("Mo Kα (X-ray)", 0.71073),
            ("Cu Kα (X-ray)", 1.54056),
            ("Custom", None),
        ]

        self.log_defaults = {
            "w_p": True,
            "relw_p": True,
            "chi_p": True,
            "w_d": True,
            "relw_d": True,
            "chi_d": True,
        }
        self.style_defaults = {
            "w_p": "binned",
            "relw_p": "binned",
            "chi_p": "binned",
            "w_d": "binned",
            "relw_d": "binned",
            "chi_d": "binned",
        }
        self.bin_defaults = {
            "w_p": 100,
            "relw_p": 100,
            "chi_p": 100,
            "w_d": 100,
            "relw_d": 100,
            "chi_d": 100,
        }

        self.default_params = {"a": 0.1, "b": 0.0, "c": 0.0, "f": 1/3}
        self._suspend_plot = False

        self._build_ui()
        self.add_curve_row()

    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)

        # Figure
        self.fig = Figure(figsize=(12.0, 8.0))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.toolbar)
        left.addWidget(self.canvas)
        left_widget = QtWidgets.QWidget(); left_widget.setLayout(left)
        left_widget.setMinimumWidth(500)

        # Controls
        controls_widget = QtWidgets.QWidget(); controls_widget.setMinimumWidth(360)
        form = QtWidgets.QFormLayout(controls_widget)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        self.btn_load = QtWidgets.QPushButton("Load .fcf")
        self.btn_load.clicked.connect(self.load_fcf_dialog)
        form.addRow(self.btn_load)

        # Lambda presets + custom
        self.combo_lambda = QtWidgets.QComboBox()
        for name, val in self.lambda_presets:
            label = f"{name}" if val is None else f"{name} ({val:.5f} Å)"
            self.combo_lambda.addItem(label, val)
        self.combo_lambda.currentIndexChanged.connect(self._on_lambda_preset_changed)

        self.spin_lambda = QtWidgets.QDoubleSpinBox()
        self.spin_lambda.setRange(1e-4, 10.0)
        self.spin_lambda.setDecimals(6)
        self.spin_lambda.setValue(0.71073)  # default Mo Kα
        self.spin_lambda.editingFinished.connect(self._on_lambda_spin_changed)
        self.spin_lambda.setEnabled(False)

        lambda_row = QtWidgets.QHBoxLayout()
        lambda_row.addWidget(self.combo_lambda)
        lambda_row.addWidget(self.spin_lambda)
        lambda_row_widget = QtWidgets.QWidget(); lambda_row_widget.setLayout(lambda_row)
        form.addRow("λ (Å)", lambda_row_widget)

        # Curve table
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["on", "a", "b", "c", "f", "label"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        # Default widths; last column stretches
        self.table.setColumnWidth(0, 20)
        for col in (1, 2, 3, 4):
            self.table.setColumnWidth(col, 30)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.AllEditTriggers)
        self.table.setDragEnabled(True)
        self.table.setAcceptDrops(True)
        self.table.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.table.itemChanged.connect(self._on_table_changed)
        form.addRow("Curves", self.table)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add curve")
        self.btn_add.clicked.connect(self.add_curve_row)
        self.btn_remove = QtWidgets.QPushButton("Remove selected")
        self.btn_remove.clicked.connect(self.remove_selected_row)
        self.btn_up = QtWidgets.QPushButton("Move up")
        self.btn_up.clicked.connect(self.move_row_up)
        self.btn_down = QtWidgets.QPushButton("Move down")
        self.btn_down.clicked.connect(self.move_row_down)
        self.btn_reset = QtWidgets.QPushButton("Reset curves")
        self.btn_reset.clicked.connect(self.reset_curves)
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_remove)
        btn_row.addWidget(self.btn_up)
        btn_row.addWidget(self.btn_down)
        btn_row.addWidget(self.btn_reset)
        btn_row_widget = QtWidgets.QWidget(); btn_row_widget.setLayout(btn_row)
        form.addRow(btn_row_widget)

        # Per-plot style, bins, and log toggles
        self.style_combos: Dict[str, QtWidgets.QComboBox] = {}
        self.bin_spins: Dict[str, QtWidgets.QSpinBox] = {}
        self.log_checks: Dict[str, QtWidgets.QCheckBox] = {}
        style_keys = [
            ("w vs P", "w_p"),
            ("rel w vs P", "relw_p"),
            ("chi vs P", "chi_p"),
            ("w vs d", "w_d"),
            ("rel w vs d", "relw_d"),
            ("chi vs d", "chi_d"),
        ]
        style_grid = QtWidgets.QGridLayout()
        for i, (label, key) in enumerate(style_keys):
            combo = QtWidgets.QComboBox()
            combo.addItems(["binned", "scatter", "both"])
            combo.setCurrentText(self.style_defaults.get(key, "binned"))
            combo.currentIndexChanged.connect(self.update_plots)
            self.style_combos[key] = combo
            spin = QtWidgets.QSpinBox()
            spin.setRange(0, 500)
            spin.setSingleStep(5)
            spin.setValue(self.bin_defaults.get(key, 80))
            spin.setToolTip("Bins (0 = off)")
            spin.valueChanged.connect(self.update_plots)
            self.bin_spins[key] = spin
            log_chk = QtWidgets.QCheckBox("log")
            log_chk.setChecked(self.log_defaults.get(key, True))
            log_chk.stateChanged.connect(self.update_plots)
            self.log_checks[key] = log_chk
            style_grid.addWidget(QtWidgets.QLabel(label), i, 0)
            style_grid.addWidget(combo, i, 1)
            style_grid.addWidget(spin, i, 2)
            style_grid.addWidget(log_chk, i, 3)
        style_group = QtWidgets.QGroupBox("Plot style / bins / log")
        style_group.setLayout(style_grid)
        form.addRow(style_group)

        # Status
        self.status_box = QtWidgets.QPlainTextEdit(); self.status_box.setReadOnly(True); self.status_box.setFixedHeight(160)
        form.addRow("Status", self.status_box)

        # Splitter
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(10)
        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(controls_widget)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setSizes([1000, 500])
        self.splitter.setStyleSheet("QSplitter::handle { background: #7f8c8d; border: 1px solid #555; margin: 0 2px; }")

        layout.addWidget(self.splitter)

        self._setup_axes()

    def _setup_axes(self) -> None:
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.22)
        # top row: w/relw/chi vs P
        self.ax_w_p = self.fig.add_subplot(gs[0, 0])
        self.ax_relw_p = self.fig.add_subplot(gs[0, 1])
        self.ax_chi_p = self.fig.add_subplot(gs[0, 2])
        # bottom row: w/relw/chi vs d
        self.ax_w_d = self.fig.add_subplot(gs[1, 0])
        self.ax_relw_d = self.fig.add_subplot(gs[1, 1])
        self.ax_chi_d = self.fig.add_subplot(gs[1, 2])
        for ax in (self.ax_w_p, self.ax_relw_p, self.ax_chi_p, self.ax_w_d, self.ax_relw_d, self.ax_chi_d):
            ax.grid(True, which="both", alpha=0.25)
        self.ax_w_p.set_ylabel("w")
        self.ax_relw_p.set_ylabel("relative w = w / (1/σ²)")
        self.ax_chi_p.set_ylabel("chi = w·(Fo² - Fc²)²")
        self.ax_w_d.set_ylabel("w")
        self.ax_relw_d.set_ylabel("relative w = w / (1/σ²)")
        self.ax_chi_d.set_ylabel("chi = w·(Fo² - Fc²)²")
        self.ax_w_p.set_xlabel("P")
        self.ax_relw_p.set_xlabel("P")
        self.ax_chi_p.set_xlabel("P")
        self.ax_w_d.set_xlabel("resolution d (Å)")
        self.ax_relw_d.set_xlabel("resolution d (Å)")
        self.ax_chi_d.set_xlabel("resolution d (Å)")

    def add_curve_row(self, overrides: Optional[dict] = None) -> None:
        params = {**self.default_params}
        if overrides:
            params.update(overrides)
        self._suspend_plot = True
        row = self.table.rowCount()
        self.table.insertRow(row)
        # on/off checkbox
        on_item = QtWidgets.QTableWidgetItem()
        on_item.setFlags(on_item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        on_item.setCheckState(QtCore.Qt.CheckState.Checked)
        on_item.setTextAlignment(int(QtCore.Qt.AlignmentFlag.AlignCenter))
        self.table.setItem(row, 0, on_item)

        vals = [params["a"], params["b"], params["c"], params["f"]]
        for col, v in enumerate(vals, start=1):
            item = QtWidgets.QTableWidgetItem(f"{v:.6g}")
            item.setTextAlignment(int(QtCore.Qt.AlignmentFlag.AlignCenter))
            self.table.setItem(row, col, item)
        label_item = QtWidgets.QTableWidgetItem("")
        label_item.setTextAlignment(int(QtCore.Qt.AlignmentFlag.AlignCenter))
        label_item.setFlags(label_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, 5, label_item)
        self._suspend_plot = False
        self.status_box.setPlainText(f"Added curve row #{row+1}")
        self.update_plots()

    def remove_selected_row(self) -> None:
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            self.status_box.setPlainText("Select a row to remove.")
            return
        self._suspend_plot = True
        for idx in sorted(sel, key=lambda x: x.row(), reverse=True):
            self.table.removeRow(idx.row())
        self._suspend_plot = False
        self.status_box.setPlainText("Selected rows removed.")
        self.update_plots()

    def move_row_up(self) -> None:
        sel = self.table.selectionModel().selectedRows()
        if len(sel) != 1:
            self.status_box.setPlainText("Select one row to move.")
            return
        row = sel[0].row()
        if row <= 0:
            return
        self._suspend_plot = True
        self.table.insertRow(row - 1)
        for col in range(self.table.columnCount()):
            item = self.table.takeItem(row + 1, col)
            self.table.setItem(row - 1, col, item)
        self.table.removeRow(row + 1)
        self.table.selectRow(row - 1)
        self._suspend_plot = False
        self.update_plots()

    def move_row_down(self) -> None:
        sel = self.table.selectionModel().selectedRows()
        if len(sel) != 1:
            self.status_box.setPlainText("Select one row to move.")
            return
        row = sel[0].row()
        if row >= self.table.rowCount() - 1:
            return
        self._suspend_plot = True
        self.table.insertRow(row + 2)
        for col in range(self.table.columnCount()):
            item = self.table.takeItem(row, col)
            self.table.setItem(row + 2, col, item)
        self.table.removeRow(row)
        self.table.selectRow(row + 1)
        self._suspend_plot = False
        self.update_plots()

    def reset_curves(self) -> None:
        self.table.setRowCount(0)
        self.add_curve_row()
        self.update_plots()

    def _on_table_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._suspend_plot:
            return
        self.update_plots()

    def _on_lambda_preset_changed(self, idx: int) -> None:
        val = self.combo_lambda.itemData(idx)
        if val is None:
            # Custom: enable manual entry but don't overwrite current value
            self.spin_lambda.setEnabled(True)
            return
        self.spin_lambda.setEnabled(False)
        self.spin_lambda.setValue(float(val))
        self.update_plots()

    def _on_lambda_spin_changed(self) -> None:
        idx = self.combo_lambda.currentIndex()
        val = self.combo_lambda.itemData(idx)
        if val is not None:
            # a preset is selected; ignore manual edit
            return
        self.spin_lambda.setEnabled(True)
        self.update_plots()

    def _read_curves(self) -> List[dict]:
        curves: List[dict] = []
        for row in range(self.table.rowCount()):
            try:
                on_item = self.table.item(row, 0)
                if on_item is None or on_item.checkState() != QtCore.Qt.CheckState.Checked:
                    continue
                a = float(self.table.item(row, 1).text())
                b = float(self.table.item(row, 2).text())
                c = float(self.table.item(row, 3).text())
                f = float(self.table.item(row, 4).text())
            except Exception:
                continue
            curves.append({"a": a, "b": b, "c": c, "f": f})
        return curves

    def load_fcf_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select .fcf", str(Path.cwd()), "FCF files (*.fcf);;All files (*)")
        if not path:
            return
        self._load_fcf(Path(path))

    def _load_fcf(self, path: Path) -> None:
        try:
            Fo2, Fc2, sig, hkl, cell = load_fcf(path)
        except Exception as exc:
            self.status_box.setPlainText(f"Failed to load {path.name}: {exc}")
            return
        self.Fo2 = Fo2
        self.Fc2 = Fc2
        self.sigFo2 = sig
        self.hkl = hkl
        self.cell = cell
        cell_txt = "; cell ok" if cell else "; cell missing"
        hkl_txt = "; hkl ok" if hkl is not None else "; hkl missing"
        self.status_box.setPlainText(f"Loaded {path.name} (n={len(Fo2)} reflections{cell_txt}{hkl_txt})")
        self._set_bin_spin_limits(len(Fo2))
        self.update_plots()

    def _set_bin_spin_limits(self, n_reflections: int) -> None:
        max_bins = max(2, int(n_reflections // 2)) if n_reflections else 2
        for spin in self.bin_spins.values():
            spin.setMaximum(max_bins)
            if spin.value() > max_bins:
                spin.setValue(max_bins)

    def update_plots(self) -> None:
        Fo2 = self.Fo2; Fc2 = self.Fc2; sigFo2 = self.sigFo2; hkl = self.hkl; cell = self.cell
        curves = self._read_curves()

        def is_log(key: str) -> bool:
            chk = self.log_checks.get(key)
            return chk.isChecked() if chk else True

        def style_mode(key: str) -> str:
            combo = self.style_combos.get(key)
            val = combo.currentText().strip().lower() if combo else "binned"
            return val if val in ("binned", "scatter", "both") else "binned"

        def bin_count(key: str) -> int:
            spin = self.bin_spins.get(key)
            if spin is None:
                return 0
            try:
                return int(spin.value())
            except Exception:
                return 0

        def nb_for(key: str) -> int:
            spin = self.bin_spins.get(key)
            n = bin_count(key)
            if n <= 0:
                return 0
            max_allowed = spin.maximum() if spin else n
            return int(max(2, min(max_allowed, n)))

        axes = {
            "w_p": self.ax_w_p,
            "relw_p": self.ax_relw_p,
            "chi_p": self.ax_chi_p,
            "w_d": self.ax_w_d,
            "relw_d": self.ax_relw_d,
            "chi_d": self.ax_chi_d,
        }
        for key, ax in axes.items():
            ax.cla()
            log_on = is_log(key)
            ax.set_xscale("log" if log_on else "linear")
            ax.set_yscale("log" if log_on else "linear")
            ax.grid(True, which="both", alpha=0.25)
        self.ax_w_p.set_ylabel("w")
        self.ax_relw_p.set_ylabel("relative w = w / (1/σ²)")
        self.ax_chi_p.set_ylabel("chi = w·(Fo² - Fc²)²")
        self.ax_w_d.set_ylabel("w")
        self.ax_relw_d.set_ylabel("relative w = w / (1/σ²)")
        self.ax_chi_d.set_ylabel("chi = w·(Fo² - Fc²)²")
        self.ax_w_p.set_xlabel("P")
        self.ax_relw_p.set_xlabel("P")
        self.ax_chi_p.set_xlabel("P")
        self.ax_w_d.set_xlabel("resolution d (Å)")
        self.ax_relw_d.set_xlabel("resolution d (Å)")
        self.ax_chi_d.set_xlabel("resolution d (Å)")

        bins = {k: nb_for(k) for k in axes.keys()}
        mode = {k: style_mode(k) for k in axes.keys()}

        if Fo2 is None or Fc2 is None or sigFo2 is None:
            self.status_box.setPlainText("Load a .fcf to plot.")
            self.canvas.draw_idle(); return
        if not curves:
            self.status_box.setPlainText("Add at least one curve.")
            self.canvas.draw_idle(); return

        if hkl is None or cell is None:
            self.status_box.setPlainText("FCF lacks hkl/cell; cannot compute s. Please load a file with hkl and cell.")
            self.canvas.draw_idle(); return

        try:
            s = _sin_theta_over_lambda(hkl, cell)
        except Exception as exc:
            self.status_box.setPlainText(f"Failed to compute s: {exc}")
            self.canvas.draw_idle(); return

        sigma2 = np.maximum(sigFo2, 1e-12) ** 2
        colors = plt_cm(len(curves))
        debug_lines = []
        P_min_pos = np.inf
        P_max_pos = 0.0
        rng = np.random.default_rng(0)
        for idx, curve in enumerate(curves):
            a = curve["a"]; b = curve["b"]; c = curve["c"]; f = curve["f"]
            # q(s) as in weights_wght
            if c == 0.0:
                q = np.ones_like(s)
            elif c > 0.0:
                q = np.exp(c * s * s)
            else:
                q = 1.0 - np.exp(c * s * s)
            q = np.maximum(q, 1e-300)

            P = float(f) * np.maximum(Fo2, 0.0) + (1.0 - float(f)) * np.maximum(Fc2, 0.0)
            P_pos = np.maximum(P, 1e-300)
            P_min_pos = min(P_min_pos, float(P_pos.min()))
            P_max_pos = max(P_max_pos, float(P_pos.max()))

            w = weights_wght(Fo2, Fc2, sigma2, s, a, b, c, f)
            w = np.maximum(w, 1e-300)
            resid = Fo2 - Fc2
            chi = np.maximum(w * (resid ** 2), 1e-300)
            w0 = 1.0 / np.maximum(sigma2, 1e-30)
            relw = np.maximum(w / w0, 1e-300)

            s_pos = np.clip(s, 1e-9, None)
            res_pos = 1.0 / np.maximum(2.0 * s_pos, 1e-12)

            nb_w_p = bins.get("w_p", 0)
            nb_relw_p = bins.get("relw_p", 0)
            nb_chi_p = bins.get("chi_p", 0)
            nb_w_d = bins.get("w_d", 0)
            nb_relw_d = bins.get("relw_d", 0)
            nb_chi_d = bins.get("chi_d", 0)
            show_bin_w_p = nb_w_p > 0 and mode["w_p"] in ("binned", "both")
            show_bin_relw_p = nb_relw_p > 0 and mode["relw_p"] in ("binned", "both")
            show_bin_chi_p = nb_chi_p > 0 and mode["chi_p"] in ("binned", "both")
            show_bin_w_d = nb_w_d > 0 and mode["w_d"] in ("binned", "both")
            show_bin_relw_d = nb_relw_d > 0 and mode["relw_d"] in ("binned", "both")
            show_bin_chi_d = nb_chi_d > 0 and mode["chi_d"] in ("binned", "both")

            show_scatter_w_p = mode["w_p"] in ("scatter", "both")
            show_scatter_relw_p = mode["relw_p"] in ("scatter", "both")
            show_scatter_chi_p = mode["chi_p"] in ("scatter", "both")
            show_scatter_w_d = mode["w_d"] in ("scatter", "both")
            show_scatter_relw_d = mode["relw_d"] in ("scatter", "both")
            show_scatter_chi_d = mode["chi_d"] in ("scatter", "both")

            xs_w_d = ys_w_d = xs_relw_d = ys_relw_d = xs_chi_d = ys_chi_d = np.array([])
            xp_w_p = yp_w_p = xp_relw_p = yp_relw_p = xp_chi_p = yp_chi_p = np.array([])
            if show_bin_w_d:
                xs_w_d, ys_w_d = bin_logx_median(res_pos, w, nb_w_d)
            if show_bin_relw_d:
                xs_relw_d, ys_relw_d = bin_logx_median(res_pos, relw, nb_relw_d)
            if show_bin_chi_d:
                xs_chi_d, ys_chi_d = bin_logx_median(res_pos, chi, nb_chi_d)
            if show_bin_w_p:
                xp_w_p, yp_w_p = bin_logx_median(P_pos, w, nb_w_p)
            if show_bin_relw_p:
                xp_relw_p, yp_relw_p = bin_logx_median(P_pos, relw, nb_relw_p)
            if show_bin_chi_p:
                xp_chi_p, yp_chi_p = bin_logx_median(P_pos, chi, nb_chi_p)

            color = colors[idx % len(colors)]
            label = f"curve {idx+1}: a={a:g}, b={b:g}, c={c:g}, f={f:g}"
            if show_bin_w_p and xp_w_p.size:
                self.ax_w_p.plot(xp_w_p, yp_w_p, color=color, alpha=0.95, label=label)
            if show_bin_relw_p and xp_relw_p.size:
                self.ax_relw_p.plot(xp_relw_p, yp_relw_p, color=color, alpha=0.95)
            if show_bin_chi_p and xp_chi_p.size:
                self.ax_chi_p.plot(xp_chi_p, yp_chi_p, color=color, alpha=0.95)
            if show_bin_w_d and xs_w_d.size:
                self.ax_w_d.plot(xs_w_d, ys_w_d, color=color, alpha=0.95)
            if show_bin_relw_d and xs_relw_d.size:
                self.ax_relw_d.plot(xs_relw_d, ys_relw_d, color=color, alpha=0.95)
            if show_bin_chi_d and xs_chi_d.size:
                self.ax_chi_d.plot(xs_chi_d, ys_chi_d, color=color, alpha=0.95)

            # raw subsamples, controlled per-plot
            idx_s_samp = np.array([], int)
            if s_pos.size > 0 and (show_scatter_w_d or show_scatter_relw_d or show_scatter_chi_d):
                n_samp_s = min(s_pos.size, 3000)
                idx_s_samp = rng.choice(s_pos.size, size=n_samp_s, replace=False)
            idx_p_samp = np.array([], int)
            if P_pos.size > 0 and (show_scatter_w_p or show_scatter_relw_p or show_scatter_chi_p):
                n_samp_p = min(P_pos.size, 3000)
                idx_p_samp = rng.choice(P_pos.size, size=n_samp_p, replace=False)

            if show_scatter_w_p and idx_p_samp.size:
                self.ax_w_p.plot(P_pos[idx_p_samp], w[idx_p_samp], linestyle="", marker=".", markersize=3, alpha=0.12, color=color)
            if show_scatter_relw_p and idx_p_samp.size:
                self.ax_relw_p.plot(P_pos[idx_p_samp], relw[idx_p_samp], linestyle="", marker=".", markersize=3, alpha=0.12, color=color)
            if show_scatter_chi_p and idx_p_samp.size:
                self.ax_chi_p.plot(P_pos[idx_p_samp], chi[idx_p_samp], linestyle="", marker=".", markersize=3, alpha=0.12, color=color)
            if show_scatter_w_d and idx_s_samp.size:
                self.ax_w_d.plot(res_pos[idx_s_samp], w[idx_s_samp], linestyle="", marker=".", markersize=3, alpha=0.12, color=color)
            if show_scatter_relw_d and idx_s_samp.size:
                self.ax_relw_d.plot(res_pos[idx_s_samp], relw[idx_s_samp], linestyle="", marker=".", markersize=3, alpha=0.12, color=color)
            if show_scatter_chi_d and idx_s_samp.size:
                self.ax_chi_d.plot(res_pos[idx_s_samp], chi[idx_s_samp], linestyle="", marker=".", markersize=3, alpha=0.12, color=color)

            # show P extents for diagnostics
            pmax = float(P_pos.max())
            p95 = float(np.percentile(P_pos, 95))
            if pmax > 0:
                self.ax_relw_p.axvline(pmax, color=color, alpha=0.25, linestyle="--")
                self.ax_chi_p.axvline(pmax, color=color, alpha=0.25, linestyle="--")
            if p95 > 0:
                self.ax_relw_p.axvline(p95, color=color, alpha=0.25, linestyle=":")
                self.ax_chi_p.axvline(p95, color=color, alpha=0.25, linestyle=":")

            bin_note = f"bins w_P={nb_w_p or 'off'} relw_P={nb_relw_p or 'off'} chi_P={nb_chi_p or 'off'} w_d={nb_w_d or 'off'} relw_d={nb_relw_d or 'off'} chi_d={nb_chi_d or 'off'}"
            debug_lines.append(
                f"#{idx+1} f={f:g} a={a:g} b={b:g} c={c:g} {bin_note} style P={mode['w_p']}/{mode['relw_p']}/{mode['chi_p']} d={mode['w_d']}/{mode['relw_d']}/{mode['chi_d']} | P max={P_pos.max():.3g} p95={np.percentile(P_pos,95):.3g} | xp_max={xp_relw_p.max() if xp_relw_p.size else float('nan'):.3g} | w max={w.max():.3g}"
            )

            # label cell color/text
            try:
                self._suspend_plot = True
                label_item = self.table.item(idx, 5)
                if label_item is None:
                    label_item = QtWidgets.QTableWidgetItem("")
                    label_item.setFlags(label_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                    label_item.setTextAlignment(int(QtCore.Qt.AlignmentFlag.AlignCenter))
                    self.table.setItem(idx, 5, label_item)
                label_item.setText(label)
                rgb = [int(max(0, min(1, c)) * 255) for c in color[:3]]
                label_item.setBackground(QBrush(QColor(*rgb)))
            finally:
                self._suspend_plot = False

        lam_txt = f"λ preset: {self.combo_lambda.currentText()} | λ = {self.spin_lambda.value():.5f} Å"
        cell_txt = "cell: none"
        if cell:
            cell_txt = "cell: " + ", ".join(
                f"{k}={cell.get(k, float('nan')):.4f}" for k in ["a", "b", "c", "alpha", "beta", "gamma"] if k in cell
            )
        status_lines = [
            f"Reflections: {len(Fo2)}",
            f"Curves plotted: {len(curves)}",
            lam_txt,
            cell_txt,
            "Bins per plot: " + ", ".join(
                [
                    f"w_P={bins['w_p'] or 'off'}",
                    f"relw_P={bins['relw_p'] or 'off'}",
                    f"chi_P={bins['chi_p'] or 'off'}",
                    f"w_d={bins['w_d'] or 'off'}",
                    f"relw_d={bins['relw_d'] or 'off'}",
                    f"chi_d={bins['chi_d'] or 'off'}",
                ]
            ),
            "c controls q(s); f mixes P = f·Fo² + (1-f)·Fc²",
        ]
        status_lines.extend(debug_lines)
        self.status_box.setPlainText("\n".join(status_lines))

        if P_max_pos > 0.0 and np.isfinite(P_min_pos):
            lo = max(P_min_pos, P_max_pos * 1e-6, 1e-9)
            hi = P_max_pos * 1.05
            self.ax_w_p.set_xlim(lo, hi)
            self.ax_relw_p.set_xlim(lo, hi)
            self.ax_chi_p.set_xlim(lo, hi)

        self.canvas.draw_idle()


def main() -> None:
    if not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    app = QtWidgets.QApplication(sys.argv)
    w = FcfWghtExplorer(); w.resize(1400, 900); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
