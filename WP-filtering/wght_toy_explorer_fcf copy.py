#!/usr/bin/env python3
"""
WGHT toy explorer (FCF-backed).
Loads a SHELXL .fcf (hklf 4) to get Fo^2, Fc^2, sigma(Fo^2), hkl, and cell (if present);
computes s = sin(theta)/lambda from hkl, cell, and user-provided lambda;
plots w, relative w, and w*(Fo^2-Fc^2)^2 for one-or-more WGHT parameter sets.
Parameters per curve: a, b, c, f. Sigma/Fc come from the loaded file.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Dict
import shlex
import sys
import os
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
    P = float(f) * np.maximum(Fo2, 0.0) + (1.0 - float(f)) * Fc2
    denom = sigma2 + (float(a) * P) ** 2 + float(b) * P  # d=e=0
    return q / np.maximum(denom, 1e-30)


def bin_logx_median(x: np.ndarray, y: np.ndarray, nbins: int = 80):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[ok]; y = y[ok]
    if x.size == 0:
        return np.array([]), np.array([])
    edges = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), nbins + 1)
    xc = np.sqrt(edges[:-1] * edges[1:])
    yy = np.full_like(xc, np.nan, float)
    for i in range(len(xc)):
        m = (x >= edges[i]) & (x < edges[i+1])
        if np.any(m):
            yy[i] = np.nanmedian(y[m])
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

        self.default_params = {"a": 0.0, "b": 0.0, "c": 0.0, "f": 1/3}
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

        # Lambda
        self.spin_lambda = QtWidgets.QDoubleSpinBox()
        self.spin_lambda.setRange(1e-4, 10.0)
        self.spin_lambda.setDecimals(6)
        self.spin_lambda.setValue(0.71073)  # default Mo Kα
        self.spin_lambda.editingFinished.connect(self.update_plots)
        form.addRow("λ (Å)", self.spin_lambda)

        # Curve table
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["a", "b", "c", "f", "label"])
        self.table.horizontalHeader().setStretchLastSection(True)
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

        # Scale toggle
        self.chk_log = QtWidgets.QCheckBox("Log axes")
        self.chk_log.setChecked(True)
        self.chk_log.stateChanged.connect(self.update_plots)
        form.addRow(self.chk_log)

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
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.0])
        self.ax_w = self.fig.add_subplot(gs[0])
        self.ax_relw = self.fig.add_subplot(gs[1])
        self.ax_term = self.fig.add_subplot(gs[2])
        for ax in (self.ax_w, self.ax_relw, self.ax_term):
            ax.grid(True, which="both", alpha=0.25)
        self.ax_w.set_ylabel("w")
        self.ax_relw.set_ylabel("relative w = w / (1/σ²)")
        self.ax_term.set_ylabel("w*(Fo² - Fc²)²")
        self.ax_term.set_xlabel("Fo²")

    def add_curve_row(self, overrides: Optional[dict] = None) -> None:
        params = {**self.default_params}
        if overrides:
            params.update(overrides)
        self._suspend_plot = True
        row = self.table.rowCount()
        self.table.insertRow(row)
        vals = [params["a"], params["b"], params["c"], params["f"]]
        for col, v in enumerate(vals):
            item = QtWidgets.QTableWidgetItem(f"{v:.6g}")
            item.setTextAlignment(int(QtCore.Qt.AlignmentFlag.AlignCenter))
            self.table.setItem(row, col, item)
        label_item = QtWidgets.QTableWidgetItem("")
        label_item.setTextAlignment(int(QtCore.Qt.AlignmentFlag.AlignCenter))
        label_item.setFlags(label_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, 4, label_item)
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

    def _read_curves(self) -> List[dict]:
        curves: List[dict] = []
        for row in range(self.table.rowCount()):
            try:
                a = float(self.table.item(row, 0).text())
                b = float(self.table.item(row, 1).text())
                c = float(self.table.item(row, 2).text())
                f = float(self.table.item(row, 3).text())
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
        self.update_plots()

    def update_plots(self) -> None:
        Fo2 = self.Fo2; Fc2 = self.Fc2; sigFo2 = self.sigFo2; hkl = self.hkl; cell = self.cell
        curves = self._read_curves()

        use_log = self.chk_log.isChecked()
        for ax in (self.ax_w, self.ax_relw, self.ax_term):
            ax.cla()
            ax.set_xscale("log" if use_log else "linear")
            ax.set_yscale("log" if use_log else "linear")
            ax.grid(True, which="both", alpha=0.25)
        self.ax_w.set_ylabel("w")
        self.ax_relw.set_ylabel("relative w = w / (1/σ²)")
        self.ax_term.set_ylabel("w*(Fo² - Fc²)²")
        self.ax_term.set_xlabel("Fo²")

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
            lambda_A = float(self.spin_lambda.value())
            # s already sin(theta)/lambda; multiply by lambda? actually s = |h*|/2; to get sin(theta)/lambda for given lambda:
            s = s / lambda_A  # adjust to user lambda
        except Exception as exc:
            self.status_box.setPlainText(f"Failed to compute s: {exc}")
            self.canvas.draw_idle(); return

        sigma2 = np.maximum(sigFo2, 1e-12) ** 2
        colors = plt_cm(len(curves))
        for idx, curve in enumerate(curves):
            a = curve["a"]; b = curve["b"]; c = curve["c"]; f = curve["f"]
            w = weights_wght(Fo2, Fc2, sigma2, s, a, b, c, f)
            resid = Fo2 - Fc2
            contrib = w * (resid ** 2)
            w0 = 1.0 / np.maximum(sigma2, 1e-30)
            relw = w / w0

            # bin medians for smoother curves
            xw, yw = bin_logx_median(Fo2, relw)
            xc, yc = bin_logx_median(Fo2, contrib)

            color = colors[idx % len(colors)]
            label = f"curve {idx+1}: a={a:g}, b={b:g}, c={c:g}, f={f:g}"
            self.ax_w.plot(xw, yw, color=color, alpha=0.95, label=label)
            self.ax_relw.plot(xw, yw, color=color, alpha=0.95)
            self.ax_term.plot(xc, yc, color=color, alpha=0.95)

            # label cell color/text
            try:
                self._suspend_plot = True
                label_item = self.table.item(idx, 4)
                if label_item is None:
                    label_item = QtWidgets.QTableWidgetItem("")
                    label_item.setFlags(label_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                    label_item.setTextAlignment(int(QtCore.Qt.AlignmentFlag.AlignCenter))
                    self.table.setItem(idx, 4, label_item)
                label_item.setText(label)
                rgb = [int(max(0, min(1, c)) * 255) for c in color[:3]]
                label_item.setBackground(QBrush(QColor(*rgb)))
            finally:
                self._suspend_plot = False

        self.status_box.setPlainText(
            f"Reflections: {len(Fo2)}\n"
            f"curves plotted: {len(curves)}\n"
            f"σ from FCF; q(c) uses s from hkl/cell and λ"
        )

        self.canvas.draw_idle()


def main() -> None:
    if not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    app = QtWidgets.QApplication(sys.argv)
    w = FcfWghtExplorer(); w.resize(1400, 900); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
