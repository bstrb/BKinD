#!/usr/bin/env python3
"""
Toy WGHT sweeper GUI (PyQt6 + Matplotlib).
Visualizes w, relative w, and w*(Fo^2 - Fc^2)^2 over a synthetic Fo^2 grid.
Start with a single editable curve (a, b, f, sigma fraction of Fo², Fc/Fo ratio),
add/remove curve rows to compare variants. No FCF loading or SHELXL runner.
"""
from __future__ import annotations

from typing import List, Optional
import os
import sys

import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QColor, QBrush
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

# -----------------
# Core math helpers
# -----------------
def weights_wght(Fo2: np.ndarray, Fc2: np.ndarray, sigma2: np.ndarray,
                 a: float, b: float, f: float) -> np.ndarray:
    P = f * np.maximum(Fo2, 0.0) + (1.0 - f) * Fc2
    denom = sigma2 + (a * P) ** 2 + b * P  # c=d=e=0
    return 1.0 / np.maximum(denom, 1e-30)


def make_Fo2_grid(Fmin: float, Fmax: float, npts: int) -> np.ndarray:
    return np.logspace(np.log10(Fmin), np.log10(Fmax), int(npts))


class ToyWghtExplorer(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Toy WGHT sweeper (PyQt6)")

        self.Fo2_min = 1e1
        self.Fo2_max = 1e6
        self.npts = 300

        # default curve parameters (editable per row)
        self.default_params = {
            "a": 0.0,
            "b": 0.0,
            "f": 1/3,
            "sigma_frac": 0.05,  # σ = sigma_frac * Fo²
            "fc_ratio": 1.1,     # Fc² = fc_ratio * Fo²
        }

        self._suspend_plot = False

        self._build_ui()
        self.reset_curves()

    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)

        # Matplotlib figure
        self.fig = Figure(figsize=(12.0, 8.0))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.toolbar)
        left.addWidget(self.canvas)

        # Controls
        controls_widget = QtWidgets.QWidget()
        controls_widget.setMinimumWidth(320)
        form = QtWidgets.QFormLayout(controls_widget)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        # Fo2 range
        self.spin_Fmin = QtWidgets.QDoubleSpinBox()
        self.spin_Fmin.setRange(1e-6, 1e9)
        self.spin_Fmin.setDecimals(3)
        self.spin_Fmin.setValue(self.Fo2_min)
        self.spin_Fmin.setSingleStep(10.0)
        self.spin_Fmin.editingFinished.connect(self.update_plots)
        form.addRow("Fo² min", self.spin_Fmin)

        self.spin_Fmax = QtWidgets.QDoubleSpinBox()
        self.spin_Fmax.setRange(1e-6, 1e12)
        self.spin_Fmax.setDecimals(3)
        self.spin_Fmax.setValue(self.Fo2_max)
        self.spin_Fmax.setSingleStep(1000.0)
        self.spin_Fmax.editingFinished.connect(self.update_plots)
        form.addRow("Fo² max", self.spin_Fmax)

        self.spin_npts = QtWidgets.QSpinBox()
        self.spin_npts.setRange(50, 2000)
        self.spin_npts.setValue(self.npts)
        self.spin_npts.editingFinished.connect(self.update_plots)
        form.addRow("n points", self.spin_npts)

        # Curve table (one row per curve)
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["a", "b", "f", "σ fraction", "Fc/Fo", "label"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.AllEditTriggers)
        self.table.itemChanged.connect(self._on_table_changed)
        form.addRow("Curves (edit cells)", self.table)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add_curve = QtWidgets.QPushButton("Add curve")
        self.btn_add_curve.clicked.connect(self.add_curve_row)
        self.btn_remove_curve = QtWidgets.QPushButton("Remove selected")
        self.btn_remove_curve.clicked.connect(self.remove_selected_row)
        self.btn_reset_curves = QtWidgets.QPushButton("Reset to default")
        self.btn_reset_curves.clicked.connect(self.reset_curves)
        btn_row.addWidget(self.btn_add_curve)
        btn_row.addWidget(self.btn_remove_curve)
        btn_row.addWidget(self.btn_reset_curves)
        btn_row_widget = QtWidgets.QWidget(); btn_row_widget.setLayout(btn_row)
        form.addRow(btn_row_widget)

        # Status box
        self.status_box = QtWidgets.QPlainTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setFixedHeight(150)
        form.addRow("Status", self.status_box)

        left_widget = QtWidgets.QWidget(); left_widget.setLayout(left)
        left_widget.setMinimumWidth(500)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(10)
        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(controls_widget)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setSizes([1000, 500])  # initial ratio; user can drag handle to adjust
        self.splitter.setStyleSheet(
            "QSplitter::handle { background: #7f8c8d; border: 1px solid #555; margin: 0 2px; }"
        )

        layout.addWidget(self.splitter)

        self._setup_axes()

    def _setup_axes(self) -> None:
        self.fig.clear()
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.0])
        self.ax_w = self.fig.add_subplot(gs[0])
        self.ax_relw = self.fig.add_subplot(gs[1])
        self.ax_term = self.fig.add_subplot(gs[2])

        for ax in (self.ax_w, self.ax_relw, self.ax_term):
            ax.set_xscale("log"); ax.set_yscale("log")
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
        vals = [params["a"], params["b"], params["f"], params["sigma_frac"], params["fc_ratio"]]
        for col, v in enumerate(vals):
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

    def reset_curves(self) -> None:
        self.table.setRowCount(0)
        self.add_curve_row()
        self.update_plots()

    def _on_table_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._suspend_plot:
            return
        self.update_plots()

    def _read_curve_rows(self) -> List[dict]:
        curves: List[dict] = []
        for row in range(self.table.rowCount()):
            try:
                a = float(self.table.item(row, 0).text())
                b = float(self.table.item(row, 1).text())
                f = float(self.table.item(row, 2).text())
                sigma_frac = float(self.table.item(row, 3).text())
                fc_ratio = float(self.table.item(row, 4).text())
            except Exception:
                continue
            curves.append({"a": a, "b": b, "f": f, "sigma_frac": sigma_frac, "fc_ratio": fc_ratio})
        return curves

    def update_plots(self) -> None:
        Fo2_min = float(self.spin_Fmin.value())
        Fo2_max = float(self.spin_Fmax.value())
        npts = int(self.spin_npts.value())
        Fo2 = make_Fo2_grid(Fo2_min, Fo2_max, npts)

        curves = self._read_curve_rows()

        # clear axes
        for ax in (self.ax_w, self.ax_relw, self.ax_term):
            ax.cla()
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)

        self.ax_w.set_ylabel("w")
        self.ax_relw.set_ylabel("relative w = w / (1/σ²)")
        self.ax_term.set_ylabel("w*(Fo² - Fc²)²")
        self.ax_term.set_xlabel("Fo²")

        if not curves:
            self.status_box.setPlainText("Add at least one curve (rows in the table).")
            self.canvas.draw_idle()
            return

        # plotting
        colors = plt_cm(len(curves))
        for idx, curve in enumerate(curves):
            a = curve["a"]; b = curve["b"]; f = curve["f"]
            sig_frac = curve["sigma_frac"]; r = curve["fc_ratio"]
            sigma = np.maximum(sig_frac * Fo2, 1e-30)
            sigma2 = sigma ** 2
            Fc2 = r * Fo2
            w = weights_wght(Fo2, Fc2, sigma2, a, b, f)
            w0 = 1.0 / np.maximum(sigma2, 1e-30)
            relw = w / w0
            term = w * (Fo2 - Fc2) ** 2

            color = colors[idx % len(colors)]
            label = f"curve {idx+1}: a={a:g}, b={b:g}, f={f:g}, σ= {sig_frac:g}·Fo², Fc/Fo={r:g}"
            self.ax_w.plot(Fo2, w, color=color, alpha=0.95, label=label)
            self.ax_relw.plot(Fo2, relw, color=color, alpha=0.95)
            self.ax_term.plot(Fo2, term, color=color, alpha=0.95)

            # paint label cell with curve color and text
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

        # no in-plot legend; labels shown beside table rows

        self.status_box.setPlainText(
            f"Fo² range: {Fo2_min:g} – {Fo2_max:g} (n={npts})\n"
            f"curves plotted: {len(curves)}\n"
            f"σ = sigma_frac * Fo²"
        )

        self.canvas.draw_idle()


def plt_cm(n: int):
    import matplotlib.pyplot as plt
    return plt.cm.tab20(np.linspace(0, 1, max(n, 1)))


def main() -> None:
    if not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    app = QtWidgets.QApplication(sys.argv)
    w = ToyWghtExplorer()
    w.resize(1400, 900)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
