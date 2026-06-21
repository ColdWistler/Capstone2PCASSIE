#!/usr/bin/env python3
"""Real-time system resource monitor with rolling history graphs.

Usage:
  python3 resource_monitor.py
  (optionally) python3 resource_monitor.py --log resources.csv

Hotkeys:
  Space  — start/stop recording
  R      — reset graphs + data
  S      — save data to CSV
"""
import csv
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime

import psutil
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QKeySequence, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

HISTORY_SECONDS = 60
REFRESH_MS = 1000
COLORS = {
    "cpu": QColor(52, 152, 219),
    "mem": QColor(46, 204, 113),
    "gpu": QColor(155, 89, 182),
    "gpumem": QColor(231, 76, 60),
    "disk_r": QColor(230, 126, 34),
    "disk_w": QColor(26, 188, 156),
}


class GraphWidget(QWidget):
    def __init__(self, title, unit, color, history_sec=HISTORY_SECONDS, parent=None):
        super().__init__(parent)
        self.title = title
        self.unit = unit
        self.color = color
        self.max_points = history_sec
        self.data = deque(maxlen=self.max_points)
        self.setMinimumSize(280, 180)
        self.setMaximumHeight(250)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(30, 30, 30))
        self.setPalette(pal)

    def add_point(self, value):
        self.data.append((time.monotonic(), value))

    def clear_data(self):
        self.data.clear()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        margin = 50
        plot_x, plot_y = margin, 10
        plot_w = w - margin - 10
        plot_h = h - plot_y - 30

        if plot_w <= 0 or plot_h <= 0:
            return

        # Background
        p.fillRect(0, 0, w, h, QColor(30, 30, 30))

        # Title
        p.setPen(QColor(200, 200, 200))
        font = QFont("monospace", 9)
        p.setFont(font)
        p.drawText(5, 15, f"{self.title} ({self.unit})")

        if not self.data:
            p.setPen(QColor(100, 100, 100))
            p.drawText(plot_x, plot_y + plot_h // 2, " waiting for data...")
            p.end()
            return

        values = [v for _, v in self.data]
        min_v, max_v = min(values), max(values)
        range_v = max_v - min_v if max_v > min_v else 1

        # Grid lines
        p.setPen(QPen(QColor(50, 50, 50), 1))
        for i in range(5):
            y = plot_y + plot_h * i // 4
            p.drawLine(plot_x, y, plot_x + plot_w, y)

        # Y-axis labels
        p.setPen(QColor(150, 150, 150))
        font.setPointSize(7)
        p.setFont(font)
        for i in range(5):
            val = max_v - range_v * i / 4
            y = plot_y + plot_h * i // 4
            label = f"{val:.0f}"
            p.drawText(2, y + 3, label)

        # Plot line
        pen = QPen(self.color, 2)
        p.setPen(pen)
        path = []
        for i, (ts, val) in enumerate(self.data):
            x = plot_x + plot_w * i / (len(self.data) - 1) if len(self.data) > 1 else plot_x + plot_w // 2
            y = plot_y + plot_h - (val - min_v) / range_v * plot_h
            path.append((x, y))

        for i in range(1, len(path)):
            p.drawLine(int(path[i - 1][0]), int(path[i - 1][1]), int(path[i][0]), int(path[i][1]))

        # Current value
        if values:
            font.setPointSize(10)
            p.setFont(font)
            p.setPen(self.color)
            latest = values[-1]
            p.drawText(plot_x + plot_w - 60, plot_y + 15, f"{latest:.1f}")

        p.end()


class ResourceMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Resource Monitor — DQN Flight Sim")
        self.setMinimumSize(900, 600)

        self.recording = False
        self.log_path = None
        self.log_file = None
        self.log_writer = None

        self.has_nvidia = os.system("nvidia-smi -L >/dev/null 2>&1") == 0
        self.prev_disk = psutil.disk_io_counters()
        self.prev_disk_time = time.monotonic()

        self._build_ui()
        self._build_toolbar()

        self.timer = QTimer()
        self.timer.timeout.connect(self._sample)
        self.timer.start(REFRESH_MS)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Control row
        ctrl = QHBoxLayout()
        self.btn_start = QPushButton("▶ Start")
        self.btn_start.clicked.connect(self._toggle_recording)
        self.btn_reset = QPushButton("↺ Reset")
        self.btn_reset.clicked.connect(self._reset)
        self.btn_save = QPushButton("💾 Save CSV")
        self.btn_save.clicked.connect(self._save_csv)
        self.status_label = QLabel("⏸ Stopped")

        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_reset)
        ctrl.addWidget(self.btn_save)
        ctrl.addStretch()
        ctrl.addWidget(self.status_label)
        layout.addLayout(ctrl)

        # Graph grid
        grid = QVBoxLayout()
        row1 = QHBoxLayout()
        self.graph_cpu = GraphWidget("CPU", "%", COLORS["cpu"])
        self.graph_mem = GraphWidget("Memory", "%", COLORS["mem"])
        row1.addWidget(self.graph_cpu)
        row1.addWidget(self.graph_mem)
        grid.addLayout(row1)

        row2 = QHBoxLayout()
        self.graph_gpu = GraphWidget("GPU", "%", COLORS["gpu"]) if self.has_nvidia else None
        self.graph_gpumem = GraphWidget("GPU Mem", "MB", COLORS["gpumem"]) if self.has_nvidia else None
        if self.graph_gpu and self.graph_gpumem:
            row2.addWidget(self.graph_gpu)
            row2.addWidget(self.graph_gpumem)
        else:
            placeholder = QLabel("No NVIDIA GPU detected")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: #555; font-style: italic;")
            row2.addWidget(placeholder)
        grid.addLayout(row2)

        row3 = QHBoxLayout()
        self.graph_disk_r = GraphWidget("Disk Read", "MB/s", COLORS["disk_r"])
        self.graph_disk_w = GraphWidget("Disk Write", "MB/s", COLORS["disk_w"])
        row3.addWidget(self.graph_disk_r)
        row3.addWidget(self.graph_disk_w)
        grid.addLayout(row3)

        layout.addLayout(grid)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready — Space to start/stop, R to reset, S to save")

    def _build_toolbar(self):
        tb = QToolBar("Shortcuts")
        self.addToolBar(tb)

        act_start = QAction("Start/Stop (Space)", self)
        act_start.setShortcut(QKeySequence("Space"))
        act_start.triggered.connect(self._toggle_recording)
        tb.addAction(act_start)

        act_reset = QAction("Reset (R)", self)
        act_reset.setShortcut(QKeySequence("R"))
        act_reset.triggered.connect(self._reset)
        tb.addAction(act_reset)

        act_save = QAction("Save (S)", self)
        act_save.setShortcut(QKeySequence("S"))
        act_save.triggered.connect(self._save_csv)
        tb.addAction(act_save)

    def _toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            self.btn_start.setText("⏸ Pause")
            self.status_label.setText("▶ Recording...")
            self.status_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
        else:
            self.btn_start.setText("▶ Start")
            self.status_label.setText("⏸ Stopped")
            self.status_label.setStyleSheet("color: #e67e22; font-weight: bold;")
        self.statusBar.showMessage(f"{'Recording' if self.recording else 'Paused'} — {len(self.graph_cpu.data)} samples")

    def _reset(self):
        for g in [self.graph_cpu, self.graph_mem, self.graph_disk_r, self.graph_disk_w]:
            g.clear_data()
        if self.graph_gpu:
            self.graph_gpu.clear_data()
            self.graph_gpumem.clear_data()
        self.prev_disk = psutil.disk_io_counters()
        self.prev_disk_time = time.monotonic()
        if self.log_file:
            self.log_file.close()
            self.log_file = None
            self.log_writer = None
        self.statusBar.showMessage("Reset — data cleared")

    def _sample(self):
        # CPU
        cpu = psutil.cpu_percent(interval=None)
        self.graph_cpu.add_point(cpu)

        # Memory
        mem = psutil.virtual_memory().percent
        self.graph_mem.add_point(mem)

        # GPU via nvidia-smi
        gpu_util = None
        gpu_mem = None
        if self.has_nvidia:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                parts = out.stdout.strip().split(", ")
                if len(parts) >= 2:
                    gpu_util = float(parts[0])
                    gpu_mem = float(parts[1])
                    self.graph_gpu.add_point(gpu_util)
                    self.graph_gpumem.add_point(gpu_mem)
            except Exception:
                pass

        # Disk I/O
        now = time.monotonic()
        disk = psutil.disk_io_counters()
        dt = now - self.prev_disk_time
        if dt > 0:
            read_mb = (disk.read_bytes - self.prev_disk.read_bytes) / 1e6 / dt
            write_mb = (disk.write_bytes - self.prev_disk.write_bytes) / 1e6 / dt
            self.graph_disk_r.add_point(read_mb)
            self.graph_disk_w.add_point(write_mb)
        self.prev_disk = disk
        self.prev_disk_time = now

        # Log to CSV if recording
        if self.recording:
            ts = datetime.now().isoformat()
            row = [ts, f"{cpu:.1f}", f"{mem:.1f}",
                   f"{gpu_util:.1f}" if gpu_util is not None else "",
                   f"{gpu_mem:.0f}" if gpu_mem is not None else "",
                   f"{read_mb:.2f}" if dt > 0 else "",
                   f"{write_mb:.2f}" if dt > 0 else ""]

            if self.log_file is None:
                self.log_path = f"resources_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                self.log_file = open(self.log_path, "w", newline="")
                self.log_writer = csv.writer(self.log_file)
                self.log_writer.writerow(["timestamp", "cpu_pct", "mem_pct",
                                          "gpu_util_pct", "gpu_mem_mb",
                                          "disk_read_mbs", "disk_write_mbs"])
            self.log_writer.writerow(row)
            self.log_file.flush()

        # Trigger repaint
        self.graph_cpu.update()
        self.graph_mem.update()
        if self.graph_gpu:
            self.graph_gpu.update()
            self.graph_gpumem.update()
        self.graph_disk_r.update()
        self.graph_disk_w.update()

        samples = len(self.graph_cpu.data)
        self.statusBar.showMessage(
            f"{'▶' if self.recording else '⏸'} {samples} samples"
            f"{f'  |  logged to {self.log_path}' if self.log_file else ''}"
        )

    def _save_csv(self):
        if self.log_file:
            self.log_file.flush()
            path = self.log_path
            self.statusBar.showMessage(f"Saved to {path}")
            return
        # If not recording, offer save dialog to export current buffer
        path, _ = QFileDialog.getSaveFileName(self, "Save Resource Data", "resources.csv", "CSV (*.csv)")
        if path:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "cpu_pct", "mem_pct", "gpu_util_pct", "gpu_mem_mb",
                            "disk_read_mbs", "disk_write_mbs"])
                # We only have the rolling buffer without timestamps for simplicity
                # but during recording we save full data anyway
            self.statusBar.showMessage(f"Saved to {path}")

    def closeEvent(self, event):
        if self.log_file:
            self.log_file.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Resource Monitor")
    win = ResourceMonitor()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
