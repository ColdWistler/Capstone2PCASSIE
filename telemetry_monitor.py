#!/usr/bin/env python3
"""Combined aircraft telemetry + system resource monitor (PyQt6).

Usage:
  python3 telemetry_monitor.py                              # manual mode
  python3 telemetry_monitor.py -- <command> [args...]       # auto-launch + monitor

Telemetry is read from telemetry/telemetry.jsonl (written by TelemetryExporter).
System resources are sampled via psutil + nvidia-smi.
"""
import csv
import json
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import psutil
from PyQt6.QtCore import QProcess, Qt, QTimer
from PyQt6.QtGui import QAction, QColor, QFont, QKeySequence, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).parent.resolve()
TELEMETRY_DIR = PROJECT_ROOT / "telemetry"
TELEMETRY_PATH = TELEMETRY_DIR / "telemetry.jsonl"

HISTORY_SECONDS = 60
REFRESH_MS = 1000
MAX_TELEMETRY_POINTS = 500

COLORS = {
    "cpu": QColor(52, 152, 219),
    "mem": QColor(46, 204, 113),
    "gpu": QColor(155, 89, 182),
    "gpumem": QColor(231, 76, 60),
    "disk_r": QColor(230, 126, 34),
    "disk_w": QColor(26, 188, 156),
    "alt": QColor(52, 152, 219),
    "spd": QColor(46, 204, 113),
    "vspd": QColor(155, 89, 182),
    "gforce": QColor(231, 76, 60),
    "epow": QColor(241, 196, 15),
    "fuel": QColor(230, 126, 34),
}


class GraphWidget(QWidget):
    def __init__(self, title, unit, color, history_sec=HISTORY_SECONDS, parent=None):
        super().__init__(parent)
        self.title = title
        self.unit = unit
        self.color = color
        self.max_points = history_sec
        self.data = deque(maxlen=self.max_points)
        self.setMinimumSize(200, 140)
        self.setMaximumHeight(200)
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
        margin = 45
        plot_x, plot_y = margin, 10
        plot_w = w - margin - 10
        plot_h = h - plot_y - 25

        if plot_w <= 0 or plot_h <= 0:
            return

        p.fillRect(0, 0, w, h, QColor(30, 30, 30))

        p.setPen(QColor(200, 200, 200))
        font = QFont("monospace", 8)
        p.setFont(font)
        p.drawText(5, 12, f"{self.title} ({self.unit})")

        if not self.data:
            p.setPen(QColor(100, 100, 100))
            p.drawText(plot_x, plot_y + plot_h // 2, " waiting...")
            p.end()
            return

        values = [v for _, v in self.data]
        min_v, max_v = min(values), max(values)
        range_v = max_v - min_v if max_v > min_v else 1

        p.setPen(QPen(QColor(50, 50, 50), 1))
        for i in range(5):
            y = plot_y + plot_h * i // 4
            p.drawLine(plot_x, y, plot_x + plot_w, y)

        p.setPen(QColor(150, 150, 150))
        font.setPointSize(7)
        p.setFont(font)
        for i in range(5):
            val = max_v - range_v * i / 4
            y = plot_y + plot_h * i // 4
            p.drawText(2, y + 3, f"{val:.0f}")

        pen = QPen(self.color, 2)
        p.setPen(pen)
        path = []
        for i, (ts, val) in enumerate(self.data):
            x = plot_x + plot_w * i / (len(self.data) - 1) if len(self.data) > 1 else plot_x + plot_w // 2
            y = plot_y + plot_h - (val - min_v) / range_v * plot_h
            path.append((x, y))

        for i in range(1, len(path)):
            p.drawLine(int(path[i - 1][0]), int(path[i - 1][1]), int(path[i][0]), int(path[i][1]))

        if values:
            font.setPointSize(10)
            p.setFont(font)
            p.setPen(self.color)
            p.drawText(plot_x + plot_w - 60, plot_y + 15, f"{values[-1]:.1f}")

        p.end()


class TelemetryMonitor(QMainWindow):
    def __init__(self, spawn_cmd=None):
        super().__init__()
        self.spawn_cmd = spawn_cmd
        self.process = None
        self.auto_mode = spawn_cmd is not None
        title = "Aircraft Telemetry & Resource Monitor"
        if self.auto_mode:
            name = " ".join(spawn_cmd[:3]) + ("..." if len(spawn_cmd) > 3 else "")
            title = f"Telemetry Monitor — {name}"
        self.setWindowTitle(title)
        self.setMinimumSize(1100, 750)

        self.recording = False
        self.log_path = None
        self.log_file = None
        self.log_writer = None

        # Telemetry state
        self.telemetry_buffer = deque(maxlen=MAX_TELEMETRY_POINTS)
        self.known_count = 0
        self.t_start = None

        # Resource state
        self.has_nvidia = os.system("nvidia-smi -L >/dev/null 2>&1") == 0
        self.prev_disk = psutil.disk_io_counters()
        self.prev_disk_time = time.monotonic()

        # Ensure telemetry directory exists
        TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
        if not TELEMETRY_PATH.exists():
            TELEMETRY_PATH.touch()

        self._build_ui()
        self._build_toolbar()

        self.timer = QTimer()
        self.timer.timeout.connect(self._sample)
        self.timer.start(REFRESH_MS)

        if self.auto_mode:
            self._launch_subprocess()

    def _launch_subprocess(self):
        self.process = QProcess(self)
        self.process.finished.connect(self._on_process_finished)
        self.process.errorOccurred.connect(self._on_process_error)
        cmd = self.spawn_cmd[0]
        args = self.spawn_cmd[1:]
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.ForwardedChannels)
        self.process.start(cmd, args)
        if self.process.waitForStarted(3000):
            pid = self.process.processId()
            self.statusBar.showMessage(f"Launched PID {pid}")
            self._toggle_recording()
        else:
            self.statusBar.showMessage(f"Failed to start: {cmd}")

    def _on_process_finished(self, exit_code, exit_status):
        self.statusBar.showMessage(f"Process exited (code {exit_code})")
        if self.recording:
            self._toggle_recording()
        self.btn_start.setEnabled(False)
        QTimer.singleShot(2000, self.close)

    def _on_process_error(self, error):
        self.statusBar.showMessage(f"Process error: {error}")

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

        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # --- Tab 1: Aircraft Telemetry ---
        tele_tab = QWidget()
        tele_layout = QVBoxLayout(tele_tab)

        # Digital readouts
        gauges_group = QGroupBox("Live Telemetry")
        gauges_grid = QGridLayout(gauges_group)

        self.labels = {}
        gauge_defs = [
            ("alt", "Altitude", "m", "{:.1f}"),
            ("spd", "Airspeed", "m/s", "{:.1f}"),
            ("vspd", "V/Speed", "m/s", "{:.2f}"),
            ("g", "G-Force", "g", "{:.2f}"),
            ("epow", "Engine", "%", "{:.0f}"),
            ("fuel", "Fuel", "%", "{:.0f}"),
            ("roll", "Roll", "°", "{:.1f}"),
            ("pitch", "Pitch", "°", "{:.1f}"),
            ("hdg", "Heading", "°", "{:.1f}"),
            ("stall", "Stalled", "", "{}"),
            ("gear", "Gear", "", "{}"),
            ("eact", "Engine", "", "{}"),
        ]

        for i, (key, label, unit, fmt) in enumerate(gauge_defs):
            row, col = i // 4, i % 4
            container = QWidget()
            cl = QVBoxLayout(container)
            cl.setContentsMargins(4, 4, 4, 4)
            name_label = QLabel(label)
            name_label.setStyleSheet("color: #aaa; font-size: 10px;")
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label = QLabel("---")
            value_label.setStyleSheet("color: #fff; font-size: 18px; font-weight: bold;")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label.setMinimumHeight(36)
            cl.addWidget(name_label)
            cl.addWidget(value_label)
            gauges_grid.addWidget(container, row, col)
            self.labels[key] = (value_label, unit, fmt)

        tele_layout.addWidget(gauges_group)

        # Waiting placeholder
        self.waiting_label = QLabel(
            "⏳ Waiting for telemetry data...\n"
            "Launch the Godot flight sim with a TelemetryExporter node attached.\n"
            f"Looking for: {TELEMETRY_PATH}"
        )
        self.waiting_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.waiting_label.setStyleSheet("color: #888; font-size: 14px; padding: 20px;")
        self.waiting_label.setMinimumHeight(80)
        tele_layout.addWidget(self.waiting_label)

        # Telemetry rolling graphs
        graphs_group = QGroupBox("Telemetry History (60s)")
        graph_grid = QGridLayout(graphs_group)

        self.graph_alt = GraphWidget("Altitude", "m", COLORS["alt"])
        self.graph_spd = GraphWidget("Airspeed", "m/s", COLORS["spd"])
        self.graph_epow = GraphWidget("Engine", "%", COLORS["epow"])
        self.graph_fuel = GraphWidget("Fuel", "%", COLORS["fuel"])

        graph_grid.addWidget(self.graph_alt, 0, 0)
        graph_grid.addWidget(self.graph_spd, 0, 1)
        graph_grid.addWidget(self.graph_epow, 1, 0)
        graph_grid.addWidget(self.graph_fuel, 1, 1)

        tele_layout.addWidget(graphs_group)
        self.graphs_group = graphs_group
        self.tabs.addTab(tele_tab, "✈ Aircraft Telemetry")

        # --- Tab 2: System Resources ---
        res_tab = QWidget()
        res_layout = QVBoxLayout(res_tab)

        res_grid = QVBoxLayout()
        row1 = QHBoxLayout()
        self.graph_cpu = GraphWidget("CPU", "%", COLORS["cpu"])
        self.graph_mem = GraphWidget("Memory", "%", COLORS["mem"])
        row1.addWidget(self.graph_cpu)
        row1.addWidget(self.graph_mem)
        res_grid.addLayout(row1)

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
        res_grid.addLayout(row2)

        row3 = QHBoxLayout()
        self.graph_disk_r = GraphWidget("Disk Read", "MB/s", COLORS["disk_r"])
        self.graph_disk_w = GraphWidget("Disk Write", "MB/s", COLORS["disk_w"])
        row3.addWidget(self.graph_disk_r)
        row3.addWidget(self.graph_disk_w)
        res_grid.addLayout(row3)

        # Resource digital readouts
        res_info = QHBoxLayout()
        self.res_labels = {}
        for key, label, color in [
            ("cpu", "CPU", "#3498db"),
            ("mem", "MEM", "#2ecc71"),
            ("gpu_u", "GPU", "#9b59b6"),
            ("gpu_m", "GPU Mem", "#e74c3c"),
        ]:
            lbl = QLabel(f"{label}: --")
            lbl.setStyleSheet(f"color: {color}; font-size: 13px; font-weight: bold;")
            lbl.setMinimumWidth(140)
            res_info.addWidget(lbl)
            self.res_labels[key] = lbl
        res_info.addStretch()
        res_layout.addLayout(res_info)

        res_layout.addLayout(res_grid)
        self.tabs.addTab(res_tab, "🖥 System Resources")

        # Status bar
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
        self.statusBar.showMessage(f"{'Recording' if self.recording else 'Paused'}")

    def _reset(self):
        for g in [self.graph_cpu, self.graph_mem, self.graph_disk_r, self.graph_disk_w,
                  self.graph_alt, self.graph_spd, self.graph_epow, self.graph_fuel]:
            g.clear_data()
        if self.graph_gpu:
            self.graph_gpu.clear_data()
            self.graph_gpumem.clear_data()
        self.telemetry_buffer.clear()
        self.known_count = 0
        self.t_start = None
        self.prev_disk = psutil.disk_io_counters()
        self.prev_disk_time = time.monotonic()
        for key in self.labels:
            lbl, _, _ = self.labels[key]
            lbl.setText("---")
        if self.log_file:
            self.log_file.close()
            self.log_file = None
            self.log_writer = None
        self.statusBar.showMessage("Reset — data cleared")

    def _read_telemetry(self):
        path = TELEMETRY_PATH
        if not path.exists() or path.stat().st_size == 0:
            return
        try:
            with open(path, "r") as f:
                lines = f.readlines()
            raw = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if not raw:
                return
            new_items = raw[self.known_count:] if self.known_count < len(raw) else []
            if new_items:
                for item in new_items:
                    t = item.get("t", 0) / 1000.0
                    if self.t_start is None:
                        self.t_start = t
                self.telemetry_buffer.extend(new_items)
                self.known_count = len(raw)
        except (OSError, IOError):
            pass

    def _update_telemetry_display(self):
        data = list(self.telemetry_buffer)
        has_data = bool(data)
        self.waiting_label.setVisible(not has_data)
        self.graphs_group.setVisible(has_data)
        if not has_data:
            return

        latest = data[-1]

        # Update digital labels
        for key, (lbl, unit, fmt) in self.labels.items():
            if key == "stall":
                val = "YES" if latest.get("stall", 0) else "NO"
                color = "#e74c3c" if latest.get("stall", 0) else "#2ecc71"
            elif key == "gear":
                val = "DOWN" if latest.get("gear", 0) else "UP"
                color = "#3498db" if latest.get("gear", 0) else "#555"
            elif key == "eact":
                val = "ON" if latest.get("eact", 0) else "OFF"
                color = "#e67e22" if latest.get("eact", 0) else "#555"
            elif key == "epow" or key == "fuel":
                raw_val = latest.get(key, 0)
                if isinstance(raw_val, (int, float)):
                    val = fmt.format(raw_val * 100)
                else:
                    val = "---"
                color = "#fff"
            else:
                raw_val = latest.get(key, None)
                if raw_val is None:
                    val = "---"
                else:
                    val = fmt.format(raw_val)
                color = "#fff"
            lbl.setText(val)
            lbl.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: bold;")

        # Update rolling graphs along real time
        self.graph_alt.clear_data()
        self.graph_spd.clear_data()
        self.graph_epow.clear_data()
        self.graph_fuel.clear_data()

        t0 = data[0].get("t", 0) / 1000.0
        for item in data:
            t_rel = item.get("t", 0) / 1000.0 - t0
            self.graph_alt.add_point(item.get("alt", 0))
            self.graph_spd.add_point(item.get("spd", 0))
            self.graph_epow.add_point(item.get("epow", 0) * 100)
            self.graph_fuel.add_point(item.get("fuel", 0) * 100)

        self.graph_alt.update()
        self.graph_spd.update()
        self.graph_epow.update()
        self.graph_fuel.update()

    def _sample(self):
        # --- System resources ---
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        self.graph_cpu.add_point(cpu)
        self.graph_mem.add_point(mem)

        self.res_labels["cpu"].setText(f"CPU: {cpu:.1f}%")
        self.res_labels["mem"].setText(f"MEM: {mem:.1f}%")

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
                    self.res_labels["gpu_u"].setText(f"GPU: {gpu_util:.1f}%")
                    self.res_labels["gpu_m"].setText(f"GPU Mem: {gpu_mem:.0f} MB")
            except Exception:
                pass

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

        self.graph_cpu.update()
        self.graph_mem.update()
        if self.graph_gpu:
            self.graph_gpu.update()
            self.graph_gpumem.update()
        self.graph_disk_r.update()
        self.graph_disk_w.update()

        # --- Telemetry ---
        self._read_telemetry()
        self._update_telemetry_display()

        # --- Logging ---
        if self.recording:
            ts = datetime.now().isoformat()
            row = [ts, f"{cpu:.1f}", f"{mem:.1f}",
                   f"{gpu_util:.1f}" if gpu_util is not None else "",
                   f"{gpu_mem:.0f}" if gpu_mem is not None else "",
                   f"{read_mb:.2f}" if dt > 0 else "",
                   f"{write_mb:.2f}" if dt > 0 else ""]

            if self.log_file is None:
                self.log_path = f"telemetry_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                self.log_file = open(self.log_path, "w", newline="")
                self.log_writer = csv.writer(self.log_file)
                self.log_writer.writerow([
                    "timestamp", "cpu_pct", "mem_pct",
                    "gpu_util_pct", "gpu_mem_mb",
                    "disk_read_mbs", "disk_write_mbs",
                ])
            self.log_writer.writerow(row)
            self.log_file.flush()

        samples = len(self.graph_cpu.data)
        tele_samples = len(self.telemetry_buffer)
        self.statusBar.showMessage(
            f"{'▶' if self.recording else '⏸'} {samples} resource samples"
            f" | {tele_samples} telemetry samples"
            f"{f'  |  logged' if self.log_file else ''}"
        )

    def _save_csv(self):
        if self.log_file:
            self.log_file.flush()
            self.statusBar.showMessage(f"Saved to {self.log_path}")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Data", "telemetry.csv", "CSV (*.csv)")
        if path:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "cpu_pct", "mem_pct"])
            self.statusBar.showMessage(f"Saved to {path}")

    def closeEvent(self, event):
        if self.log_file:
            self.log_file.close()
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()
            if not self.process.waitForFinished(3000):
                self.process.kill()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Telemetry Monitor")

    spawn_cmd = None
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        spawn_cmd = sys.argv[idx + 1:]
        if not spawn_cmd:
            print("error: nothing after --", file=sys.stderr)
            sys.exit(1)

    win = TelemetryMonitor(spawn_cmd=spawn_cmd)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
