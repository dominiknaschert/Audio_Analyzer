"""
Flutter Echo Analysis Widget
Displays distogram data and visualizes the assignment to room dimensions.
"""

import numpy as np
from typing import Optional
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QGroupBox, QFormLayout, QSplitter, QComboBox,
    QDoubleSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QLineEdit, QPushButton, QDialog, QGridLayout
)
from PySide6.QtCore import Qt, QPointF, QRectF, Slot, Signal
from PySide6.QtGui import QWheelEvent, QKeyEvent, QPainter, QPen, QColor, QPolygonF, QBrush, QDoubleValidator

from ..core.flutter_detection import (
    FlutterEchoDetector, FlutterEchoResult, PeakInfo, SOUND_SPEED,
    FlutterDetectionConfig, compute_decay_curve, DecayCurveResult, analyze_flutter
)


class AnalysisPipelineDialog(QDialog):
    """Dialog for visualizing the complete Room Analysis pipeline."""
    def __init__(self, result: FlutterEchoResult, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Room Analysis Pipeline Details (after Schanda et al.)")
        self.resize(1100, 850)
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")
        
        layout = QVBoxLayout(self)
        
        # Grid for the 6 plots
        grid = QGridLayout()
        grid.setSpacing(15)
        
        # Helper function for plot creation
        def create_plot(title, xlabel, ylabel=None, units_x=None, units_y=None):
            p = pg.PlotWidget()
            p.setBackground('#1e1e2e')
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setTitle(title, color="#cdd6f4", size="11pt")
            p.setLabel('bottom', xlabel, units=units_x)
            if ylabel:
                p.setLabel('left', ylabel, units=units_y)
            return p

        # 1. Original RIR
        p1 = create_plot("1. Original Impulse Response (RIR)", "Time", units_x="s")
        if result.rir_raw is not None:
            p1.plot(result.t, result.rir_raw, pen=pg.mkPen('#6c7086', width=0.5))
        grid.addWidget(p1, 0, 0)
        
        # 2. Bandpass filtered
        p2 = create_plot(f"2. Bandpass Filtered ({result.flutter_tonality_hz:.0f} Hz)", "Time", units_x="s")
        if result.rir_bp is not None:
            p2.plot(result.t, result.rir_bp, pen=pg.mkPen('#89b4fa', width=0.5))
        grid.addWidget(p2, 0, 1)
        
        # 3. Level-Time with regressions
        p3 = create_plot("3. Echogram & Regressions", "Time", "Level", units_x="s", units_y="dB")
        if result.l_ir is not None:
            p3.plot(result.t, result.l_ir, pen=pg.mkPen('#45475a', width=1))
            # Mark region I (pre-shading) - use actual start
            t_start_used = result.t_start_analysis if hasattr(result, 't_start_analysis') else 0.2
            vspan = pg.LinearRegionItem([0, t_start_used], movable=False, brush=QBrush(QColor(243, 139, 168, 40)))
            p3.addItem(vspan)
            
            # Regressions - use actual values
            if result.p_decay is not None:
                # Use regression marker positions if available, otherwise use analysis start
                t_plot_start = getattr(result, 't_regression_start', t_start_used)
                t_plot_end = result.t_intersect + 0.1
                t_plot = np.linspace(t_plot_start, t_plot_end, 100)
                l_plot = np.polyval(result.p_decay, t_plot)
                p3.plot(t_plot, l_plot, pen=pg.mkPen('#a6e3a1', width=2))
                
                # Noise floor
                p3.addLine(y=result.noise_level, pen=pg.mkPen('#89b4fa', width=2, style=Qt.DashLine))
                
                # Intersection point
                p3.plot([result.t_intersect], [result.noise_level], pen=None, symbol='o', symbolSize=8, symbolBrush='#f9e2af')
            
            # Trend line
            if result.l_trend is not None:
                p3.plot(result.t, result.l_trend, pen=pg.mkPen('#cdd6f4', width=1, style=Qt.DotLine))
            
            p3.setYRange(max(np.min(result.l_ir), result.noise_level - 20), 5)
        grid.addWidget(p3, 1, 0)
        
        # 4. Corrected Signal L_FE
        p4 = create_plot("4. Corrected Signal L_FE (Echos Isolated)", "Time", "Amplitude", units_x="s", units_y="dB")
        if result.l_fe is not None:
            # Use actual time windowing from analysis
            t_start_used = result.t_start_analysis if hasattr(result, 't_start_analysis') else 0.2
            t_end_used = result.t_end_analysis if hasattr(result, 't_end_analysis') else result.t_intersect
            mask = (result.t >= t_start_used) & (result.t <= t_end_used)
            p4.plot(result.t[mask], result.l_fe[mask], pen=pg.mkPen('#f38ba8', width=1))
        grid.addWidget(p4, 1, 1)
        
        # 5. Autocorrelation
        p5 = create_plot("5. Autocorrelation Function (ACF)", "Lag", units_x="s")
        if result.acf is not None:
            # Calculate fs from t
            fs = 1.0 / (result.t[1] - result.t[0]) if len(result.t) > 1 else 48000
            t_acf = np.arange(len(result.acf)) / fs
            p5.plot(t_acf, result.acf, pen=pg.mkPen('#a6e3a1', width=1.5))
            p5.setXRange(0, 0.4)
        grid.addWidget(p5, 2, 0)
        
        # 6. Distogram
        p6 = create_plot("6. Distogram (FFT of ACF)", "Distance", units_x="m")
        p6.plot(result.distances, result.amplitudes, pen=pg.mkPen('#cba6f7', width=1.5))
        # Mark peaks
        if result.peaks:
            p6.plot([p.distance_m for p in result.peaks], [p.amplitude for p in result.peaks], 
                   pen=None, symbol='t', symbolSize=8, symbolBrush='#f38ba8')
        # Limit X-range to 0-20m (standard)
        if len(result.distances) > 0:
            p6.setXRange(0, min(20.0, np.max(result.distances) * 1.1))
        grid.addWidget(p6, 2, 1)
        
        layout.addLayout(grid)
        
        # Close Button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45475a;
            }
        """)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)


class RT60DetailDialog(QDialog):
    """Dialog for visualizing level decay and RT60 regression."""
    def __init__(self, result: FlutterEchoResult, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RT60 Details - Level Decay")
        self.resize(800, 500)
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")
        
        layout = QVBoxLayout(self)
        
        # Info Label - RT60 in gray at the top
        info_label = QLabel(f"RT60: {result.rt60_s:.2f} s")
        info_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #6c7086; margin-bottom: 5px;")
        layout.addWidget(info_label)
        
        # Plot
        self.plot = pg.PlotWidget()
        self.plot.setBackground('#1e1e2e')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel('left', 'Level', units='dB')
        self.plot.setLabel('bottom', 'Time', units='s')
        
        if result.t is not None and result.l_ir is not None:
            # Level progression
            self.plot.plot(result.t, result.l_ir, pen=pg.mkPen('#89b4fa', width=1), name="Signal (Filtered)")
            
            # Calculate regression and level drop using the selected regression markers
            actual_level_drop = None
            if result.p_decay is not None:
                # Use regression marker positions (user-selected start and end points)
                t_regression_start = getattr(result, 't_regression_start', result.t_start_analysis)
                t_regression_end = getattr(result, 't_regression_end', result.t_end_analysis)
                
                # Calculate level at regression start and end points
                level_start = np.polyval(result.p_decay, t_regression_start)
                level_end = np.polyval(result.p_decay, t_regression_end)
                actual_level_drop = level_start - level_end  # Positive value = drop in dB
                
                # Draw regression line between selected start and end points
                # Extend to intersection with noise floor for RT60 extrapolation
                t_intersect = result.t_intersect
                t_plot_start = t_regression_start
                t_plot_end = min(t_intersect, result.t[-1])  # Don't extend beyond signal
                
                # Draw regression line from start marker to intersection
                t_regression_plot = np.linspace(t_plot_start, t_plot_end, 100)
                l_regression_plot = np.polyval(result.p_decay, t_regression_plot)
                self.plot.plot(t_regression_plot, l_regression_plot, pen=pg.mkPen('#6c7086', width=2.5), name="RT60 Regression")
                
                # Draw vertical lines at regression start and end markers (gray)
                start_line = pg.InfiniteLine(
                    pos=t_regression_start, angle=90, movable=False,
                    pen=pg.mkPen('#6c7086', width=1.5, style=Qt.DashLine)
                )
                self.plot.addItem(start_line)
                
                end_line = pg.InfiniteLine(
                    pos=t_regression_end, angle=90, movable=False,
                    pen=pg.mkPen('#6c7086', width=1.5, style=Qt.DashLine)
                )
                self.plot.addItem(end_line)
                
                # Draw line showing the actual level drop between regression markers (gray)
                t_drop = np.array([t_regression_start, t_regression_end])
                l_drop = np.array([level_start, level_end])
                self.plot.plot(t_drop, l_drop, pen=pg.mkPen('#6c7086', width=2, style=Qt.DotLine), name="Level Drop")
                
                # Add annotation showing the level drop value (gray)
                # Position it at the top of the plot
                mid_t = (t_regression_start + t_regression_end) / 2
                min_l = np.min(result.l_ir)
                max_l = np.max(result.l_ir)
                # Position at top of plot (near max level)
                top_l = max_l - 2  # Slightly below max to avoid edge
                drop_text = pg.TextItem(
                    text=f"Δ = {actual_level_drop:.1f} dB",
                    anchor=(0.5, 0.5),
                    color='#6c7086'
                )
                drop_text.setPos(mid_t, top_l)
                self.plot.addItem(drop_text)
            
            # Scale axes appropriately (level range)
            min_l = np.min(result.l_ir)
            max_l = np.max(result.l_ir)
            self.plot.setYRange(max(min_l, max_l - 80), max_l + 5)
            
        layout.addWidget(self.plot)
        
        # Close Button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45475a;
            }
        """)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)


class RT60CorrectionDialog(QDialog):
    """
    Dialog for interactive correction of RT60 regression.
    
    Regression: Select start/end time, regression is automatically calculated.
    """
    
    def __init__(self, decay_result: DecayCurveResult, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RT60 Correction")
        self.resize(1000, 700)
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")
        
        self._decay = decay_result
        self._accepted = False
        
        # Store regression parameters (will be calculated in _update_display)
        self._regression_slope = None
        self._regression_intercept = None
        
        # Excluded regions (start and end)
        self._t_exclude_start = decay_result.t_start_fit  # Default: 0.2s, adjustable
        self._t_exclude_end = decay_result.t[-1]  # Default: end of RIR (no exclusion), adjustable
        
        # Initial values from automatic calculation
        # Regression markers must be within the non-excluded area (can touch exclusions)
        t_start = decay_result.t_start_fit
        # Ensure start marker is >= start exclusion (can touch)
        if t_start < self._t_exclude_start:
            t_start = self._t_exclude_start
        # Ensure it's also <= end exclusion
        if t_start > self._t_exclude_end:
            t_start = self._t_exclude_end
        idx_start = np.argmin(np.abs(decay_result.t - t_start))
        self._start_x = t_start
        self._start_y = decay_result.l_ir[idx_start]
        
        t_end = decay_result.t_end_fit
        # Ensure end marker is <= end exclusion (can touch)
        if t_end > self._t_exclude_end:
            t_end = self._t_exclude_end
        # Ensure it's also >= start exclusion
        if t_end < self._t_exclude_start:
            t_end = self._t_exclude_start + 0.01
        idx_end = np.argmin(np.abs(decay_result.t - t_end))
        self._end_x = t_end
        self._end_y = decay_result.l_ir[idx_end]
        
        # Noise floor
        self._noise_level = decay_result.noise_level
        
        # RT60 and intersection point
        self._rt60 = decay_result.rt60_s
        self._t_intersect = decay_result.t_intersect
        
        self._init_ui()
        self._update_display()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Info text
        info_layout = QHBoxLayout()
        info_layout.addStretch()
        
        self.info_label = QLabel(
            "Regression: Select start/end time and noise floor. The regression is automatically calculated."
        )
        self.info_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label, stretch=1)
        
        layout.addLayout(info_layout)
        
        # Plot
        self.plot = pg.PlotWidget()
        self.plot.setBackground('#1e1e2e')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel('left', 'Level', units='dB')
        self.plot.setLabel('bottom', 'Time', units='s')
        self.plot.getPlotItem().setMenuEnabled(False)
        
        # Excluded regions (shading) - start and end
        self.exclude_region_start = pg.LinearRegionItem(
            values=[0, self._t_exclude_start],
            orientation='vertical',
            movable=False,
            brush=pg.mkBrush(243, 139, 168, 40),
            pen=pg.mkPen(None)
        )
        self.plot.addItem(self.exclude_region_start)
        
        self.exclude_region_end = pg.LinearRegionItem(
            values=[self._t_exclude_end, self._decay.t[-1]],
            orientation='vertical',
            movable=False,
            brush=pg.mkBrush(243, 139, 168, 40),
            pen=pg.mkPen(None)
        )
        self.plot.addItem(self.exclude_region_end)
        
        # Start exclusion boundary (vertical line, draggable)
        self.exclude_start_line = pg.InfiniteLine(
            pos=self._t_exclude_start, angle=90, movable=True,
            pen=pg.mkPen('#f38ba8', width=2, style=Qt.DotLine),
            label="Start Exclusion", labelOpts={'position': 0.98, 'color': '#f38ba8'}
        )
        self.exclude_start_line.sigPositionChanged.connect(self._on_exclude_start_changed)
        self.plot.addItem(self.exclude_start_line)
        
        # End exclusion boundary (vertical line, draggable)
        self.exclude_end_line = pg.InfiniteLine(
            pos=self._t_exclude_end, angle=90, movable=True,
            pen=pg.mkPen('#f38ba8', width=2, style=Qt.DotLine),
            label="End Exclusion", labelOpts={'position': 0.97, 'color': '#f38ba8'}
        )
        self.exclude_end_line.sigPositionChanged.connect(self._on_exclude_end_changed)
        self.plot.addItem(self.exclude_end_line)
        
        # Level progression (background)
        self.decay_curve = self.plot.plot(
            self._decay.t, self._decay.l_ir, 
            pen=pg.mkPen('#89b4fa', width=1.5)
        )
        
        # Decay line (redrawn on updates)
        self.trend_line = self.plot.plot(
            [], [], 
            pen=pg.mkPen('#fab387', width=2.5)
        )
        
        # Noise floor line (horizontal, always draggable)
        self.noise_line = pg.InfiniteLine(
            pos=self._noise_level, angle=0, movable=True,
            pen=pg.mkPen('#89b4fa', width=2, style=Qt.DashLine),
            label="Noise Floor", labelOpts={'position': 0.95, 'color': '#89b4fa'}
        )
        self.noise_line.sigPositionChanged.connect(self._on_noise_changed)
        self.plot.addItem(self.noise_line)
        
        # Start line (vertical, draggable) - gray
        self.start_line = pg.InfiniteLine(
            pos=self._start_x, angle=90, movable=True,
            pen=pg.mkPen('#6c7086', width=2),
            label="Start", labelOpts={'position': 0.9, 'color': '#6c7086'}
        )
        self.start_line.sigPositionChanged.connect(self._on_start_line_changed)
        self.plot.addItem(self.start_line)
        
        # End line (vertical, draggable) - gray
        self.end_line = pg.InfiniteLine(
            pos=self._end_x, angle=90, movable=True,
            pen=pg.mkPen('#6c7086', width=2),
            label="End", labelOpts={'position': 0.8, 'color': '#6c7086'}
        )
        self.end_line.sigPositionChanged.connect(self._on_end_line_changed)
        self.plot.addItem(self.end_line)
        
        # Intersection point marker (yellow, not draggable)
        self.intersect_point = pg.ScatterPlotItem(
            pos=[(0, 0)],
            size=14,
            pen=pg.mkPen('#f9e2af', width=2),
            brush=pg.mkBrush('#f9e2af'),
            symbol='x'
        )
        self.plot.addItem(self.intersect_point)
        
        # Axis limits
        max_l = np.max(self._decay.l_ir)
        min_l = np.min(self._decay.l_ir)
        self.plot.setYRange(max(min_l, self._noise_level - 20), max_l + 5)
        self.plot.setXRange(0, self._decay.t[-1])
        
        layout.addWidget(self.plot, stretch=1)
        
        # Info row at bottom
        info_layout = QHBoxLayout()
        info_layout.setSpacing(15)
        
        self.rt60_label = QLabel(f"RT60: {self._rt60:.2f} s")
        self.rt60_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #a6adc8;")
        info_layout.addWidget(self.rt60_label)
        
        self.start_label = QLabel("")
        self.start_label.setStyleSheet("font-size: 11px; color: #a6adc8;")
        info_layout.addWidget(self.start_label)
        
        self.end_label = QLabel("")
        self.end_label.setStyleSheet("font-size: 11px; color: #a6adc8;")
        info_layout.addWidget(self.end_label)
        
        self.noise_label = QLabel(f"Noise Floor: {self._noise_level:.1f} dB")
        self.noise_label.setStyleSheet("font-size: 11px; color: #a6adc8;")
        info_layout.addWidget(self.noise_label)
        
        self.intersect_label = QLabel(f"Intersection: {self._t_intersect:.2f} s")
        self.intersect_label.setStyleSheet("font-size: 11px; color: #a6adc8;")
        info_layout.addWidget(self.intersect_label)
        
        info_layout.addStretch()
        layout.addLayout(info_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        btn_style = """
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                padding: 8px 20px;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #45475a;
            }
        """
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(btn_style)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        accept_btn = QPushButton("Apply")
        accept_btn.setStyleSheet(btn_style.replace("#313244", "#45475a").replace("#45475a", "#585b70"))
        accept_btn.clicked.connect(self._accept)
        btn_layout.addWidget(accept_btn)
        
        layout.addLayout(btn_layout)
    
    def _on_exclude_start_changed(self):
        """Start exclusion line was moved."""
        new_pos = self.exclude_start_line.value()
        # Constrain: must be >= 0, <= end of RIR, and at least 100ms before end exclusion
        new_pos = max(0.0, min(new_pos, self._decay.t[-1], self._t_exclude_end - 0.1))
        self._t_exclude_start = new_pos
        self.exclude_start_line.setValue(new_pos)
        # Update exclusion region
        self.exclude_region_start.setRegion([0, new_pos])
        # Constrain regression markers to be within non-excluded area (can touch exclusion)
        if self._start_x < new_pos:
            self._start_x = new_pos
            self.start_line.setValue(self._start_x)
        self._update_display()
    
    def _on_exclude_end_changed(self):
        """End exclusion line was moved."""
        new_pos = self.exclude_end_line.value()
        # Constrain: must be >= 0, <= end of RIR, and at least 100ms after start exclusion
        new_pos = max(0.0, self._t_exclude_start + 0.1, min(new_pos, self._decay.t[-1]))
        self._t_exclude_end = new_pos
        self.exclude_end_line.setValue(new_pos)
        # Update exclusion region
        self.exclude_region_end.setRegion([new_pos, self._decay.t[-1]])
        # Constrain regression markers to be within non-excluded area (can touch exclusion)
        if self._end_x > new_pos:
            self._end_x = new_pos
            self.end_line.setValue(self._end_x)
        self._update_display()
    
    def _on_start_line_changed(self):
        """Start line was moved."""
        new_pos = self.start_line.value()
        # Constrain: must be >= start exclusion (can touch), < end regression marker, and <= end exclusion (can touch)
        new_pos = max(self._t_exclude_start, min(new_pos, self._end_x - 0.01, self._t_exclude_end))
        self._start_x = new_pos
        self.start_line.setValue(new_pos)
        self._update_display()
    
    def _on_end_line_changed(self):
        """End line was moved."""
        new_pos = self.end_line.value()
        # Constrain: must be > start regression marker and <= end exclusion (can touch)
        new_pos = max(self._start_x + 0.01, min(new_pos, self._t_exclude_end))
        self._end_x = new_pos
        self.end_line.setValue(new_pos)
        self._update_display()
    
    def _on_noise_changed(self):
        """Noise floor was moved."""
        self._noise_level = self.noise_line.value()
        self.noise_label.setText(f"Noise Floor: {self._noise_level:.1f} dB")
        self._update_display()
    
    def _update_display(self):
        """Update line, RT60 and intersection point."""
        t = self._decay.t
        l_ir = self._decay.l_ir
        
        # Get exact positions from line widgets to ensure sync
        self._start_x = self.start_line.value()
        self._end_x = self.end_line.value()
        self._t_exclude_start = self.exclude_start_line.value()
        self._t_exclude_end = self.exclude_end_line.value()
        
        # Ensure regression markers are within valid bounds (can touch exclusions)
        self._start_x = max(self._t_exclude_start, min(self._start_x, self._t_exclude_end, self._end_x - 0.01))
        self._end_x = max(self._start_x + 0.01, min(self._end_x, self._t_exclude_end))
        
        # Update line widgets if values were constrained
        if abs(self.start_line.value() - self._start_x) > 1e-6:
            self.start_line.setValue(self._start_x)
        if abs(self.end_line.value() - self._end_x) > 1e-6:
            self.end_line.setValue(self._end_x)
        
        # Update exclusion regions
        self.exclude_region_start.setRegion([0, self._t_exclude_start])
        self.exclude_region_end.setRegion([self._t_exclude_end, t[-1]])
        
        # Regression in selected region (must be within non-excluded area, can touch boundaries)
        mask = (t >= self._start_x) & (t <= self._end_x) & (t >= self._t_exclude_start) & (t <= self._t_exclude_end)
        if not np.any(mask):
            return
        
        # Calculate regression slope in the selected region
        p = np.polyfit(t[mask], l_ir[mask], 1)
        slope = p[0]
        
        # Calculate regression value at start marker
        regression_start_y = np.polyval(p, self._start_x)
        
        # Force regression to pass exactly through start marker position
        # This ensures the line always aligns with the start marker visually
        intercept = regression_start_y - slope * self._start_x
        p_forced = np.array([slope, intercept])  # [slope, intercept] for polyval
        
        # Store regression parameters for use in analysis
        self._regression_slope = slope
        self._regression_intercept = intercept
        
        # Start/End-Y for display (using forced regression)
        self._start_y = np.polyval(p_forced, self._start_x)  # Should equal regression_start_y
        self._end_y = np.polyval(p_forced, self._end_x)
        
        # Intersection with noise floor
        if abs(slope) > 1e-6:
            self._t_intersect = (self._noise_level - intercept) / slope
        else:
            self._t_intersect = t[-1]
        
        # Determine endpoint: use end marker if it's before intersection, otherwise use intersection
        t_start = self._start_x
        t_end = min(self._end_x, self._t_intersect) if self._end_x < self._t_intersect else self._t_intersect
        
        # Make sure endpoint is valid
        if t_end < t_start:
            t_end = t_start + 0.01  # Fallback if endpoint is before start
        
        # Use enough points for smooth rendering
        num_points = max(100, int((t_end - t_start) * 1000))  # At least 1 point per ms
        t_plot = np.linspace(t_start, t_end, num_points)
        # Explicitly set endpoints to exact positions to ensure visual connection
        t_plot[0] = t_start
        t_plot[-1] = t_end
        l_plot = np.polyval(p_forced, t_plot)
        # Ensure the start point matches exactly at start marker
        l_plot[0] = self._start_y
        # Ensure the end point matches the regression at endpoint
        if t_end == self._t_intersect:
            l_plot[-1] = self._noise_level  # At intersection with noise floor
        else:
            l_plot[-1] = np.polyval(p_forced, t_end)  # At end marker
        self.trend_line.setData(t_plot, l_plot)
        
        # Calculate RT60
        if abs(p[0]) > 1e-3:
            self._rt60 = 60.0 / abs(p[0])
        else:
            self._rt60 = 0.0
        
        # Labels
        self.start_label.setText(f"Start: {self._start_x:.2f} s")
        self.end_label.setText(f"End: {self._end_x:.2f} s")
        
        # Update intersection point marker
        if 0 < self._t_intersect < t[-1]:
            self.intersect_point.setData(pos=[(self._t_intersect, self._noise_level)])
            self.intersect_point.show()
        else:
            self.intersect_point.hide()
        
        # Update labels
        self.rt60_label.setText(f"RT60: {self._rt60:.2f} s")
        self.intersect_label.setText(f"Intersection: {self._t_intersect:.2f} s")
    
    def _accept(self):
        """Apply clicked."""
        self._accepted = True
        self.accept()
    
    def get_corrected_values(self) -> dict:
        """Returns the corrected values."""
        return {
            'start_x': self._start_x,
            'start_y': self._start_y,
            'end_x': self._end_x,
            'end_y': self._end_y,
            'noise_level': self._noise_level,
            't_intersect': self._t_intersect,
            't_exclude_start': self._t_exclude_start,
            't_exclude_end': self._t_exclude_end,
            'rt60': self._rt60,
            'regression_slope': self._regression_slope,
            'regression_intercept': self._regression_intercept,
        }
    
    def get_corrected_times(self) -> tuple:
        """Returns the corrected start and end times (for analysis)."""
        # Start = start exclusion boundary, End = intersection point
        return (self._t_exclude_start, self._t_intersect)
    
    def was_accepted(self) -> bool:
        """True if the user clicked 'Apply'."""
        return self._accepted


class RoomSchematic(QWidget):
    """Draws a proportional isometric room sketch for visualizing wall pairs."""
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(180, 180)
        self.setCursor(Qt.PointingHandCursor)
        self.active_dims = set() # Set of "L", "B", "H"
        self.dims = (5.0, 4.0, 3.0) # Default dimensions for preview
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
        else:
            super().mousePressEvent(event)
        
    def set_active_dimensions(self, dims: list[str]):
        new_dims = set(dims)
        if self.active_dims != new_dims:
            self.active_dims = new_dims
            self.update()

    def set_dimensions(self, l: float, b: float, h: float):
        # If dimensions are 0, use a default cube for preview
        new_dims = (max(0.1, l), max(0.1, b), max(0.1, h))
        if self.dims != new_dims:
            self.dims = new_dims
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h_widget = self.width(), self.height()
        
        l_m, b_m, h_m = self.dims
        
        # Isometrische Winkel (30 Grad)
        cos30 = 0.866
        sin30 = 0.5
        
        # Calculate maximum scaling so room fits "edge-to-edge"
        # Projected width: (bs + ls) * cos30
        # Projected height: hs + (bs + ls) * sin30
        # We want: scale * (b_m + l_m) * cos30 <= w * 0.9
        # and: scale * (h_m + (b_m + l_m) * sin30) <= h_widget * 0.9
        
        denom_w = (b_m + l_m) * cos30
        denom_h = h_m + (b_m + l_m) * sin30
        
        scale_w = (w * 0.9) / denom_w if denom_w > 0 else 100
        scale_h = (h_widget * 0.9) / denom_h if denom_h > 0 else 100
        
        scale = min(scale_w, scale_h)
        
        ls = l_m * scale
        bs = b_m * scale
        hs = h_m * scale
        
        # Center calculation for perfect centering of projected box
        # Compensate horizontal offset from unequal lengths/widths
        cx = w / 2 - (bs - ls) * cos30 / 2
        # Vertical offset (projected box goes from y=-hs to y=+(bs+ls)*sin30 relative to origin)
        cy = h_widget / 2 - ((bs + ls) * sin30 - hs) / 2
        
        # 3D points -> 2D projection
        # We define points (x=width, y=length, z=height)
        def project(x3d, y3d, z3d):
            x2d = cx + (x3d * cos30) - (y3d * cos30)
            y2d = cy + (x3d * sin30) + (y3d * sin30) - z3d
            return QPointF(x2d, y2d)

        # Corner points of the box
        p0 = project(0, 0, 0)    # bottom-center-front
        p1 = project(bs, 0, 0)   # bottom-right
        p2 = project(bs, ls, 0)  # bottom-back
        p3 = project(0, ls, 0)   # bottom-left
        
        p4 = project(0, 0, hs)   # top-center-front
        p5 = project(bs, 0, hs)  # top-right
        p6 = project(bs, ls, hs) # top-back
        p7 = project(0, ls, hs)  # top-left

        def draw_face(pts, color, highlight=False):
            poly = QPolygonF(pts)
            if highlight:
                # Strong highlight
                painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 200)))
                painter.setPen(QPen(QColor("#ffffff"), 2))
            else:
                # Subtle base appearance
                painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 40)))
                painter.setPen(QPen(QColor("#585b70"), 1))
            painter.drawPolygon(poly)

        # Base colors
        col_l = QColor("#a6e3a1") # Green (Length)
        col_b = QColor("#89b4fa") # Blue (Width)
        col_h = QColor("#f9e2af") # Yellow (Height)
        
        # Draw faces (from back to front for correct overlap)
        # 1. Floor (H-partner bottom)
        draw_face([p0, p1, p2, p3], col_h, "H" in self.active_dims)
        
        # 2. Back walls (L/B partner back)
        draw_face([p1, p2, p6, p5], col_b, "B" in self.active_dims) # Back-Right (belongs to B)
        draw_face([p3, p2, p6, p7], col_l, "L" in self.active_dims) # Back-Left (belongs to L)
        
        # 3. Front walls
        draw_face([p0, p3, p7, p4], col_b, "B" in self.active_dims) # Front-Left
        draw_face([p0, p1, p5, p4], col_l, "L" in self.active_dims) # Front-Right
        
        # 4. Ceiling
        draw_face([p4, p5, p6, p7], col_h, "H" in self.active_dims)


class ShiftZoomPlotWidget(pg.PlotWidget):
    """Custom PlotWidget with Shift key for Y-axis control."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseEnabled(x=True, y=False)
        self._shift_active = False
    
    def wheelEvent(self, ev: QWheelEvent):
        if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.setMouseEnabled(x=False, y=True)
            super().wheelEvent(ev)
            self.setMouseEnabled(x=True, y=False)
        else:
            super().wheelEvent(ev)

    def mousePressEvent(self, ev):
        if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self._shift_active = True
            self.setMouseEnabled(x=False, y=True)
        else:
            self._shift_active = False
            self.setMouseEnabled(x=True, y=False)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            if not self._shift_active:
                self._shift_active = True
                self.setMouseEnabled(x=False, y=True)
        else:
            if self._shift_active:
                self._shift_active = False
                self.setMouseEnabled(x=True, y=False)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        self._shift_active = False
        self.setMouseEnabled(x=True, y=False)


class FlutterEchoWidget(QWidget):
    """Widget for Room Analysis with interactive peak selection."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio_data: Optional[np.ndarray] = None
        self._sample_rate: int = 48000
        self._result: Optional[FlutterEchoResult] = None
        self._selected_peak_index: int = 0
        
        self.setFocusPolicy(Qt.StrongFocus)
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # --- PLOT BEREICH ---
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        
        header_layout = QHBoxLayout()
        title_label = QLabel("Distogram")
        title_label.setStyleSheet("color: #cdd6f4; font-size: 14px; font-weight: bold;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        source_label = QLabel('<a href="file:///Users/dominiknaschert/Desktop/GitHub/Audio_Analyzer/Literatur/000213.pdf" style="color: #89b4fa;">Schanda et al. (DAGA 2023)</a>')
        source_label.setOpenExternalLinks(True)
        source_label.setStyleSheet("font-size: 11px;")
        header_layout.addWidget(source_label)
        plot_layout.addLayout(header_layout)
        
        self.plot = ShiftZoomPlotWidget()
        self.plot.setBackground('#1e1e2e')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel('bottom', 'Distance to Reflection Surfaces', units='m')
        self.plot.getPlotItem().setMenuEnabled(False)
        
        self.curve = self.plot.plot(pen=pg.mkPen('#89b4fa', width=1.5))
        
        # Reference lines for L, B, H
        self.line_l = pg.InfiniteLine(angle=90, pen=pg.mkPen('#a6e3a1', width=1, style=Qt.DashLine), label="L", labelOpts={'position': 0.9, 'color': '#a6e3a1'})
        self.line_b = pg.InfiniteLine(angle=90, pen=pg.mkPen('#89b4fa', width=1, style=Qt.DashLine), label="B", labelOpts={'position': 0.8, 'color': '#89b4fa'})
        self.line_h = pg.InfiniteLine(angle=90, pen=pg.mkPen('#f9e2af', width=1, style=Qt.DashLine), label="H", labelOpts={'position': 0.7, 'color': '#f9e2af'})
        for line in [self.line_l, self.line_b, self.line_h]:
            self.plot.addItem(line)
            line.hide()

        self.peak_scatter = pg.ScatterPlotItem(pen=pg.mkPen('#cdd6f4', width=2), brush=pg.mkBrush('#6c7086'), size=10)
        self.peak_scatter.sigClicked.connect(self._on_peak_clicked)
        self.plot.addItem(self.peak_scatter)
        
        self.selected_peak_marker = pg.ScatterPlotItem(pen=pg.mkPen('#89b4fa', width=3), brush=pg.mkBrush('#89b4fa'), size=14)
        self.plot.addItem(self.selected_peak_marker)
        
        self.selected_peak_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('#89b4fa', width=2, style=Qt.PenStyle.DashLine))
        self.plot.addItem(self.selected_peak_line)
        self.selected_peak_line.hide()
        
        self.peak_text = pg.TextItem(anchor=(0.5, 1.2))
        self.peak_text.setColor('#89b4fa')
        self.plot.addItem(self.peak_text)
        self.peak_text.hide()
        
        plot_layout.addWidget(self.plot)
        splitter.addWidget(plot_container)
        
        # --- LOWER AREA (COMPACT) ---
        results_container = QWidget()
        results_layout = QHBoxLayout(results_container)
        results_layout.setContentsMargins(30, 8, 10, 8)  # More margin left for spacing from edge
        results_layout.setSpacing(25)  # More spacing between columns
        
        # Left column: Data & inputs
        data_column = QVBoxLayout()
        data_column.setSpacing(10)
        
        # Results table (3 rows: Distance, Repetition Freq., Amplitude)
        self.results_table = QTableWidget(3, 2)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.horizontalHeader().setVisible(False)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4; gridline-color: #313244; border: 1px solid #313244;")
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setFocusPolicy(Qt.NoFocus)
        self.results_table.setFixedHeight(110) # Adjusted height for 3 rows
        
        headers = ["Distance", "Repetition Freq.", "Amplitude"]
        for i, h in enumerate(headers):
            self.results_table.setItem(i, 0, QTableWidgetItem(h))
            self.results_table.setItem(i, 1, QTableWidgetItem("–"))
            self.results_table.item(i, 1).setTextAlignment(Qt.AlignCenter)
        
        data_column.addWidget(self.results_table)
        
        data_column.addSpacing(15) # More spacing between table and room dimensions
        
        # Room dimensions
        room_group = QGroupBox("Room Dimensions (L x B x H)")
        room_v_layout = QVBoxLayout(room_group)
        
        room_inputs = QHBoxLayout()
        validator = QDoubleValidator(0.0, 100.0, 2)
        validator.setNotation(QDoubleValidator.StandardNotation)
        
        input_style = """
            QLineEdit {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                padding: 4px 8px;
                border-radius: 4px;
                min-width: 50px;
            }
            QLineEdit:focus {
                border-color: #89b4fa;
            }
        """
        
        self.length_input = QLineEdit()
        self.length_input.setPlaceholderText("L [m]")
        self.length_input.setValidator(validator)
        self.length_input.setStyleSheet(input_style)
        self.length_input.textChanged.connect(self._on_room_dimensions_changed)
        
        self.width_input = QLineEdit()
        self.width_input.setPlaceholderText("B [m]")
        self.width_input.setValidator(validator)
        self.width_input.setStyleSheet(input_style)
        self.width_input.textChanged.connect(self._on_room_dimensions_changed)
        
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText("H [m]")
        self.height_input.setValidator(validator)
        self.height_input.setStyleSheet(input_style)
        self.height_input.textChanged.connect(self._on_room_dimensions_changed)
        
        room_inputs.addWidget(QLabel("L:")); room_inputs.addWidget(self.length_input)
        room_inputs.addWidget(QLabel("B:")); room_inputs.addWidget(self.width_input)
        room_inputs.addWidget(QLabel("H:")); room_inputs.addWidget(self.height_input)
        room_v_layout.addLayout(room_inputs)
        
        data_column.addWidget(room_group)
        
        # RT60 Plot button (removed from bottom of data column, moved to right)
        # (Previously here: data_column.addLayout(rt60_layout))
        
        data_column.addStretch() # Pushes everything up
        
        results_layout.addLayout(data_column, 2) # 20-30% width
        
        # Right column: Room schematic & RT60 (Large)
        schematic_column = QVBoxLayout()
        schematic_column.setSpacing(0)
        
        self.room_schematic = RoomSchematic()
        self.room_schematic.clicked.connect(self._show_analysis_pipeline)
        schematic_column.addWidget(self.room_schematic, stretch=1)
        
        # RT60 display & Plot button at bottom right
        rt60_footer = QHBoxLayout()
        rt60_footer.addStretch()
        
        self.rt60_label = QLabel("RT60: –")
        self.rt60_label.setStyleSheet("color: #cdd6f4; font-size: 11px; padding: 2px 5px;")
        rt60_footer.addWidget(self.rt60_label)
        
        self.rt60_plot_btn = QPushButton("Plot")
        self.rt60_plot_btn.setToolTip("Show level decay and regression")
        self.rt60_plot_btn.setFixedWidth(50)
        self.rt60_plot_btn.setEnabled(False)
        self.rt60_plot_btn.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                font-size: 10px;
                padding: 2px;
            }
            QPushButton:hover {
                background-color: #45475a;
            }
            QPushButton:disabled {
                color: #585b70;
            }
        """)
        self.rt60_plot_btn.clicked.connect(self._show_rt60_details)
        rt60_footer.addWidget(self.rt60_plot_btn)
        
        schematic_column.addLayout(rt60_footer)
        
        results_layout.addLayout(schematic_column, 5) # 70-80% width
        
        splitter.addWidget(results_container)
        splitter.setSizes([300, 350]) # Lower area slightly larger (pulled up)
        layout.addWidget(splitter)

    def keyPressEvent(self, event: QKeyEvent):
        if self._result and self._result.peaks:
            if event.key() == Qt.Key_Left:
                # Previous peak (smaller index / smaller distance)
                self._selected_peak_index = (self._selected_peak_index - 1) % len(self._result.peaks)
                self._update_peak_display()
                event.accept()
            elif event.key() == Qt.Key_Right:
                # Next peak (larger index / larger distance)
                self._selected_peak_index = (self._selected_peak_index + 1) % len(self._result.peaks)
                self._update_peak_display()
                event.accept()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def _get_input_value(self, input_widget: QLineEdit) -> float:
        try:
            text = input_widget.text().replace(",", ".")
            return float(text) if text else 0.0
        except ValueError:
            return 0.0

    def set_audio_data(self, data: np.ndarray, sample_rate: int):
        self._audio_data = data
        self._sample_rate = sample_rate

    def analyze(self):
        """
        Performs Room Analysis:
        1. Calculates level progression
        2. Shows RT60 correction dialog
        3. On confirmation: Performs peak detection with corrected values
        """
        if self._audio_data is None: 
            return
        
        # Step 1: Calculate level progression
        config = FlutterDetectionConfig(sample_rate=self._sample_rate)
        decay_result = compute_decay_curve(self._audio_data, config)
        
        # Step 2: Show RT60 correction dialog
        dialog = RT60CorrectionDialog(decay_result, self)
        dialog.exec()
        
        if not dialog.was_accepted():
            # User cancelled
            return
        
        # Step 3: Apply corrected values and perform analysis
        values = dialog.get_corrected_values()
        
        # Use the regression parameters calculated in the dialog
        config_corrected = FlutterDetectionConfig(
            sample_rate=self._sample_rate,
            t_start_fit_override=values['t_exclude_start'],
            t_end_fit_override=values['t_intersect'],
            noise_level_override=values['noise_level'],
            # Use the regression parameters from the dialog (user-selected markers)
            manual_slope=values['regression_slope'],
            manual_intercept=values['regression_intercept'],
        )
        
        self._result = analyze_flutter(self._audio_data, config_corrected)
        # Store regression marker positions for RT60 plot
        self._result.t_regression_start = values['start_x']
        self._result.t_regression_end = values['end_x']
        self._selected_peak_index = 0
        self._update_display()

    def _on_peak_clicked(self, scatter, points):
        if len(points) > 0 and self._result:
            clicked_x = points[0].pos().x()
            self._selected_peak_index = np.argmin([abs(p.distance_m - clicked_x) for p in self._result.peaks])
            self._update_peak_display()

    def _update_display(self):
        if not self._result: return
        self.curve.setData(self._result.distances, self._result.amplitudes)
        self.peak_scatter.setData(x=[p.distance_m for p in self._result.peaks], y=[p.amplitude for p in self._result.peaks])
        # Setze X-Bereich auf 0-20m (Standard)
        if len(self._result.distances) > 0:
            max_dist = min(20.0, np.max(self._result.distances) * 1.1)  # 10% Padding, max 20m
            self.plot.setXRange(0, max_dist)
        self._update_peak_display()

    def _update_peak_display(self):
        if not self._result or not self._result.peaks: return
        peak = self._result.peaks[self._selected_peak_index]
        
        self.selected_peak_marker.setData(x=[peak.distance_m], y=[peak.amplitude])
        self.selected_peak_line.setPos(peak.distance_m)
        self.selected_peak_line.show()
        self.peak_text.setText(f"{peak.distance_m:.2f} m")
        self.peak_text.setPos(peak.distance_m, peak.amplitude)
        self.peak_text.show()
        
        self.results_table.item(0, 1).setText(f"{peak.distance_m:.2f} m")
        self.results_table.item(1, 1).setText(f"{peak.repetition_frequency_hz:.1f} Hz")
        self.results_table.item(2, 1).setText(f"{peak.amplitude:.0f}")
        
        # Update RT60 display
        if self._result.rt60_s > 0:
            self.rt60_label.setText(f"RT60: {self._result.rt60_s:.2f} s")
            self.rt60_plot_btn.setEnabled(True)
        else:
            self.rt60_label.setText("RT60: –")
            self.rt60_plot_btn.setEnabled(False)
        
        self._update_wall_assignment()

    @Slot()
    def _show_rt60_details(self):
        """Opens the detail dialog for RT60."""
        if self._result:
            dialog = RT60DetailDialog(self._result, self)
            dialog.exec()

    @Slot()
    def _show_analysis_pipeline(self):
        """Opens the detail dialog for the complete analysis pipeline."""
        if self._result:
            dialog = AnalysisPipelineDialog(self._result, self)
            dialog.exec()

    def _on_room_dimensions_changed(self):
        l = self._get_input_value(self.length_input)
        b = self._get_input_value(self.width_input)
        h = self._get_input_value(self.height_input)
        
        # Update proportional model
        self.room_schematic.set_dimensions(l, b, h)
        
        # Reference lines
        if l > 0: self.line_l.setPos(l); self.line_l.show()
        else: self.line_l.hide()
        if b > 0: self.line_b.setPos(b); self.line_b.show()
        else: self.line_b.hide()
        if h > 0: self.line_h.setPos(h); self.line_h.show()
        else: self.line_h.hide()
        
        self._update_wall_assignment()

    def _update_wall_assignment(self):
        if not self._result or not self._result.peaks: 
            self.room_schematic.set_active_dimensions([])
            return
            
        peak_dist = self._result.peaks[self._selected_peak_index].distance_m
        l = self._get_input_value(self.length_input)
        b = self._get_input_value(self.width_input)
        h = self._get_input_value(self.height_input)
        
        matches = []
        active_dims = []
        
        if l > 0 and abs(peak_dist - l) / l < 0.05:
            matches.append(f"L ({l:.2f}m)"); active_dims.append("L")
        if b > 0 and abs(peak_dist - b) / b < 0.05:
            matches.append(f"B ({b:.2f}m)"); active_dims.append("B")
        if h > 0 and abs(peak_dist - h) / h < 0.05:
            matches.append(f"H ({h:.2f}m)"); active_dims.append("H")
        
        self.room_schematic.set_active_dimensions(active_dims)

    def clear(self):
        self._result = None
        self.curve.clear()
        self.peak_scatter.clear()
        self.selected_peak_marker.clear()
        self.selected_peak_line.hide()
        self.peak_text.hide()
        self.line_l.hide(); self.line_b.hide(); self.line_h.hide()
        self.room_schematic.set_active_dimensions([])
        self.length_input.clear(); self.width_input.clear(); self.height_input.clear()
        self.rt60_label.setText("RT60: –")
        self.rt60_plot_btn.setEnabled(False)
        for i in range(3): self.results_table.item(i, 1).setText("–")
