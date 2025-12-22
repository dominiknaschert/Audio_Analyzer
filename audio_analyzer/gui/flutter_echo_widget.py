"""
Flatterecho-Analyse Widget
Zeigt Distogramm-Daten an und visualisiert die Zuordnung zu Raumdimensionen.
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
    """Dialog zur Visualisierung der gesamten Room-Analysis-Pipeline."""
    def __init__(self, result: FlutterEchoResult, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Room-Analysis Pipeline Details (nach Schanda et al.)")
        self.resize(1100, 850)
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")
        
        layout = QVBoxLayout(self)
        
        # Grid für die 6 Plots
        grid = QGridLayout()
        grid.setSpacing(15)
        
        # Hilfsfunktion für Plot-Erstellung
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
        p1 = create_plot("1. Originale Impulsantwort (RIR)", "Zeit", units_x="s")
        if result.rir_raw is not None:
            p1.plot(result.t, result.rir_raw, pen=pg.mkPen('#6c7086', width=0.5))
        grid.addWidget(p1, 0, 0)
        
        # 2. Bandpass gefiltert
        p2 = create_plot(f"2. Bandpass gefiltert ({result.flutter_tonality_hz:.0f} Hz)", "Zeit", units_x="s")
        if result.rir_bp is not None:
            p2.plot(result.t, result.rir_bp, pen=pg.mkPen('#89b4fa', width=0.5))
        grid.addWidget(p2, 0, 1)
        
        # 3. Pegel-Zeit mit Regressionen
        p3 = create_plot("3. Pegel-Zeit & Regressionen", "Zeit", "Pegel", units_x="s", units_y="dB")
        if result.l_ir is not None:
            p3.plot(result.t, result.l_ir, pen=pg.mkPen('#45475a', width=1))
            # Bereich I markieren (Vorschattierung)
            vspan = pg.LinearRegionItem([0, 0.2], movable=False, brush=QBrush(QColor(243, 139, 168, 40)))
            p3.addItem(vspan)
            
            # Regressionen
            if result.p_decay is not None:
                t_plot = np.linspace(0.2, result.t_intersect + 0.1, 100)
                l_plot = np.polyval(result.p_decay, t_plot)
                p3.plot(t_plot, l_plot, pen=pg.mkPen('#a6e3a1', width=2))
                
                # Rauschboden
                p3.addLine(y=result.noise_level, pen=pg.mkPen('#89b4fa', width=2, style=Qt.DashLine))
                
                # Schnittpunkt
                p3.plot([result.t_intersect], [result.noise_level], pen=None, symbol='o', symbolSize=8, symbolBrush='#f9e2af')
                
            # Trendlinie
            if result.l_trend is not None:
                p3.plot(result.t, result.l_trend, pen=pg.mkPen('#cdd6f4', width=1, style=Qt.DotLine))
            
            p3.setYRange(max(np.min(result.l_ir), result.noise_level - 20), 5)
        grid.addWidget(p3, 1, 0)
        
        # 4. Korrigiertes Signal L_FE
        p4 = create_plot("4. Korrigiertes Signal L_FE (Echos isoliert)", "Zeit", "Amplitude", units_x="s", units_y="dB")
        if result.l_fe is not None:
            mask = (result.t >= 0.2) & (result.t <= result.t_intersect)
            p4.plot(result.t[mask], result.l_fe[mask], pen=pg.mkPen('#f38ba8', width=1))
        grid.addWidget(p4, 1, 1)
        
        # 5. Autokorrelation
        p5 = create_plot("5. Autokorrelationsfunktion (ACF)", "Lag", units_x="s")
        if result.acf is not None:
            # fs berechnen aus t
            fs = 1.0 / (result.t[1] - result.t[0]) if len(result.t) > 1 else 48000
            t_acf = np.arange(len(result.acf)) / fs
            p5.plot(t_acf, result.acf, pen=pg.mkPen('#a6e3a1', width=1.5))
            p5.setXRange(0, 0.4)
        grid.addWidget(p5, 2, 0)
        
        # 6. Distogramm
        p6 = create_plot("6. Distogramm (FFT der ACF)", "Abstand", units_x="m")
        p6.plot(result.distances, result.amplitudes, pen=pg.mkPen('#cba6f7', width=1.5))
        # Peaks markieren
        if result.peaks:
            p6.plot([p.distance_m for p in result.peaks], [p.amplitude for p in result.peaks], 
                   pen=None, symbol='t', symbolSize=8, symbolBrush='#f38ba8')
        grid.addWidget(p6, 2, 1)
        
        layout.addLayout(grid)
        
        # Close Button
        close_btn = QPushButton("Schließen")
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
    """Dialog zur Visualisierung des Pegelabfalls und der RT60-Regression."""
    def __init__(self, result: FlutterEchoResult, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RT60 Details - Pegelabfall")
        self.resize(800, 500)
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")
        
        layout = QVBoxLayout(self)
        
        # Info Label
        info_label = QLabel(f"Extrapolierte Nachhallzeit RT60: {result.rt60_s:.2f} s")
        info_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #a6e3a1; margin-bottom: 5px;")
        layout.addWidget(info_label)
        
        # Plot
        self.plot = pg.PlotWidget()
        self.plot.setBackground('#1e1e2e')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel('left', 'Pegel', units='dB')
        self.plot.setLabel('bottom', 'Zeit', units='s')
        
        if result.t is not None and result.l_ir is not None:
            # Pegelverlauf
            self.plot.plot(result.t, result.l_ir, pen=pg.mkPen('#89b4fa', width=1), name="Signal (Gefiltert)")
            
            # Trendlinie
            if result.l_trend is not None:
                self.plot.plot(result.t, result.l_trend, pen=pg.mkPen('#fab387', width=2, style=Qt.DashLine), name="Trend / RT60 Fit")
            
            # Achsen sinnvoll skalieren (Pegelbereich)
            min_l = np.min(result.l_ir)
            max_l = np.max(result.l_ir)
            self.plot.setYRange(max(min_l, max_l - 80), max_l + 5)
            
        layout.addWidget(self.plot)
        
        # Close Button
        close_btn = QPushButton("Schließen")
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
    Dialog zur interaktiven Korrektur der RT60-Regression.
    
    Zwei Modi:
    - Regression: Wähle Start-/Endzeit, Regression wird automatisch berechnet
    - Manuell: Setze Start (X,Y), Ende (X,Y), Rauschboden und ausgeschlossenen Bereich
    """
    
    def __init__(self, decay_result: DecayCurveResult, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RT60 Korrektur")
        self.resize(1000, 700)
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")
        
        self._decay = decay_result
        self._accepted = False
        self._mode = "regression"  # "regression" oder "manual"
        
        # Ausgeschlossener Bereich (Direktschall)
        self._t_exclude = decay_result.t_start_fit  # Standard: 0.2s
        
        # Initiale Werte aus der automatischen Berechnung
        t_start = decay_result.t_start_fit
        idx_start = np.argmin(np.abs(decay_result.t - t_start))
        self._start_x = t_start
        self._start_y = decay_result.l_ir[idx_start]
        
        t_end = decay_result.t_end_fit
        idx_end = np.argmin(np.abs(decay_result.t - t_end))
        self._end_x = t_end
        self._end_y = decay_result.l_ir[idx_end]
        
        # Rauschboden
        self._noise_level = decay_result.noise_level
        
        # RT60 und Schnittpunkt
        self._rt60 = decay_result.rt60_s
        self._t_intersect = decay_result.t_intersect
        
        self._init_ui()
        self._update_mode()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Modus-Auswahl oben
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Modus:")
        mode_label.setStyleSheet("font-weight: bold;")
        mode_layout.addWidget(mode_label)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Regression", "Manuell"])
        self.mode_combo.setStyleSheet("""
            QComboBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                padding: 5px 10px;
                border-radius: 4px;
                min-width: 120px;
            }
        """)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        
        mode_layout.addStretch()
        
        # Info-Text (ändert sich je nach Modus)
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        self.info_label.setWordWrap(True)
        mode_layout.addWidget(self.info_label, stretch=1)
        
        layout.addLayout(mode_layout)
        
        # Plot
        self.plot = pg.PlotWidget()
        self.plot.setBackground('#1e1e2e')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel('left', 'Pegel', units='dB')
        self.plot.setLabel('bottom', 'Zeit', units='s')
        self.plot.getPlotItem().setMenuEnabled(False)
        
        # Ausgeschlossener Bereich (Schattierung)
        self.exclude_region = pg.LinearRegionItem(
            values=[0, self._t_exclude],
            orientation='vertical',
            movable=False,
            brush=pg.mkBrush(243, 139, 168, 40),
            pen=pg.mkPen(None)
        )
        self.plot.addItem(self.exclude_region)
        
        # Ausschluss-Grenze (vertikale Linie, nur in Manuell-Modus ziehbar)
        self.exclude_line = pg.InfiniteLine(
            pos=self._t_exclude, angle=90, movable=False,
            pen=pg.mkPen('#f38ba8', width=2, style=Qt.DotLine),
            label="Ausschluss", labelOpts={'position': 0.98, 'color': '#f38ba8'}
        )
        self.exclude_line.sigPositionChanged.connect(self._on_exclude_changed)
        self.plot.addItem(self.exclude_line)
        
        # Pegelverlauf (Hintergrund)
        self.decay_curve = self.plot.plot(
            self._decay.t, self._decay.l_ir, 
            pen=pg.mkPen('#89b4fa', width=1.5)
        )
        
        # Abfallgerade (wird bei Updates neu gezeichnet)
        self.trend_line = self.plot.plot(
            [], [], 
            pen=pg.mkPen('#fab387', width=2.5)
        )
        
        # Rauschboden-Linie (horizontal, IMMER ziehbar)
        self.noise_line = pg.InfiniteLine(
            pos=self._noise_level, angle=0, movable=True,
            pen=pg.mkPen('#89b4fa', width=2, style=Qt.DashLine),
            label="Rauschboden", labelOpts={'position': 0.95, 'color': '#89b4fa'}
        )
        self.noise_line.sigPositionChanged.connect(self._on_noise_changed)
        self.plot.addItem(self.noise_line)
        
        # Start-Linie für Regression-Modus (vertikal, ziehbar)
        self.start_line = pg.InfiniteLine(
            pos=self._start_x, angle=90, movable=True,
            pen=pg.mkPen('#a6e3a1', width=2),
            label="Start", labelOpts={'position': 0.9, 'color': '#a6e3a1'}
        )
        self.start_line.sigPositionChanged.connect(self._on_start_line_changed)
        self.plot.addItem(self.start_line)
        
        # End-Linie für Regression-Modus (vertikal, ziehbar)
        self.end_line = pg.InfiniteLine(
            pos=self._end_x, angle=90, movable=True,
            pen=pg.mkPen('#f38ba8', width=2),
            label="Ende", labelOpts={'position': 0.8, 'color': '#f38ba8'}
        )
        self.end_line.sigPositionChanged.connect(self._on_end_line_changed)
        self.plot.addItem(self.end_line)
        
        # Startpunkt für Manuell-Modus (grün, ziehbar)
        self.start_point = pg.TargetItem(
            pos=(self._start_x, self._start_y),
            size=14,
            pen=pg.mkPen('#a6e3a1', width=2),
            brush=pg.mkBrush('#a6e3a1'),
            movable=True,
            label="Start",
            labelOpts={'offset': (10, -10), 'color': '#a6e3a1'}
        )
        self.start_point.sigPositionChanged.connect(self._on_start_point_changed)
        self.plot.addItem(self.start_point)
        
        # Endpunkt für Manuell-Modus (rot, ziehbar)
        self.end_point = pg.TargetItem(
            pos=(self._end_x, self._end_y),
            size=14,
            pen=pg.mkPen('#f38ba8', width=2),
            brush=pg.mkBrush('#f38ba8'),
            movable=True,
            label="Ende",
            labelOpts={'offset': (10, -10), 'color': '#f38ba8'}
        )
        self.end_point.sigPositionChanged.connect(self._on_end_point_changed)
        self.plot.addItem(self.end_point)
        
        # Schnittpunkt-Marker (gelb, nicht ziehbar)
        self.intersect_point = pg.ScatterPlotItem(
            pos=[(0, 0)],
            size=14,
            pen=pg.mkPen('#f9e2af', width=2),
            brush=pg.mkBrush('#f9e2af'),
            symbol='x'
        )
        self.plot.addItem(self.intersect_point)
        
        # Achsenlimits
        max_l = np.max(self._decay.l_ir)
        min_l = np.min(self._decay.l_ir)
        self.plot.setYRange(max(min_l, self._noise_level - 20), max_l + 5)
        self.plot.setXRange(0, self._decay.t[-1])
        
        layout.addWidget(self.plot, stretch=1)
        
        # Info-Zeile unten
        info_layout = QHBoxLayout()
        info_layout.setSpacing(15)
        
        self.rt60_label = QLabel(f"RT60: {self._rt60:.2f} s")
        self.rt60_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #a6e3a1;")
        info_layout.addWidget(self.rt60_label)
        
        self.start_label = QLabel("")
        self.start_label.setStyleSheet("font-size: 11px; color: #a6e3a1;")
        info_layout.addWidget(self.start_label)
        
        self.end_label = QLabel("")
        self.end_label.setStyleSheet("font-size: 11px; color: #f38ba8;")
        info_layout.addWidget(self.end_label)
        
        self.noise_label = QLabel(f"Rauschboden: {self._noise_level:.1f} dB")
        self.noise_label.setStyleSheet("font-size: 11px; color: #89b4fa;")
        info_layout.addWidget(self.noise_label)
        
        self.intersect_label = QLabel(f"Schnittpunkt: {self._t_intersect:.2f} s")
        self.intersect_label.setStyleSheet("font-size: 11px; color: #f9e2af;")
        info_layout.addWidget(self.intersect_label)
        
        self.exclude_label = QLabel(f"Ausschluss: {self._t_exclude:.2f} s")
        self.exclude_label.setStyleSheet("font-size: 11px; color: #f38ba8;")
        info_layout.addWidget(self.exclude_label)
        
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
        
        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.setStyleSheet(btn_style)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        accept_btn = QPushButton("Übernehmen")
        accept_btn.setStyleSheet(btn_style.replace("#313244", "#45475a").replace("#45475a", "#585b70"))
        accept_btn.clicked.connect(self._accept)
        btn_layout.addWidget(accept_btn)
        
        layout.addLayout(btn_layout)
    
    def _on_mode_changed(self, text: str):
        """Modus wurde geändert."""
        old_mode = self._mode
        self._mode = "regression" if text == "Regression" else "manual"
        
        # Wenn von Regression auf Manuell gewechselt wird:
        # Die Punkte mit den Werten der Regression an Start/Ende setzen
        if old_mode == "regression" and self._mode == "manual":
            t = self._decay.t
            l_ir = self._decay.l_ir
            
            # Regression im aktuellen Bereich berechnen
            mask = (t >= self._start_x) & (t <= self._end_x)
            if np.any(mask):
                p = np.polyfit(t[mask], l_ir[mask], 1)
                # Y-Werte an Start und Ende der Regression
                self._start_y = np.polyval(p, self._start_x)
                self._end_y = np.polyval(p, self._end_x)
                
                # Punkte aktualisieren
                self.start_point.setPos(self._start_x, self._start_y)
                self.end_point.setPos(self._end_x, self._end_y)
        
        self._update_mode()
    
    def _update_mode(self):
        """Aktualisiert die Sichtbarkeit der Elemente je nach Modus."""
        is_manual = (self._mode == "manual")
        
        # Linien (Regression-Modus)
        self.start_line.setVisible(not is_manual)
        self.end_line.setVisible(not is_manual)
        
        # Punkte (Manuell-Modus)
        self.start_point.setVisible(is_manual)
        self.end_point.setVisible(is_manual)
        
        # Rauschboden IMMER ziehbar (in beiden Modi)
        self.noise_line.setMovable(True)
        # Ausschluss nur in Manuell ziehbar
        self.exclude_line.setMovable(is_manual)
        
        # Farben anpassen - Rauschboden immer blau (aktiv)
        self.noise_line.setPen(pg.mkPen('#89b4fa', width=2, style=Qt.DashLine))
        self.noise_label.setStyleSheet("font-size: 11px; color: #89b4fa;")
        
        if is_manual:
            self.info_label.setText(
                "Manuell: Verschieben Sie Start-/Endpunkt (X,Y), Rauschboden und Ausschluss-Grenze."
            )
        else:
            self.info_label.setText(
                "Regression: Wählen Sie Start-/Endzeit und Rauschboden. Die Regression wird automatisch berechnet."
            )
        
        # Ausschluss-Label nur in Manuell sichtbar
        self.exclude_label.setVisible(is_manual)
        
        # Update
        self._update_display()
    
    def _on_exclude_changed(self):
        """Ausschluss-Grenze wurde verschoben (nur Manuell-Modus)."""
        new_pos = max(0.01, min(self.exclude_line.value(), self._start_x - 0.01))
        self._t_exclude = new_pos
        self.exclude_region.setRegion([0, self._t_exclude])
        self.exclude_label.setText(f"Ausschluss: {self._t_exclude:.2f} s")
    
    def _on_start_line_changed(self):
        """Start-Linie wurde verschoben (Regression-Modus)."""
        new_pos = self.start_line.value()
        new_pos = max(self._t_exclude + 0.01, min(new_pos, self._end_x - 0.01))
        self._start_x = new_pos
        self._update_display()
    
    def _on_end_line_changed(self):
        """End-Linie wurde verschoben (Regression-Modus)."""
        new_pos = self.end_line.value()
        new_pos = max(self._start_x + 0.01, min(new_pos, self._decay.t[-1] * 0.95))
        self._end_x = new_pos
        self._update_display()
    
    def _on_start_point_changed(self):
        """Startpunkt wurde verschoben (Manuell-Modus)."""
        pos = self.start_point.pos()
        self._start_x = max(self._t_exclude + 0.01, pos.x())
        self._start_y = pos.y()
        self._update_display()
    
    def _on_end_point_changed(self):
        """Endpunkt wurde verschoben (Manuell-Modus)."""
        pos = self.end_point.pos()
        self._end_x = max(self._start_x + 0.01, pos.x())
        self._end_y = pos.y()
        self._update_display()
    
    def _on_noise_changed(self):
        """Rauschboden wurde verschoben (Manuell-Modus)."""
        self._noise_level = self.noise_line.value()
        self.noise_label.setText(f"Rauschboden: {self._noise_level:.1f} dB")
        self._update_display()
    
    def _update_display(self):
        """Aktualisiert Gerade, RT60 und Schnittpunkt."""
        t = self._decay.t
        l_ir = self._decay.l_ir
        
        if self._mode == "regression":
            # Regression im gewählten Bereich
            mask = (t >= self._start_x) & (t <= self._end_x)
            if not np.any(mask):
                return
            
            p = np.polyfit(t[mask], l_ir[mask], 1)
            slope = p[0]
            
            # Start-/End-Y für Anzeige
            self._start_y = np.polyval(p, self._start_x)
            self._end_y = np.polyval(p, self._end_x)
            
            # Gerade zeichnen - NUR zwischen Start und Ende
            t_plot = np.linspace(self._start_x, self._end_x, 100)
            l_plot = np.polyval(p, t_plot)
            self.trend_line.setData(t_plot, l_plot)
            
            # Schnittpunkt mit Rauschboden
            if abs(slope) > 1e-6:
                self._t_intersect = (self._noise_level - p[1]) / slope
            else:
                self._t_intersect = t[-1]
            
            # Labels
            self.start_label.setText(f"Start: {self._start_x:.2f} s")
            self.end_label.setText(f"Ende: {self._end_x:.2f} s")
            
        else:  # manual
            # Gerade durch zwei Punkte
            dx = self._end_x - self._start_x
            dy = self._end_y - self._start_y
            
            if abs(dx) < 1e-6:
                return
            
            slope = dy / dx
            
            # Schnittpunkt mit Rauschboden
            if abs(slope) > 1e-6:
                self._t_intersect = (self._noise_level - self._start_y) / slope + self._start_x
            else:
                self._t_intersect = t[-1]
            
            # Gerade zeichnen - NUR zwischen Start und Ende
            t_plot = np.linspace(self._start_x, self._end_x, 100)
            l_plot = slope * (t_plot - self._start_x) + self._start_y
            self.trend_line.setData(t_plot, l_plot)
            
            # Labels
            self.start_label.setText(f"Start: ({self._start_x:.2f}s, {self._start_y:.1f}dB)")
            self.end_label.setText(f"Ende: ({self._end_x:.2f}s, {self._end_y:.1f}dB)")
        
        # RT60 berechnen
        if self._mode == "regression":
            if abs(p[0]) > 1e-3:
                self._rt60 = 60.0 / abs(p[0])
            else:
                self._rt60 = 0.0
        else:
            dx = self._end_x - self._start_x
            dy = self._end_y - self._start_y
            if abs(dx) > 1e-6 and abs(dy/dx) > 1e-3:
                self._rt60 = 60.0 / abs(dy / dx)
            else:
                self._rt60 = 0.0
        
        # Schnittpunkt-Marker aktualisieren
        if 0 < self._t_intersect < t[-1]:
            self.intersect_point.setData(pos=[(self._t_intersect, self._noise_level)])
            self.intersect_point.show()
        else:
            self.intersect_point.hide()
        
        # Labels updaten
        self.rt60_label.setText(f"RT60: {self._rt60:.2f} s")
        self.intersect_label.setText(f"Schnittpunkt: {self._t_intersect:.2f} s")
    
    def _accept(self):
        """Übernehmen geklickt."""
        self._accepted = True
        self.accept()
    
    def get_corrected_values(self) -> dict:
        """Gibt die korrigierten Werte zurück."""
        return {
            'mode': self._mode,
            'start_x': self._start_x,
            'start_y': self._start_y,
            'end_x': self._end_x,
            'end_y': self._end_y,
            'noise_level': self._noise_level,
            't_intersect': self._t_intersect,
            't_exclude': self._t_exclude,
            'rt60': self._rt60,
        }
    
    def get_corrected_times(self) -> tuple:
        """Gibt die korrigierten Start- und Endzeiten zurück (für Analyse)."""
        # Start = Ausschluss-Grenze (t_exclude), Ende = Schnittpunkt
        return (self._t_exclude, self._t_intersect)
    
    def was_accepted(self) -> bool:
        """True wenn der Benutzer 'Übernehmen' geklickt hat."""
        return self._accepted


class RoomSchematic(QWidget):
    """Zeichnet eine proportionale isometrische Raumskizze zur Visualisierung der Wandpaare."""
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(180, 180)
        self.setCursor(Qt.PointingHandCursor)
        self.active_dims = set() # Set von "L", "B", "H"
        self.dims = (5.0, 4.0, 3.0) # Standardmaße für Vorschau
        
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
        # Wenn Maße 0 sind, nutzen wir einen Standard-Würfel für die Vorschau
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
        
        # Maximale Skalierung berechnen, sodass der Raum "Edge-to-Edge" passt
        # Projizierte Breite: (bs + ls) * cos30
        # Projizierte Höhe: hs + (bs + ls) * sin30
        # Wir wollen: scale * (b_m + l_m) * cos30 <= w * 0.9
        # und: scale * (h_m + (b_m + l_m) * sin30) <= h_widget * 0.9
        
        denom_w = (b_m + l_m) * cos30
        denom_h = h_m + (b_m + l_m) * sin30
        
        scale_w = (w * 0.9) / denom_w if denom_w > 0 else 100
        scale_h = (h_widget * 0.9) / denom_h if denom_h > 0 else 100
        
        scale = min(scale_w, scale_h)
        
        ls = l_m * scale
        bs = b_m * scale
        hs = h_m * scale
        
        # Zentrumsberechnung für perfekte Zentrierung des projizierten Quaders
        # Horizontaler Versatz durch ungleiche Längen/Breiten ausgleichen
        cx = w / 2 - (bs - ls) * cos30 / 2
        # Vertikaler Versatz (projizierte Box geht von y=-hs bis y=+(bs+ls)*sin30 relativ zum Ursprung)
        cy = h_widget / 2 - ((bs + ls) * sin30 - hs) / 2
        
        # 3D Punkte -> 2D Projektion
        # Wir definieren Punkte (x=Breite, y=Länge, z=Höhe)
        def project(x3d, y3d, z3d):
            x2d = cx + (x3d * cos30) - (y3d * cos30)
            y2d = cy + (x3d * sin30) + (y3d * sin30) - z3d
            return QPointF(x2d, y2d)

        # Eckpunkte des Quaders
        p0 = project(0, 0, 0)    # unten-mitte-vorn
        p1 = project(bs, 0, 0)   # unten-rechts
        p2 = project(bs, ls, 0)  # unten-hinten
        p3 = project(0, ls, 0)   # unten-links
        
        p4 = project(0, 0, hs)   # oben-mitte-vorn
        p5 = project(bs, 0, hs)  # oben-rechts
        p6 = project(bs, ls, hs) # oben-hinten
        p7 = project(0, ls, hs)  # oben-links

        def draw_face(pts, color, highlight=False):
            poly = QPolygonF(pts)
            if highlight:
                # Kräftiges Highlight
                painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 200)))
                painter.setPen(QPen(QColor("#ffffff"), 2))
            else:
                # Dezente Basis-Optik
                painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 40)))
                painter.setPen(QPen(QColor("#585b70"), 1))
            painter.drawPolygon(poly)

        # Grundfarben
        col_l = QColor("#a6e3a1") # Grün (Länge)
        col_b = QColor("#89b4fa") # Blau (Breite)
        col_h = QColor("#f9e2af") # Gelb (Höhe)
        
        # Zeichne Flächen (von hinten nach vorn für richtige Überdeckung)
        # 1. Boden (H-Partner unten)
        draw_face([p0, p1, p2, p3], col_h, "H" in self.active_dims)
        
        # 2. Rückwände (L/B Partner hinten)
        draw_face([p1, p2, p6, p5], col_b, "B" in self.active_dims) # Hinten-Rechts (gehört zu B)
        draw_face([p3, p2, p6, p7], col_l, "L" in self.active_dims) # Hinten-Links (gehört zu L)
        
        # 3. Vorderwände
        draw_face([p0, p3, p7, p4], col_b, "B" in self.active_dims) # Vorne-Links
        draw_face([p0, p1, p5, p4], col_l, "L" in self.active_dims) # Vorne-Rechts
        
        # 4. Decke
        draw_face([p4, p5, p6, p7], col_h, "H" in self.active_dims)


class ShiftZoomPlotWidget(pg.PlotWidget):
    """Custom PlotWidget mit Shift-Taste für Y-Achsen-Kontrolle."""
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
    """Widget für Room-Analysis mit interaktiver Peak-Auswahl."""
    
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
        title_label = QLabel("Distogramm")
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
        self.plot.setLabel('bottom', 'Abstand Reflexionsflächen', units='m')
        self.plot.getPlotItem().setMenuEnabled(False)
        
        self.curve = self.plot.plot(pen=pg.mkPen('#89b4fa', width=1.5))
        
        # Referenzlinien für L, B, H
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
        
        # --- UNTERER BEREICH (KOMPAKT) ---
        results_container = QWidget()
        results_layout = QHBoxLayout(results_container)
        results_layout.setContentsMargins(30, 8, 10, 8)  # Mehr Margin links für Abstand zum Rand
        results_layout.setSpacing(25)  # Etwas mehr Spacing zwischen den Spalten
        
        # Linke Spalte: Daten & Eingaben
        data_column = QVBoxLayout()
        data_column.setSpacing(10)
        
        # Tabelle für Ergebnisse (jetzt 4 Zeilen, RT60 entfernt)
        self.results_table = QTableWidget(4, 2)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.horizontalHeader().setVisible(False)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4; gridline-color: #313244; border: 1px solid #313244;")
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setFocusPolicy(Qt.NoFocus)
        self.results_table.setFixedHeight(145) # Höher, damit kein Scrollen nötig ist
        
        headers = ["Abstand", "Repetitionsfreq.", "Amplitude", "Hörbarkeit"]
        for i, h in enumerate(headers):
            self.results_table.setItem(i, 0, QTableWidgetItem(h))
            self.results_table.setItem(i, 1, QTableWidgetItem("–"))
            self.results_table.item(i, 1).setTextAlignment(Qt.AlignCenter)
        
        data_column.addWidget(self.results_table)
        
        data_column.addSpacing(15) # Mehr Abstand zwischen Tabelle und Raumdimensionen
        
        # Raumdimensionen
        room_group = QGroupBox("Raumdimensionen (L x B x H)")
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
        
        # RT60 Plot-Button (unten in der Daten-Spalte entfernt, wird nach rechts verschoben)
        # (Ehemals hier: data_column.addLayout(rt60_layout))
        
        data_column.addStretch() # Drückt alles nach oben
        
        results_layout.addLayout(data_column, 2) # 20-30% Breite
        
        # Rechte Spalte: Raum-Schematik & RT60 (Groß)
        schematic_column = QVBoxLayout()
        schematic_column.setSpacing(0)
        
        self.room_schematic = RoomSchematic()
        self.room_schematic.clicked.connect(self._show_analysis_pipeline)
        schematic_column.addWidget(self.room_schematic, stretch=1)
        
        # RT60 Anzeige & Plot Button ganz unten rechts
        rt60_footer = QHBoxLayout()
        rt60_footer.addStretch()
        
        self.rt60_label = QLabel("RT60: –")
        self.rt60_label.setStyleSheet("color: #cdd6f4; font-size: 11px; padding: 2px 5px;")
        rt60_footer.addWidget(self.rt60_label)
        
        self.rt60_plot_btn = QPushButton("Plot")
        self.rt60_plot_btn.setToolTip("Pegelabfall und Regression anzeigen")
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
        
        results_layout.addLayout(schematic_column, 5) # 70-80% Breite
        
        splitter.addWidget(results_container)
        splitter.setSizes([300, 350]) # Unterer Bereich etwas größer (nach oben gezogen)
        layout.addWidget(splitter)

    def keyPressEvent(self, event: QKeyEvent):
        if self._result and self._result.peaks:
            if event.key() == Qt.Key_Left:
                # Vorheriger Peak (kleinerer Index / kleinere Distanz)
                self._selected_peak_index = (self._selected_peak_index - 1) % len(self._result.peaks)
                self._update_peak_display()
                event.accept()
            elif event.key() == Qt.Key_Right:
                # Nächster Peak (größerer Index / größere Distanz)
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
        Führt die Room-Analysis aus:
        1. Berechnet Pegelverlauf
        2. Zeigt RT60-Korrektur-Dialog
        3. Bei Bestätigung: Führt Peak-Detektion mit korrigierten Werten aus
        """
        if self._audio_data is None: 
            return
        
        # Schritt 1: Pegelverlauf berechnen
        config = FlutterDetectionConfig(sample_rate=self._sample_rate)
        decay_result = compute_decay_curve(self._audio_data, config)
        
        # Schritt 2: RT60-Korrektur-Dialog anzeigen
        dialog = RT60CorrectionDialog(decay_result, self)
        dialog.exec()
        
        if not dialog.was_accepted():
            # Benutzer hat abgebrochen
            return
        
        # Schritt 3: Korrigierte Werte übernehmen und Analyse durchführen
        values = dialog.get_corrected_values()
        
        config_corrected = FlutterDetectionConfig(
            sample_rate=self._sample_rate,
            t_start_fit_override=values['t_exclude'],
            t_end_fit_override=values['t_intersect'],
            noise_level_override=values['noise_level'],
        )
        
        # Bei manuellem Modus auch Steigung und Achsenabschnitt übergeben
        if values['mode'] == 'manual':
            dx = values['end_x'] - values['start_x']
            dy = values['end_y'] - values['start_y']
            if abs(dx) > 1e-6:
                config_corrected.manual_slope = dy / dx
                config_corrected.manual_intercept = values['start_y'] - (dy / dx) * values['start_x']
        
        self._result = analyze_flutter(self._audio_data, config_corrected)
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
        self.results_table.item(3, 1).setText(self._result.severity if peak.is_main else self._estimate_peak_severity(peak.amplitude))
        
        # RT60 Anzeige updaten
        if self._result.rt60_s > 0:
            self.rt60_label.setText(f"RT60: {self._result.rt60_s:.2f} s")
            self.rt60_plot_btn.setEnabled(True)
        else:
            self.rt60_label.setText("RT60: –")
            self.rt60_plot_btn.setEnabled(False)
        
        self._update_wall_assignment()

    @Slot()
    def _show_rt60_details(self):
        """Öffnet den Detail-Dialog für RT60."""
        if self._result:
            dialog = RT60DetailDialog(self._result, self)
            dialog.exec()

    @Slot()
    def _show_analysis_pipeline(self):
        """Öffnet den Detail-Dialog für die gesamte Analyse-Pipeline."""
        if self._result:
            dialog = AnalysisPipelineDialog(self._result, self)
            dialog.exec()

    def _on_room_dimensions_changed(self):
        l = self._get_input_value(self.length_input)
        b = self._get_input_value(self.width_input)
        h = self._get_input_value(self.height_input)
        
        # Proportionales Modell updaten
        self.room_schematic.set_dimensions(l, b, h)
        
        # Referenzlinien
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
        
    def _estimate_peak_severity(self, amp):
        if amp > 800: return "stark"
        if amp > 400: return "mittel"
        return "schwach"

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
        for i in range(4): self.results_table.item(i, 1).setText("–")
