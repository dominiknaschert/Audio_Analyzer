"""
Room-Analysis - Core Strukturen, Wrapper und Algorithmus.
Basierend auf Schanda/Hoffbauer/Lachenmayr (DAGA 2023) und der Zwei-Geraden-Regression.
Vgl. Paper in `Literatur/000213.pdf`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal

import numpy as np
from scipy import signal


__all__ = [
    "PeakInfo",
    "FlutterEchoResult",
    "FlutterDetectionConfig",
    "DecayCurveResult",
    "analyze_flutter",
    "compute_decay_curve",
    "FlutterEchoDetector",
    "SOUND_SPEED",
    "distance_to_frequency",
    "frequency_to_distance",
    "distance_to_samples",
    "samples_to_distance",
    "schroeder_integration",
    "_butter_bandpass",
    "_butter_lowpass",
]
SOUND_SPEED = 343.0  # m/s bei 20°C


# ============================================================
# DATENSTRUKTUREN
# ============================================================

@dataclass
class PeakInfo:
    """
    Informationen zu einem einzelnen Peak im Distogramm.
    """
    distance_m: float              # Abstand in Metern
    amplitude: float               # Amplitude des Peaks (w.E.)
    repetition_frequency_hz: float # Repetitionsfrequenz f_rep
    is_main: bool = False          # True für Hauptpeak


@dataclass
class FlutterEchoResult:
    """
    Ergebnis der Flatterecho-Detektion für die GUI.
    """
    distances: np.ndarray          # X-Achse: Abstände in Metern
    amplitudes: np.ndarray         # Y-Achse: Amplituden (w.E.)
    peaks: List[PeakInfo]          # Alle erkannten Peaks
    
    main_distance_m: float         # Hauptabstand
    distance_uncertainty_m: float  # Genauigkeit
    repetition_frequency_hz: float # f_rep
    relative_amplitude: float      # Peak-Amplitude
    flutter_tonality_hz: float     # Mittenfrequenz der Analyse (z.B. 1000 Hz)
    severity: str                  # Hörbarkeit ("nicht hörbar", etc.)
    detected: bool                 # Erfolg der Detektion
    rt60_s: float = 0.0            # Nachhallzeit RT60 (beruhend auf Trendlinie)
    
    # Debug/Plot Daten für Pipeline Details
    t: Optional[np.ndarray] = None
    l_ir: Optional[np.ndarray] = None
    l_trend: Optional[np.ndarray] = None
    rir_raw: Optional[np.ndarray] = None
    rir_bp: Optional[np.ndarray] = None
    l_fe: Optional[np.ndarray] = None
    acf: Optional[np.ndarray] = None
    t_intersect: float = 0.0
    noise_level: float = 0.0
    p_decay: Optional[np.ndarray] = None
    
    # Rückwärtskompatibilität
    peak_distances: List[float] = None
    peak_amplitudes: List[float] = None


@dataclass
class FlutterDetectionConfig:
    """Konfiguration für den Detektionsalgorithmus."""
    sample_rate: int = 48_000
    band_center_hz: float = 1_000.0  # Oktavband wie im Paper
    band_q: float = 1 / np.log(2)    # entspricht ~1 Oktave (BW = fc / Q)
    lp_cut_hz: float = 200.0         # Tiefpass für Hüllkurve
    t_start_fit: float = 0.2         # Ignoriere erste 200ms (Direktschall)
    method: Literal["regression", "schroeder"] = "regression"
    min_distance_m: float = 0.5
    max_distance_m: float = 20.0
    peak_rel_height: float = 0.1     # Peaks: relative Schwelle zu max
    peak_prominence: float = 0.05    # relative Prominenz
    # Manuelle Override-Werte für RT60-Korrektur
    t_start_fit_override: Optional[float] = None  # Manueller Startpunkt für Regression
    t_end_fit_override: Optional[float] = None    # Manueller Endpunkt für Regression
    noise_level_override: Optional[float] = None  # Manueller Rauschboden (dB)
    manual_slope: Optional[float] = None          # Manuelle Steigung (dB/s)
    manual_intercept: Optional[float] = None      # Manueller Y-Achsenabschnitt (dB)


@dataclass
class DecayCurveResult:
    """Ergebnis der Pegelverlauf-Berechnung (vor Peak-Detektion)."""
    t: np.ndarray               # Zeitachse in Sekunden
    l_ir: np.ndarray            # Pegelverlauf in dB
    l_trend: np.ndarray         # Trendlinie in dB
    rt60_s: float               # Berechnete RT60
    t_start_fit: float          # Start der Regression
    t_end_fit: float            # Ende der Regression
    t_intersect: float          # Schnittpunkt mit Rauschboden
    noise_level: float          # Rauschboden in dB
    p_decay: Optional[np.ndarray]  # Polynomkoeffizienten der Regression
    rir_bp: np.ndarray          # Bandpass-gefiltertes Signal
    sample_rate: int            # Sample-Rate


# ============================================================
# KERN-ALGORITHMUS (SIGNALVERARBEITUNG)
# ============================================================

def _butter_bandpass(fc: float, q: float, fs: int, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    bw = fc / q
    low = max(1e-6, (fc - bw / 2) / (fs / 2))
    high = min(0.999, (fc + bw / 2) / (fs / 2))
    return signal.butter(order, [low, high], btype="bandpass")


def _butter_lowpass(fc: float, fs: int, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    wn = min(0.999, fc / (fs / 2))
    return signal.butter(order, wn, btype="lowpass")


def schroeder_integration(env: np.ndarray) -> np.ndarray:
    """Rückwärtsintegration nach Schroeder auf der Energiezeitreihe."""
    rev_cum = np.cumsum(env[::-1])
    sch = rev_cum[::-1]
    if np.max(sch) > 0:
        sch = sch / np.max(sch) * np.max(env)
    return sch


def compute_decay_curve(
    impulse_response: np.ndarray,
    config: Optional[FlutterDetectionConfig] = None,
) -> DecayCurveResult:
    """
    Berechnet nur den Pegelverlauf und die initiale Trendlinie.
    Wird vor dem RT60-Korrektur-Dialog aufgerufen.
    """
    cfg = config or FlutterDetectionConfig()
    fs = cfg.sample_rate
    rir = np.asarray(impulse_response, dtype=float)
    t = np.arange(len(rir)) / fs

    # 1) Bandpass
    b_bp, a_bp = _butter_bandpass(cfg.band_center_hz, cfg.band_q, fs, order=4)
    rir_bp = signal.filtfilt(b_bp, a_bp, rir)

    # 2) Pegel-Zeit-Darstellung (Hüllkurve in dB)
    window_len = int(0.002 * fs)
    energy = rir_bp**2
    smoothed_energy = np.convolve(energy, np.ones(window_len)/window_len, mode='same')
    l_ir = 10 * np.log10(smoothed_energy + 1e-12)

    # 3) Initiale Regression
    t_start = cfg.t_start_fit_override if cfg.t_start_fit_override is not None else cfg.t_start_fit
    t_end_auto = min(t[-1] * 0.8, 1.2)
    t_end = cfg.t_end_fit_override if cfg.t_end_fit_override is not None else t_end_auto
    
    mask_decay = (t >= t_start) & (t <= t_end)
    
    rt60 = 0.0
    t_intersect = t[-1]
    noise_level = -100.0
    p_decay = None
    l_trend = np.full_like(l_ir, np.mean(l_ir))
    
    if np.any(mask_decay):
        p_decay = np.polyfit(t[mask_decay], l_ir[mask_decay], 1)
        
        # RT60 berechnen: Abfall um 60 dB (extrapoliert)
        if abs(p_decay[0]) > 1e-3:
            rt60 = 60.0 / abs(p_decay[0])
        
        # Rauschboden (letzte 10%)
        t_noise_start = t[-1] * 0.9
        noise_level = np.mean(l_ir[t >= t_noise_start])
        
        # Schnittpunkt
        if abs(p_decay[0]) > 1e-6:
            t_intersect = (noise_level - p_decay[1]) / p_decay[0]
        
        # Trendlinie
        l_trend = np.zeros_like(l_ir)
        l_trend[t < t_start] = np.polyval(p_decay, t_start)
        mask_before_intersect = (t >= t_start) & (t <= t_intersect)
        l_trend[mask_before_intersect] = np.polyval(p_decay, t[mask_before_intersect])
        l_trend[t > t_intersect] = noise_level

    return DecayCurveResult(
        t=t,
        l_ir=l_ir,
        l_trend=l_trend,
        rt60_s=rt60,
        t_start_fit=t_start,
        t_end_fit=t_end,
        t_intersect=t_intersect,
        noise_level=noise_level,
        p_decay=p_decay,
        rir_bp=rir_bp,
        sample_rate=fs,
    )


def analyze_flutter(
    impulse_response: np.ndarray,
    config: Optional[FlutterDetectionConfig] = None,
) -> FlutterEchoResult:
    """
    Führt die Flatterecho-Detektion aus (Pipeline nach DAGA 2023).
    """
    cfg = config or FlutterDetectionConfig()
    fs = cfg.sample_rate
    rir = np.asarray(impulse_response, dtype=float)
    t = np.arange(len(rir)) / fs

    # 1) Bandpass
    b_bp, a_bp = _butter_bandpass(cfg.band_center_hz, cfg.band_q, fs, order=4)
    rir_bp = signal.filtfilt(b_bp, a_bp, rir)

    # 2) Pegel-Zeit-Darstellung (Hüllkurve in dB)
    window_len = int(0.002 * fs)
    energy = rir_bp**2
    smoothed_energy = np.convolve(energy, np.ones(window_len)/window_len, mode='same')
    l_ir = 10 * np.log10(smoothed_energy + 1e-12)

    # 3) Trend-Korrektur (Nachhall isolation)
    rt60 = 0.0
    t_intersect = 0.0
    noise_level = -100.0
    p_decay = None
    if cfg.method == "regression":
        # Regression: Verwende Override-Werte falls gesetzt
        t_start = cfg.t_start_fit_override if cfg.t_start_fit_override is not None else cfg.t_start_fit
        t_end_auto = min(t[-1] * 0.8, 1.2)
        t_end_fit = cfg.t_end_fit_override if cfg.t_end_fit_override is not None else t_end_auto
        
        # Prüfe ob manuelle Steigung gesetzt ist
        if cfg.manual_slope is not None and cfg.manual_intercept is not None:
            # Manuelle Werte verwenden
            p_decay = np.array([cfg.manual_slope, cfg.manual_intercept])
            
            # RT60 aus manueller Steigung
            if abs(cfg.manual_slope) > 1e-3:
                rt60 = 60.0 / abs(cfg.manual_slope)
            
            # Rauschboden: Override oder automatisch
            if cfg.noise_level_override is not None:
                noise_level = cfg.noise_level_override
            else:
                t_noise_start = t[-1] * 0.9
                noise_level = np.mean(l_ir[t >= t_noise_start])
            
            # Schnittpunkt
            if abs(p_decay[0]) > 1e-6:
                t_intersect = (noise_level - p_decay[1]) / p_decay[0]
            else:
                t_intersect = t[-1]
            
            # Trendlinie
            l_trend = np.zeros_like(l_ir)
            l_trend[t < t_start] = np.polyval(p_decay, t_start)
            mask_before_intersect = (t >= t_start) & (t <= t_intersect)
            l_trend[mask_before_intersect] = np.polyval(p_decay, t[mask_before_intersect])
            l_trend[t > t_intersect] = noise_level
            
        else:
            # Automatische Regression
            mask_decay = (t >= t_start) & (t <= t_end_fit)
            
            if np.any(mask_decay):
                p_decay = np.polyfit(t[mask_decay], l_ir[mask_decay], 1)
                
                # RT60 berechnen: Abfall um 60 dB (extrapoliert)
                if abs(p_decay[0]) > 1e-3:
                    rt60 = 60.0 / abs(p_decay[0])
                
                # Rauschboden: Override oder automatisch (letzte 10%)
                if cfg.noise_level_override is not None:
                    noise_level = cfg.noise_level_override
                else:
                    t_noise_start = t[-1] * 0.9
                    noise_level = np.mean(l_ir[t >= t_noise_start])
                
                # Schnittpunkt
                if abs(p_decay[0]) > 1e-6:
                    t_intersect = (noise_level - p_decay[1]) / p_decay[0]
                else:
                    t_intersect = t[-1]
                
                # Trendlinie L_dif
                l_trend = np.zeros_like(l_ir)
                l_trend[t < t_start] = np.polyval(p_decay, t_start)
                mask_before_intersect = (t >= t_start) & (t <= t_intersect)
                l_trend[mask_before_intersect] = np.polyval(p_decay, t[mask_before_intersect])
                l_trend[t > t_intersect] = noise_level
            else:
                l_trend = np.full_like(l_ir, np.mean(l_ir))
                t_intersect = t[-1]

        l_fe_db = l_ir - l_trend
        mask_analysis = (t >= t_start) & (t <= t_intersect)
        analysis_signal = l_fe_db[mask_analysis]
        
    else: # Schroeder Methode
        env_sch = schroeder_integration(energy)
        l_trend = 10 * np.log10(env_sch + 1e-12)
        l_fe_db = l_ir - l_trend
        mask_analysis = (t >= cfg.t_start_fit)
        analysis_signal = l_fe_db[mask_analysis]
        t_intersect = t[-1]
        # RT60 aus Schroeder-Kurve (vereinfacht)
        # Hier verzichten wir auf eine komplexe RT60 Schätzung für Schroeder
        rt60 = 0.0

    # 4) Autokorrelation (ACF)
    if len(analysis_signal) < 100:
        return FlutterEchoResult(distances=np.array([]), amplitudes=np.array([]), peaks=[], 
                                main_distance_m=0, distance_uncertainty_m=0, 
                                repetition_frequency_hz=0, relative_amplitude=0, 
                                flutter_tonality_hz=cfg.band_center_hz, severity="n/a", detected=False,
                                rt60_s=rt60, rir_raw=rir, rir_bp=rir_bp, l_fe=l_fe_db, t=t, l_ir=l_ir, l_trend=l_trend)

    analysis_signal -= np.mean(analysis_signal)
    acf_full = signal.correlate(analysis_signal, analysis_signal, mode="full")
    acf = acf_full[acf_full.size // 2 :]
    acf /= (acf[0] + 1e-12)

    # 5) FFT -> Repetitionsspektrum
    spec = np.abs(np.fft.rfft(acf))
    freqs = np.fft.rfftfreq(len(acf), d=1 / fs)

    # 6) Frequenzen -> Distanzen
    valid_mask = (freqs > 0) & (freqs < fs / 2)
    freqs_valid = freqs[valid_mask]
    distances_all = SOUND_SPEED / (2.0 * freqs_valid + 1e-10)
    amplitudes_all = spec[valid_mask]

    # Bereich filtern
    dist_mask = (distances_all >= cfg.min_distance_m) & (distances_all <= cfg.max_distance_m)
    distances = distances_all[dist_mask]
    amplitudes = amplitudes_all[dist_mask]

    # Peak-Erkennung
    peaks = []
    if len(amplitudes) > 0:
        height_thr = cfg.peak_rel_height * np.max(amplitudes)
        prominence_thr = cfg.peak_prominence * np.max(amplitudes)
        peak_idx, _ = signal.find_peaks(amplitudes, height=height_thr, prominence=prominence_thr)
        
        for i, idx in enumerate(peak_idx):
            dist = distances[idx]
            amp = amplitudes[idx]
            peaks.append(PeakInfo(
                distance_m=dist,
                amplitude=amp,
                repetition_frequency_hz=distance_to_frequency(dist),
                is_main=False # wird nach Sortierung gesetzt
            ))

        # Peaks nach Distanz sortieren (aufsteigend)
        peaks.sort(key=lambda p: p.distance_m)
        
        # Hauptpeak markieren (der mit der höchsten Amplitude)
        if peaks:
            max_amp_peak = max(peaks, key=lambda p: p.amplitude)
            max_amp_peak.is_main = True

    main_peak = max(peaks, key=lambda p: p.amplitude) if peaks else None
    main_distance = main_peak.distance_m if main_peak else 0.0
    main_amp = main_peak.amplitude if main_peak else 0.0

    return FlutterEchoResult(
        distances=distances,
        amplitudes=amplitudes,
        peaks=peaks,
        main_distance_m=main_distance,
        distance_uncertainty_m=0.1,
        repetition_frequency_hz=distance_to_frequency(main_distance),
        relative_amplitude=main_amp,
        flutter_tonality_hz=cfg.band_center_hz,
        severity="hörbar" if len(peaks) > 0 else "nicht hörbar",
        detected=len(peaks) > 0,
        rt60_s=rt60,
        t=t,
        l_ir=l_ir,
        l_trend=l_trend,
        rir_raw=rir,
        rir_bp=rir_bp,
        l_fe=l_fe_db,
        acf=acf,
        t_intersect=t_intersect,
        noise_level=noise_level,
        p_decay=p_decay,
        peak_distances=[p.distance_m for p in peaks],
        peak_amplitudes=[p.amplitude for p in peaks],
    )


# ============================================================
# WRAPPER KLASSE FÜR GUI
# ============================================================

class FlutterEchoDetector:
    """
    Wrapper für den Flatterecho-Detektionsalgorithmus.
    """
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
    
    def analyze(self, impulse_response: np.ndarray) -> FlutterEchoResult:
        """
        Analysiere Raumimpulsantwort auf Flatterechos.
        """
        cfg = FlutterDetectionConfig(sample_rate=self.sample_rate)
        return analyze_flutter(impulse_response, cfg)


# ============================================================
# HILFS-FUNKTIONEN (UMRECHNUNG)
# ============================================================

def distance_to_samples(distance_m: float, sample_rate: int) -> int:
    """Konvertiere Abstand (m) zu Samples (Hin- und Rückweg)."""
    time_s = (2 * distance_m) / SOUND_SPEED
    return int(time_s * sample_rate)


def samples_to_distance(samples: int, sample_rate: int) -> float:
    """Konvertiere Samples zu Abstand (m)."""
    time_s = samples / sample_rate
    return (time_s * SOUND_SPEED) / 2


def frequency_to_distance(freq_hz: float) -> float:
    """Konvertiere Repetitionsfrequenz zu Abstand."""
    if freq_hz <= 0:
        return 0
    return SOUND_SPEED / (2 * freq_hz)


def distance_to_frequency(distance_m: float) -> float:
    """Konvertiere Abstand zu Repetitionsfrequenz."""
    if distance_m <= 0:
        return 0
    return SOUND_SPEED / (2 * distance_m)
