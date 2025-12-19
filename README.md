# Audio Analyzer

## Analysetool von Audiodateien mit Fokus auf zeitabhängige Terzband-Impulsantworten zur besseren Diagnose von Flatterechos.

Das tool dient der bestimmung von Flatterechos aus Audioaufzeichnungen. Für die eingelesene Audiodatei können trezband-Impulsantworten (Audio/Zeitverlauf) erstellt werden. Mithilfe deiser kann einfach die Terzbänder des Flatterechos bestimmt werden. Die Terzbandimpulsantowrten können als wav exportiert werden.

<img width="994" height="736" alt="Image" src="https://github.com/user-attachments/assets/2335b1ac-ab21-4697-87bb-a3e05fa803fb" />

<img width="999" height="738" alt="Image" src="https://github.com/user-attachments/assets/94a48b2e-38ba-4bc7-b449-a2bd680be28e" />

<img width="996" height="736" alt="Image" src="https://github.com/user-attachments/assets/82eebb82-a4db-4030-ab79-07c1a3e71931" />

## Installation

### Entwicklungsumgebung

```bash
# Python 3.11+ erforderlich
python -m venv venv
source venv/bin/activate  # Linux/macOS
# oder: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Windows-Executable erstellen

```bash
pyinstaller audio_analyzer.spec
# Ergebnis in dist/AudioAnalyzer.exe
```

## Technische Dokumentation

### Terzbandfilterbank (IEC 61260)

Die Filterbank verwendet **IIR Butterworth-Filter 6. Ordnung** mit:
- Normgerechten Mittenfrequenzen nach IEC 61260-1
- Bandbreite: fm × (2^(1/6) - 2^(-1/6))
- Dokumentiertes Phasenverhalten und Gruppenlaufzeit


## Abhängigkeiten

- Python 3.11+
- PyQt6 (GUI)
- numpy, scipy (Signalverarbeitung)
- pyqtgraph (interaktive Visualisierung)
- soundfile (WAV-I/O)
- librosa/audioread (MP3-Dekodierung)
- sounddevice (Wiedergabe)

## Lizenz

MIT License
