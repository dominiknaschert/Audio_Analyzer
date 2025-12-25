## Description

**Echolab** is an analysis tool for the detection of flutter echoes. It uses room impulse responses to visualize flutter echoes in the time domain. In addition to the visual time-domain analysis, the tool implements the detection algorithm presented by Schanda et al. (DAGA 2023) [1]. The algorithm identifies repetition frequencies using autocorrelation and calculates the corresponding distances.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation 

1. clone repository:
```bash
git clone https://github.com/username/echolab.git
cd echolab
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Mac/Windows Executable

```bash
pyinstaller echolab.spec
```
executable will be created in `dist/`.

## Core Functionality

#### Implementation of the flutter echo detection algorithm (Schanda et al., DAGA 2023)

The detection method aims to identify flutter echoes in room impulse responses and determine their propagation distances. The implemented algorithm [1] applies bandpass filtering at 1000 Hz center frequency with one-octave bandwidth (Q ≈ 1.44, bandwidth ≈ 693 Hz), converts the signal to a level-time representation, windows the time domain, subtracts the reverberation decay trend, computes the autocorrelation function of the corrected signal, transforms it to the repetition spectrum via FFT, and converts the repetition frequencies to distances. As a result, the method produces a distogram showing detected flutter echo distances with their corresponding amplitudes from the repetition spectrum, indicating the strength of the periodic pattern at each distance.
<p align="center">
<img src="https://github.com/user-attachments/assets/cc1efe70-d0e1-4928-9585-ed13896ef194" width="60%" alt="Room Analysis" />
</p>

**Workflow:**

1. **Bandpass Filtering**: The impulse response is filtered around a center frequency (typically 1-2 kHz, where flutter echoes are most audible) to isolate the relevant frequency content.

2. **Level-Time Representation**: The filtered signal is converted to a level-time representation (envelope in dB) to visualize the amplitude evolution.

3. **Trend Correction (Regression Method)**: The general reverberation decay is isolated using a linear regression fitted to the decay curve. This allows you to:
   - Manually adjust the regression region by setting start and end markers
   - Set the noise floor level
   - The regression line represents the general room decay, which is subtracted from the signal to isolate flutter echoes

4. **Time Windowing**: The corrected signal (L_FE) is windowed to the analysis region, typically from the regression start point to the intersection with the noise floor. This time windowing ensures that only the relevant portion of the signal (where flutter echoes occur) is analyzed, excluding early reflections and noise.

5. **Autocorrelation (ACF)**: The windowed, corrected signal is autocorrelated to detect periodic patterns that indicate flutter echoes.

6. **FFT and Distance Calculation**: The autocorrelation function is transformed to the frequency domain (FFT) to determine repetition frequencies, which are then converted to distances using the speed of sound.

7. **Peak Detection**: The resulting distogram (distance vs. amplitude) is analyzed to identify significant peaks, each representing a potential flutter echo path.

**Interaction Features:**
- Adjust regression markers and noise floor in the RT60 correction dialog
- Navigate through detected peaks using arrow keys
- View detailed pipeline visualization showing each processing step
- Assign detected distances to room dimensions

### Additional Analysis Tools

- Time-domain analysis with level evolution and parallel spectrogram display
- FFT-based spectral analysis with configurable window functions and FFT sizes
- Third-octave band impulse response analysis to identify flutter-prone frequency regions
- Audio playback and WAV export of band-limited impulse responses

## Primary References

The detection algorithm is based on:
[1] **Schanda, J., Hoffbauer, P., & Lachenmayr, G.** (2023). *Flutter Echo Detection in Room Impulse Responses*. DAGA 2023.

## Supporting Literature

- **Halmrast, T.** (2015). "Why Do Flutter Echos Always End Up Around 1-2 kHz?" In: Proceedings of the Institute of Acoustics, Vol. 37 Pt. 3, pp. 395-408.
- **Kuhl, W.** (1984). Nachhallzeiten schwach gedämpfter geschlossener Wellenzüge. Acustica 55, pp. 187-192.
- **Lorenz-Kierakiewitz, K.-H.** (2019). Flatterechos und wo sie zu finden sind. In: Fortschritte der Akustik - DAGA 2019.
- **Rindel, J. H.** (2016). Detection of Colouration in Rooms by use of Cepstrum Technique. In: Fortschritte der Akustik - DAGA 2016.
- **Yamada, Y., et al.** (2006). A simple method to detect audible echoes in room acoustical design. Applied Acoustics 67(9), pp. 835-848.

## License

[MIT](https://choosealicense.com/licenses/mit/)
