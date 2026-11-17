# EMG decoding for bionic-limb control

A single-file Python reference pipeline that turns surface EMG (sEMG) from forearm
muscles into prosthetic-hand commands. It implements the two control schemes used in
real myoelectric prostheses — two-site direct control and multi-channel pattern
recognition — and evaluates them offline (leave-repetitions-out) and in a causal
online replay. Input is a CSV recording (or built-in synthetic data); output is a
printed classification report, confusion matrix, mean-frequency check and a stream
of `Command(gesture, speed, confidence)` objects you can map to actuator targets.

> Status: research-only prototype. The bundled demo runs on synthetic data; no real
> amputee or hardware recordings are included. Not a medical device.

## What it does
- Band-pass (20–450 Hz, 4th-order Butterworth) + mains-notch (fundamental and two harmonics) filtering, zero-phase offline (`preprocess`) and stateful causal for streaming (`CausalFilter`).
- Sliding 200 ms / 50 ms windows with the Hudgins time-domain feature set (MAV, RMS, WL, ZC, SSC) plus log-variance (`td_features`, `extract`).
- `PatternRecognitionDecoder`: StandardScaler → LDA classifier, 5-window majority vote and proportional speed from normalised MAV (rest → MVC).
- `DirectControl`: two-electrode scheme — flexor closes, extensor opens, co-contraction cycles through five grip patterns with a refractory period.
- Leakage-free evaluation (`evaluate`, `split_by_rep`): trains on early repetitions and tests on the last two; prints per-class report, confusion matrix and per-channel mean frequency.
- `synthesize`: seeded synthetic 8-channel sEMG with mains hum, motion artefact and rep-to-rep drift for testing without hardware.

## Quick start
```bash
git clone https://github.com/herrrickshaw/bionic-limb-control.git
cd bionic-limb-control
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python emg_decoder.py --demo                               # synthetic 8-channel data
python emg_decoder.py --csv rec.csv --fs 2000 --mains 50   # your own recording
```

CSV format: one row per sample, one column per channel, plus optional `label`
(int, 0 = rest) and `rep` (repetition index) columns. Columns named `time` /
`timestamp` are ignored. Without `label` the tool prints per-channel RMS and mean
frequency only; without `rep` the recording is split into five contiguous fifths.

## How it works

| Stage | Choice | Module symbol | Why |
|---|---|---|---|
| Filter | 4th-order Butterworth 20–450 Hz + mains notch (×3 harmonics) | `design_filters`, `preprocess`, `CausalFilter` | Removes motion artefact and power-line hum. Offline uses `sosfiltfilt`; on the device, `CausalFilter` keeps filter state between chunks. |
| Window | 200 ms window, 50 ms step | `sliding_windows`, `extract` | Keeps controller delay under the ~300 ms users can tolerate. |
| Features | Hudgins TD set (MAV, RMS, WL, ZC, SSC) + log-variance; spectral mean frequency for fatigue checks | `td_features`, `mean_frequency` | The standard set for clinical pattern recognition. Cheap enough for a microcontroller. |
| Classifier | StandardScaler → LDA | `PatternRecognitionDecoder.fit` | Used by Coapt and most PR systems in the literature. Fast, and it needs little training data. |
| Post-processing | 5-window majority vote + proportional speed from normalised MAV (rest → MVC) | `PatternRecognitionDecoder.stream` / `_decide` | Stops the hand chattering between classes and gives velocity control. |
| Direct control | 2 electrodes. Flexor closes, extensor opens, co-contraction cycles the grip. | `DirectControl` | The scheme used by Open Bionics Hero and most commercial hands. |

**Evaluation trains on early repetitions and tests on later ones.** Splitting
overlapping windows at random leaks neighbouring windows into the test set and
inflates accuracy by 10–20 points. Many GitHub EMG projects make this mistake.

Demo results (synthetic data, so they are optimistic; `python emg_decoder.py --demo`,
seed 0): 97.2% offline window accuracy across 6 classes, and 88.2% agreement in a
causal online replay. The online gap comes mostly from the ~250 ms vote-smoothing
lag at gesture transitions. Expect 75–90% offline on real amputee data, and lower
across sessions unless you recalibrate.

All tunable defaults (sampling rate, mains frequency, pass-band, window/step,
vote depth, zero-crossing dead-band, held-out repetitions) are named constants at
the top of `emg_decoder.py` and are also parameters of the relevant functions.

## Repository layout
| Path | Purpose |
|---|---|
| `emg_decoder.py` | The whole pipeline: filters, features, decoder, direct control, synthetic data, evaluation, CLI |
| `tests/` | pytest suite (pure-function unit tests, end-to-end characterisation on synthetic data, CLI smoke tests) |
| `requirements.txt` | Runtime dependencies (numpy, scipy, scikit-learn) |
| `pyproject.toml` | Package metadata, ruff and pytest configuration |
| `.github/workflows/ci.yml` | Lint + tests on push and pull request (Python 3.9 and 3.12) |

No data files are tracked; `*.csv` is gitignored.

## Configuration
Everything is a CLI flag or a function argument — there are no environment
variables, config files or secrets.

| Flag | Default | Meaning |
|---|---|---|
| `--demo` | off | Run on synthetic data (also the default when `--csv` is absent) |
| `--csv PATH` | — | Recording to analyse |
| `--fs` | 2000 | Sampling rate in Hz |
| `--mains` | 50 | Power-line frequency: 50 (India/EU) or 60 (US) |

## Testing
```bash
pip install pytest ruff
ruff check .
pytest -q
```

## Limitations and known issues
- Demo accuracy is measured on synthetic data and is optimistic; real sEMG, especially across sessions or on amputee data, will score lower.
- `PatternRecognitionDecoder.stream` emits one `Command` per `step_ms` of input, including `win_ms / step_ms` decisions on the very first full buffer; downstream code should treat the first ~200 ms as warm-up.
- Only LDA is implemented; there is no cross-user or cross-session adaptation.
- The code never touches hardware. Before attaching anything to a person, enforce current limits, end-stops and a watchdog in firmware, independent of this code.
- Not a medical device.

### Suggested path to real hardware
1. Record from 2–8 electrodes with a BioAmp/OpenBCI/Myo board through BrainFlow. Log the data to CSV with `label` and `rep` columns (about 5 reps × 3 s per gesture).
2. Run `--csv` to check accuracy and mean frequency. Mean frequency should be about 60–120 Hz on real muscle. A value near 50/60 Hz means hum is leaking through the notch.
3. Feed live chunks into `PatternRecognitionDecoder.stream()` and map each `Command(gesture, speed)` to servo targets.
4. Before attaching anything to a person, enforce current limits, end-stops and a watchdog in firmware, independent of this code.

## Related open-source projects

Star counts were taken from GitHub on 2026-10-25.

### Open-hardware prosthetic hands and arms
| Repo | ★ | Notes |
|---|---|---|
| [OpenBionics/Prosthetic-Hands](https://github.com/OpenBionics/Prosthetic-Hands) | 191 | OpenBionics.org (NTUA lab, *not* the UK Open Bionics company): < $200 adaptive underactuated hands. Last push 2018. |
| [pslade2/TactHand](https://github.com/pslade2/TactHand) | 27 | Tact: open-source myoelectric hand (Stanford), with CAD, electronics and firmware. |
| [MelbourneUniHRL/X-Limb](https://github.com/MelbourneUniHRL/X-Limb) | 22 | 3D-printed soft robotic, multi-articulating hand. |
| [Humanos3D/ElMedalloBionicArm](https://github.com/Humanos3D/ElMedalloBionicArm) | 22 | Low-cost open bionic arm (Colombian NGO). |
| [TheForgotensoul/3D-printed_Myo-electric-prosthetic-hand](https://github.com/TheForgotensoul/3D-printed_Myo-electric-prosthetic-hand) | 37 | Build guide for a 3D-printed myoelectric hand. |
| [openbiomedical/OBM-FABLE](https://github.com/openbiomedical/OBM-FABLE) | 12 | FABLE electromechanical hand driven by EMG. |
| [ashpande/ProstheticHand](https://github.com/ashpande/ProstheticHand) | 12 | Modular myoelectric arm electronics. |
| [Wosk1947/WML_Bionic_Arm](https://github.com/Wosk1947/WML_Bionic_Arm) | 6 | Open bionic arm project. |
| [jaundice/Bytefarm.OpenBionics.Contrib.Ada.Simulator](https://github.com/jaundice/Bytefarm.OpenBionics.Contrib.Ada.Simulator) / [MHKit/bionicohand-ada](https://github.com/MHKit/bionicohand-ada) | 7 / 1 | Simulator and firmware for the Open Bionics **Ada** hand (the company's open-source hand). |

### Myoelectric control and EMG signal processing
| Repo | ★ | Notes |
|---|---|---|
| [LibEMG/libemg](https://github.com/LibEMG/libemg) | 81 | **Best next step.** A full Python myoelectric-control library: features, classifiers, online streaming, many dataset loaders. |
| [biopatrec/biopatrec](https://github.com/biopatrec/biopatrec) | 85 | BioPatRec (MATLAB, Chalmers): a classic research platform for PR prosthetic control. |
| [agamemnonc/pyEMG](https://github.com/agamemnonc/pyEMG) | 38 | Offline and real-time myoelectric control with sEMG + IMU. |
| [tlh24/myopen](https://github.com/tlh24/myopen) | 10 | Myoelectric/neuro tools (origin of the VIE used by early OpenBionics control work). |
| [neuropsychology/NeuroKit](https://github.com/neuropsychology/NeuroKit) | 2.4k | General biosignal toolbox, including EMG activation detection. |
| [fraunhoferportugal/tsfel](https://github.com/fraunhoferportugal/tsfel) | 1.1k | 60+ time-series features if you want to go beyond the Hudgins set. |

### Acquisition hardware and SDKs
| Repo | ★ | Notes |
|---|---|---|
| [brainflow-dev/brainflow](https://github.com/brainflow-dev/brainflow) | 1.8k | One API across OpenBCI, Myo-like boards and others. Simplest way to feed live data into `PatternRecognitionDecoder.stream()`. |
| [OpenBCI/OpenBCI_GUI](https://github.com/OpenBCI/OpenBCI_GUI) | 974 | GUI for the Cyton and Ganglion boards (8–16 channel EMG/EEG). |
| [upsidedownlabs/BioAmp-EXG-Pill](https://github.com/upsidedownlabs/BioAmp-EXG-Pill) | 438 | Cheap open analog front end (Indian maker), works with Arduino. |
| [PerlinWarp/pyomyo](https://github.com/PerlinWarp/pyomyo) | 189 | Python driver for the discontinued but widely available Myo armband (8 channels at 200 Hz). |
| [feraldolim/bionic_hand](https://github.com/feraldolim/bionic_hand) / [ma581/HandController](https://github.com/ma581/HandController) | — | Myo armband → Open Bionics Ada hand (ROS / direct). |

### Datasets and deep-learning baselines
| Repo | ★ | Notes |
|---|---|---|
| [facebookresearch/generic-neuromotor-interface](https://github.com/facebookresearch/generic-neuromotor-interface) | 272 | Meta's wristband sEMG models that generalise across users (Nature 2025). |
| [facebookresearch/emg2pose](https://github.com/facebookresearch/emg2pose) | 284 | sEMG → continuous hand-joint angles (regression rather than classification). |
| [facebookresearch/emg2qwerty](https://github.com/facebookresearch/emg2qwerty) | 142 | sEMG typing dataset + baselines. |
| [jehanyang/emgbench](https://github.com/jehanyang/emgbench) | 24 | Cross-user and cross-session domain-generalisation benchmark. |
| [Lif3line/nina_helper_package_mk2](https://github.com/Lif3line/nina_helper_package_mk2), [Zeng-Jia/Ninapro-dataset-processing](https://github.com/Zeng-Jia/Ninapro-dataset-processing) | 21 / 35 | Loaders for **NinaPro**, the standard public sEMG database. DB3 includes amputees. |
| [tsagkas/sEMG-HandGestureRecognition](https://github.com/tsagkas/sEMG-HandGestureRecognition) | 134 | CNNs on NinaPro. |
| [MIC-Laboratory/Real-time-Bionic-Arm-Control-via-CNN-on-Sony-Spresense](https://github.com/MIC-Laboratory/Real-time-Bionic-Arm-Control-via-CNN-on-Sony-Spresense) | 15 | Embedded CNN running on a microcontroller. |

## Licence
No licence file yet — all rights reserved.
