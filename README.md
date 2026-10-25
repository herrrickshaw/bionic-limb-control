# EMG decoding for bionic-limb control

`emg_decoder.py` is a single-file reference pipeline that turns surface EMG (sEMG) from forearm muscles into prosthetic-hand commands.

```bash
pip install numpy scipy scikit-learn
python emg_decoder.py --demo                               # synthetic 8-channel data
python emg_decoder.py --csv rec.csv --fs 2000 --mains 50   # your own recording
```

CSV format: one row per sample, one column per channel, plus optional `label` (int, 0 = rest) and `rep` (repetition index) columns.

## What it does

| Stage | Choice | Why |
|---|---|---|
| Filter | 4th-order Butterworth 20–450 Hz + mains notch (×3 harmonics) | Removes motion artefact and power-line hum. Offline uses `sosfiltfilt`; on the device, `CausalFilter` keeps filter state between chunks. |
| Window | 200 ms window, 50 ms step | Keeps controller delay under the ~300 ms users can tolerate. |
| Features | Hudgins TD set (MAV, RMS, WL, ZC, SSC) + log-variance; spectral mean frequency for fatigue checks | The standard set for clinical pattern recognition. Cheap enough for a microcontroller. |
| Classifier | StandardScaler → LDA | Used by Coapt and most PR systems in the literature. Fast, and it needs little training data. |
| Post-processing | 5-window majority vote + proportional speed from normalised MAV (rest → MVC) | Stops the hand chattering between classes and gives velocity control. |
| Direct control | `DirectControl`: 2 electrodes. Flexor closes, extensor opens, co-contraction cycles the grip. | The scheme used by Open Bionics Hero and most commercial hands. |

**Evaluation trains on early repetitions and tests on later ones.** Splitting overlapping windows at random leaks neighbouring windows into the test set and inflates accuracy by 10–20 points. Many GitHub EMG projects make this mistake.

Demo results (synthetic data, so they are optimistic): 97.2% offline window accuracy across 6 classes, and 89% agreement in a causal online replay. The online gap comes mostly from the ~250 ms vote-smoothing lag at gesture transitions. Expect 75–90% offline on real amputee data, and lower across sessions unless you recalibrate.

## Similar open-source projects

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

## Suggested path to real hardware
1. Record from 2–8 electrodes with a BioAmp/OpenBCI/Myo board through BrainFlow. Log the data to CSV with `label` and `rep` columns (about 5 reps × 3 s per gesture).
2. Run `--csv` to check accuracy and mean frequency. Mean frequency should be about 60–120 Hz on real muscle. A value near 50/60 Hz means hum is leaking through the notch.
3. Feed live chunks into `PatternRecognitionDecoder.stream()` and map each `Command(gesture, speed)` to servo targets.
4. Before attaching anything to a person, enforce current limits, end-stops and a watchdog in firmware, independent of this code.

Not a medical device.
