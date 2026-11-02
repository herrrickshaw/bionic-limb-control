"""
emg_decoder.py — surface-EMG analysis and decoding for prosthetic-hand control.

Two control schemes, both used in real myoelectric prostheses:

1. DirectControl  — the 2-electrode scheme in most commercial hands (Open Bionics
   Hero, Ottobock, etc.): flexor channel closes, extensor channel opens, speed is
   proportional to contraction strength, a brief co-contraction cycles grip modes.

2. PatternRecognitionDecoder — multi-channel (typically 6–8 electrodes) gesture
   classification: band-pass + notch filter -> sliding windows -> Hudgins
   time-domain features -> LDA -> majority-vote smoothing, plus a proportional
   speed estimate from normalised signal amplitude.

Input format for real recordings: CSV, one row per sample, one column per EMG
channel, optional final integer column `label` (0 = rest) and `rep` column
(repetition index, used for leakage-free train/test splits).

    python emg_decoder.py --demo                          # synthetic data
    python emg_decoder.py --csv rec.csv --fs 2000         # your recording
    python emg_decoder.py --csv rec.csv --fs 2000 --mains 60

Not a medical device. Any code that drives a physical actuator needs hardware
limits (current, end-stops, watchdog) independent of this software.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GESTURES = ["rest", "power_grip", "open_hand", "pinch", "point", "wrist_pronate"]

# --------------------------------------------------------------------------- #
# Defaults (units in the name; all tunable per call)
# --------------------------------------------------------------------------- #
DEFAULT_FS = 2000.0            # Hz, sampling rate
DEFAULT_MAINS = 50.0           # Hz, power-line frequency (50 India/EU, 60 US)
DEFAULT_BAND = (20.0, 450.0)   # Hz, sEMG pass-band
MAX_PASSBAND_FRACTION = 0.45   # upper band edge is capped at this fraction of fs
NOTCH_Q = 30                   # quality factor of each mains notch
NOTCH_HARMONICS = 3            # notch the fundamental and the next two harmonics
DEFAULT_WIN_MS = 200           # analysis window; keeps controller delay < ~300 ms
DEFAULT_STEP_MS = 50           # window hop
DEFAULT_VOTE = 5               # majority-vote depth in windows (~250 ms lag)
DEFAULT_ZC_THRESH = 0.01       # ZC/SSC dead-band, ~resting noise amplitude
N_TEST_REPS = 2                # repetitions held out for evaluation
MEAN_FREQ_NPERSEG = 256        # Welch segment length for mean frequency


def samples(fs: float, ms: float) -> int:
    """Number of samples in `ms` milliseconds at `fs` Hz (truncated)."""
    return int(fs * ms / 1000)


def gesture_name(cls: int) -> str:
    """Human-readable name for an integer class id; unknown ids print as digits."""
    return GESTURES[cls] if cls < len(GESTURES) else str(cls)


# --------------------------------------------------------------------------- #
# Pre-processing
# --------------------------------------------------------------------------- #
def design_filters(fs: float, band: tuple[float, float] = DEFAULT_BAND, mains: float = DEFAULT_MAINS) -> np.ndarray:
    """Band-pass (removes motion artefact <20 Hz) + mains notch and its harmonics, as one SOS array."""
    hi = min(band[1], MAX_PASSBAND_FRACTION * fs)
    sos = [signal.butter(4, [band[0], hi], btype="band", fs=fs, output="sos")]
    for h in range(1, NOTCH_HARMONICS + 1):
        f0 = mains * h
        if f0 < hi:
            b, a = signal.iirnotch(f0, Q=NOTCH_Q, fs=fs)
            sos.append(signal.tf2sos(b, a))
    return np.vstack(sos)


def preprocess(x: np.ndarray, fs: float, mains: float = DEFAULT_MAINS) -> np.ndarray:
    """Zero-phase filtering for offline analysis. x: (n_samples, n_channels)."""
    return signal.sosfiltfilt(design_filters(fs, mains=mains), x, axis=0)


class CausalFilter:
    """Stateful causal filter for streaming use (filtfilt is not causal)."""

    def __init__(self, fs: float, n_ch: int, mains: float = DEFAULT_MAINS):
        self.sos = design_filters(fs, mains=mains)
        self.zi = np.zeros((self.sos.shape[0], 2, n_ch))

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        y, self.zi = signal.sosfilt(self.sos, chunk, axis=0, zi=self.zi)
        return y


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def td_features(win: np.ndarray, thresh: float = DEFAULT_ZC_THRESH) -> np.ndarray:
    """Hudgins time-domain set per channel: MAV, RMS, WL, ZC, SSC (+ log-var).

    win: (n_samples, n_channels). `thresh` suppresses noise-driven ZC/SSC
    counts and should be on the order of the resting noise amplitude.
    Returns a flat vector of length 6 * n_channels, feature-major.
    """
    d = np.diff(win, axis=0)
    mav = np.mean(np.abs(win), axis=0)
    rms = np.sqrt(np.mean(win**2, axis=0))
    wl = np.sum(np.abs(d), axis=0)
    zc = np.sum((win[:-1] * win[1:] < 0) & (np.abs(d) >= thresh), axis=0)
    ssc = np.sum((d[:-1] * d[1:] < 0) & ((np.abs(d[:-1]) >= thresh) | (np.abs(d[1:]) >= thresh)), axis=0)
    logvar = np.log(np.var(win, axis=0) + 1e-12)
    return np.concatenate([mav, rms, wl, zc, ssc, logvar])


def mean_frequency(win: np.ndarray, fs: float) -> np.ndarray:
    """Per-channel spectral mean frequency (Hz) — drops as muscles fatigue."""
    f, p = signal.welch(win, fs=fs, nperseg=min(MEAN_FREQ_NPERSEG, len(win)), axis=0)
    return (f[:, None] * p).sum(0) / (p.sum(0) + 1e-12)


def sliding_windows(
    n: int, fs: float, win_ms: int = DEFAULT_WIN_MS, step_ms: int = DEFAULT_STEP_MS
) -> Iterator[tuple[int, int]]:
    """Yield (start, stop) sample indices of consecutive windows over `n` samples."""
    w, s = samples(fs, win_ms), samples(fs, step_ms)
    for start in range(0, n - w + 1, s):
        yield start, start + w


def extract(
    x: np.ndarray,
    fs: float,
    labels: np.ndarray | None = None,
    reps: np.ndarray | None = None,
    win_ms: int = DEFAULT_WIN_MS,
    step_ms: int = DEFAULT_STEP_MS,
    thresh: float = DEFAULT_ZC_THRESH,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Windowed features; each window takes the label/rep at its last sample."""
    feats, y, r = [], [], []
    for a, b in sliding_windows(len(x), fs, win_ms, step_ms):
        feats.append(td_features(x[a:b], thresh))
        if labels is not None:
            y.append(labels[b - 1])
        if reps is not None:
            r.append(reps[b - 1])
    return np.array(feats), (np.array(y) if y else None), (np.array(r) if r else None)


# --------------------------------------------------------------------------- #
# Pattern-recognition decoder
# --------------------------------------------------------------------------- #
@dataclass
class Command:
    gesture: str
    speed: float  # 0..1 proportional velocity for the actuator
    confidence: float


class PatternRecognitionDecoder:
    """Filter -> window -> Hudgins features -> StandardScaler+LDA -> majority vote."""

    def __init__(
        self,
        fs: float,
        n_ch: int,
        mains: float = DEFAULT_MAINS,
        win_ms: int = DEFAULT_WIN_MS,
        step_ms: int = DEFAULT_STEP_MS,
        vote: int = DEFAULT_VOTE,
    ):
        self.fs, self.n_ch, self.mains = fs, n_ch, mains
        self.win_ms, self.step_ms = win_ms, step_ms
        self.model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        self.thresh = DEFAULT_ZC_THRESH
        self.rest_mav: float | None = None
        self.mvc_mav: float | None = None
        self._votes: deque = deque(maxlen=vote)

    # -- training -------------------------------------------------------------
    def fit(self, x_filt: np.ndarray, labels: np.ndarray) -> PatternRecognitionDecoder:
        """Fit on filtered samples (n, n_ch) with per-sample integer labels (0 = rest)."""
        rest = x_filt[labels == 0]
        self.thresh = 3 * np.std(rest) if len(rest) else DEFAULT_ZC_THRESH
        X, y, _ = extract(x_filt, self.fs, labels, None, self.win_ms, self.step_ms, self.thresh)
        self.model.fit(X, y)
        mav = X[:, : self.n_ch].mean(1)
        self.rest_mav = np.percentile(mav[y == 0], 95) if np.any(y == 0) else 0.0
        self.mvc_mav = np.percentile(mav[y != 0], 95) if np.any(y != 0) else 1.0
        return self

    def predict_windows(self, x_filt: np.ndarray) -> np.ndarray:
        """Class id per window of an already-filtered recording."""
        X, _, _ = extract(x_filt, self.fs, None, None, self.win_ms, self.step_ms, self.thresh)
        return self.model.predict(X)

    # -- online ---------------------------------------------------------------
    def stream(self, raw_chunks: Iterable[np.ndarray]) -> Iterator[Command]:
        """Consume raw (n, n_ch) chunks; yield a Command every step_ms."""
        filt = CausalFilter(self.fs, self.n_ch, self.mains)
        w = samples(self.fs, self.win_ms)
        s = samples(self.fs, self.step_ms)
        buf = np.zeros((0, self.n_ch))
        pending = 0
        for chunk in raw_chunks:
            buf = np.vstack([buf, filt(chunk)])[-w:]
            pending += len(chunk)
            while pending >= s and len(buf) >= w:
                pending -= s
                yield self._decide(buf)

    def _decide(self, win: np.ndarray) -> Command:
        f = td_features(win, self.thresh)[None]
        proba = self.model.predict_proba(f)[0]
        cls = int(self.model.classes_[proba.argmax()])
        self._votes.append(cls)
        voted, n = Counter(self._votes).most_common(1)[0]
        mav = f[0, : self.n_ch].mean()
        speed = np.clip((mav - self.rest_mav) / (self.mvc_mav - self.rest_mav + 1e-12), 0, 1)
        if voted == 0 or speed == 0:
            return Command("rest", 0.0, float(proba.max()))
        return Command(gesture_name(voted), float(speed), n / len(self._votes))


# --------------------------------------------------------------------------- #
# Two-site direct control (commercial-hand style)
# --------------------------------------------------------------------------- #
class DirectControl:
    """Flexor -> close, extensor -> open, co-contraction -> next grip pattern."""

    GRIPS = ["power", "tripod", "pinch", "point", "key"]
    ENVELOPE_CUTOFF_HZ = 3.0

    def __init__(self, fs: float, on_thresh: float = 0.15, cocon_ms: int = 150, refractory_ms: int = 600):
        self.fs = fs
        self.on = on_thresh
        self.cocon_n = int(cocon_ms / 1000 * fs)
        self.refr_n = int(refractory_ms / 1000 * fs)
        self.grip = 0
        self._cocon = 0
        self._refr = 0
        self.lp = signal.butter(2, self.ENVELOPE_CUTOFF_HZ, fs=fs, output="sos")  # envelope low-pass
        self.zi = np.zeros((1, 2, 2))
        self.mvc = np.ones(2)

    def calibrate(self, mvc_envelope: np.ndarray) -> None:
        """mvc_envelope: (n, 2) envelope recorded during max voluntary contractions."""
        self.mvc = np.percentile(mvc_envelope, 95, axis=0)

    def envelope(self, x_filt: np.ndarray) -> np.ndarray:
        """Rectify + low-pass, normalised to MVC and clipped to [0, 1]. Stateful."""
        env, self.zi = signal.sosfilt(self.lp, np.abs(x_filt), axis=0, zi=self.zi)
        return np.clip(env / self.mvc, 0, 1)

    def step(self, e_flex: float, e_ext: float) -> tuple[str, float]:
        """One sample of normalised envelope -> (grip, velocity in [-1, 1])."""
        if self._refr:
            self._refr -= 1
            return self.GRIPS[self.grip], 0.0
        if e_flex > self.on and e_ext > self.on:
            self._cocon += 1
            if self._cocon >= self.cocon_n:
                self.grip = (self.grip + 1) % len(self.GRIPS)
                self._cocon, self._refr = 0, self.refr_n
            return self.GRIPS[self.grip], 0.0
        self._cocon = 0
        v = (max(e_flex - self.on, 0) - max(e_ext - self.on, 0)) / (1 - self.on)
        return self.GRIPS[self.grip], float(v)  # + close, - open


# --------------------------------------------------------------------------- #
# Synthetic data (for testing without hardware)
# --------------------------------------------------------------------------- #
def synthesize(
    fs: float = DEFAULT_FS,
    n_ch: int = 8,
    reps: int = 6,
    hold_s: float = 3.0,
    rest_s: float = 2.0,
    mains: float = DEFAULT_MAINS,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Band-limited noise amplitude-modulated per gesture-specific channel pattern,
    plus mains hum, low-frequency motion artefact and per-repetition variability.
    Returns (x (n, n_ch), labels (n,), reps (n,))."""
    rng = np.random.default_rng(seed)
    n_g = len(GESTURES)
    patterns = rng.uniform(0.05, 1.0, (n_g, n_ch)) ** 2
    patterns[0] = 0.02
    bp = signal.butter(4, [30, 300], btype="band", fs=fs, output="sos")
    xs, ys, rs = [], [], []
    for rep in range(reps):
        for g in rng.permutation(np.arange(1, n_g)):
            for gest, dur in ((0, rest_s), (g, hold_s)):
                n = int(dur * fs)
                ramp = np.minimum(1, np.minimum(np.arange(n), np.arange(n)[::-1]) / (0.3 * fs))
                amp = patterns[gest] * rng.uniform(0.7, 1.3, n_ch)  # rep-to-rep drift
                env = amp * (ramp[:, None] if gest else 1)
                carrier = signal.sosfilt(bp, rng.standard_normal((n, n_ch)), axis=0)
                xs.append(env * carrier)
                ys.append(np.full(n, gest))
                rs.append(np.full(n, rep))
    x = np.vstack(xs)
    t = np.arange(len(x)) / fs
    x += 0.05 * np.sin(2 * np.pi * mains * t)[:, None]          # mains hum
    x += 0.2 * np.sin(2 * np.pi * 1.2 * t)[:, None] * rng.uniform(0, 1, n_ch)  # motion
    x += 0.005 * rng.standard_normal(x.shape)                     # amplifier noise
    return x, np.concatenate(ys), np.concatenate(rs)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def split_by_rep(reps: np.ndarray, n_test: int = N_TEST_REPS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-repetitions-out masks: (train_mask, test_mask, test_rep_ids).

    The last `n_test` repetition ids are held out. Never split overlapping
    windows randomly — neighbours leak into the test set and inflate accuracy
    by 10-20 points."""
    test_reps = np.unique(reps)[-n_test:]
    te = np.isin(reps, test_reps)
    return ~te, te, test_reps


def evaluate(
    x: np.ndarray, labels: np.ndarray, reps: np.ndarray, fs: float, mains: float = DEFAULT_MAINS
) -> PatternRecognitionDecoder:
    """Train on early reps, test on the last ones; print a report and return the fitted decoder."""
    xf = preprocess(x, fs, mains)
    tr, te, test_reps = split_by_rep(reps)
    dec = PatternRecognitionDecoder(fs, x.shape[1], mains).fit(xf[tr], labels[tr])
    _, y_te, _ = extract(xf[te], fs, labels[te], None, dec.win_ms, dec.step_ms)
    pred = dec.predict_windows(xf[te])
    names = [gesture_name(i) for i in np.unique(y_te)]
    print(f"Train reps {np.setdiff1d(np.unique(reps), test_reps).tolist()}, test reps {test_reps.tolist()}")
    print(classification_report(y_te, pred, target_names=names, digits=3))
    print("Confusion matrix (rows=true):\n", confusion_matrix(y_te, pred))

    # Mean frequency on the training set shows whether filtering worked
    # (healthy sEMG: ~60-120 Hz) — a value near the mains frequency means hum leaked through.
    active = xf[tr][labels[tr] != 0][: int(fs)]
    if len(active):
        print("Mean frequency per channel (Hz):", np.round(mean_frequency(active, fs), 1))
    return dec


def demo_stream(
    dec: PatternRecognitionDecoder, x: np.ndarray, labels: np.ndarray, fs: float, chunk_ms: int = 10
) -> None:
    """Replay a recording through the causal online path, as on the device."""
    n = samples(fs, chunk_ms)
    chunks = (x[i : i + n] for i in range(0, len(x) - n + 1, n))
    step = samples(fs, dec.step_ms)
    w = samples(fs, dec.win_ms)
    hits = total = 0
    print("\nOnline replay (every 20th decision):")
    for k, cmd in enumerate(dec.stream(chunks)):
        idx = w - 1 + k * step
        truth = GESTURES[labels[min(idx, len(labels) - 1)]]
        hits += cmd.gesture == truth
        total += 1
        if k % 20 == 0:
            print(f"  t={idx / fs:6.2f}s  true={truth:<13} -> {cmd.gesture:<13} speed={cmd.speed:.2f}")
    print(f"Online agreement with labels: {hits / total:.1%} over {total} decisions "
          f"(includes transition lag from vote smoothing)")


def demo_direct(fs: float = 1000, seed: int = 1) -> None:
    """Two-site control: close, open, co-contract to switch grip."""
    rng = np.random.default_rng(seed)
    bp = signal.butter(4, [30, 300], btype="band", fs=fs, output="sos")
    seq = [(0.0, 0.0, 1.0), (0.8, 0.0, 1.5), (0.0, 0.0, 1.0), (0.0, 0.7, 1.5),
           (0.6, 0.6, 0.8), (0.0, 0.0, 1.0), (0.5, 0.0, 1.0)]
    x = np.vstack([np.array([f, e]) * signal.sosfilt(bp, rng.standard_normal((int(d * fs), 2)), axis=0)
                   for f, e, d in seq])
    dc = DirectControl(fs)
    xf = preprocess(x, fs)
    dc.calibrate(signal.sosfilt(dc.lp, np.abs(xf), axis=0) + 1e-6)
    outs = [dc.step(*v) for v in dc.envelope(xf)]
    print("\nDirect two-site control:")
    t0 = 0
    for f, e, d in seq:
        seg = outs[t0 + int(0.3 * fs): t0 + int(d * fs)]  # skip envelope rise time
        print(f"  flexor={f:.1f} extensor={e:.1f} -> grip={seg[-1][0]:<7} "
              f"mean velocity={np.mean([o[1] for o in seg]):+.2f}")
        t0 += int(d * fs)


# --------------------------------------------------------------------------- #
# I/O and CLI
# --------------------------------------------------------------------------- #
NON_CHANNEL_COLUMNS = ("label", "rep", "time", "timestamp")


def load_csv(path: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Read a recording CSV -> (x (n, n_ch), labels or None, reps or None).

    Header names are matched case-insensitively; every column not in
    NON_CHANNEL_COLUMNS is treated as an EMG channel."""
    with open(path) as fh:
        header = next(csv.reader(fh))
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    cols = {h.strip().lower(): i for i, h in enumerate(header)}
    lab = data[:, cols["label"]].astype(int) if "label" in cols else None
    rep = data[:, cols["rep"]].astype(int) if "rep" in cols else None
    ch = [i for h, i in cols.items() if h not in NON_CHANNEL_COLUMNS]
    return data[:, ch], lab, rep


def contiguous_reps(n: int, n_reps: int = 5) -> np.ndarray:
    """Fallback repetition index when a CSV has no `rep` column: contiguous fifths."""
    return np.minimum(np.arange(n) * n_reps // n, n_reps - 1)


def run_csv(path: str, fs: float, mains: float) -> None:
    """Evaluate a recorded CSV, or print signal statistics when it has no labels."""
    x, lab, rep = load_csv(path)
    if lab is None:
        xf = preprocess(x, fs, mains)
        print("No labels: per-channel RMS", np.round(np.sqrt((xf**2).mean(0)), 4))
        print("Mean frequency (Hz):", np.round(mean_frequency(xf, fs), 1))
        return
    if rep is None:
        rep = contiguous_reps(len(x))
    dec = evaluate(x, lab, rep, fs, mains)
    demo_stream(dec, x[rep == rep.max()], lab[rep == rep.max()], fs)


def run_demo(fs: float, mains: float) -> None:
    """Synthetic end-to-end run: offline evaluation, causal replay, two-site control."""
    x, lab, rep = synthesize(fs=fs, mains=mains)
    print(f"Synthetic: {x.shape[1]} channels, {len(x) / fs:.0f}s, {len(GESTURES)} classes\n")
    dec = evaluate(x, lab, rep, fs, mains)
    demo_stream(dec, x[rep == rep.max()], lab[rep == rep.max()], fs)
    demo_direct()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--csv")
    ap.add_argument("--fs", type=float, default=DEFAULT_FS)
    ap.add_argument("--mains", type=float, default=DEFAULT_MAINS, help="50 (India/EU) or 60 (US)")
    return ap


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    if a.csv:
        run_csv(a.csv, a.fs, a.mains)
    else:
        run_demo(a.fs, a.mains)


if __name__ == "__main__":
    main()
