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
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GESTURES = ["rest", "power_grip", "open_hand", "pinch", "point", "wrist_pronate"]


# --------------------------------------------------------------------------- #
# Pre-processing
# --------------------------------------------------------------------------- #
def design_filters(fs: float, band=(20.0, 450.0), mains=50.0):
    """Band-pass (removes motion artefact <20 Hz) + mains notch and its harmonics."""
    hi = min(band[1], 0.45 * fs)
    sos = [signal.butter(4, [band[0], hi], btype="band", fs=fs, output="sos")]
    for h in range(1, 4):
        f0 = mains * h
        if f0 < hi:
            b, a = signal.iirnotch(f0, Q=30, fs=fs)
            sos.append(signal.tf2sos(b, a))
    return np.vstack(sos)


def preprocess(x: np.ndarray, fs: float, mains=50.0) -> np.ndarray:
    """Zero-phase filtering for offline analysis. x: (n_samples, n_channels)."""
    return signal.sosfiltfilt(design_filters(fs, mains=mains), x, axis=0)


class CausalFilter:
    """Stateful causal filter for streaming use (filtfilt is not causal)."""

    def __init__(self, fs: float, n_ch: int, mains=50.0):
        self.sos = design_filters(fs, mains=mains)
        self.zi = np.zeros((self.sos.shape[0], 2, n_ch))

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        y, self.zi = signal.sosfilt(self.sos, chunk, axis=0, zi=self.zi)
        return y


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def td_features(win: np.ndarray, thresh: float = 0.01) -> np.ndarray:
    """Hudgins time-domain set per channel: MAV, RMS, WL, ZC, SSC (+ log-var).

    win: (n_samples, n_channels). `thresh` suppresses noise-driven ZC/SSC
    counts and should be on the order of the resting noise amplitude.
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
    """Per-channel spectral mean frequency — drops as muscles fatigue."""
    f, p = signal.welch(win, fs=fs, nperseg=min(256, len(win)), axis=0)
    return (f[:, None] * p).sum(0) / (p.sum(0) + 1e-12)


def sliding_windows(n: int, fs: float, win_ms=200, step_ms=50):
    """Yield (start, stop). 200 ms keeps controller delay under ~300 ms."""
    w, s = int(fs * win_ms / 1000), int(fs * step_ms / 1000)
    for start in range(0, n - w + 1, s):
        yield start, start + w


def extract(x, fs, labels=None, reps=None, win_ms=200, step_ms=50, thresh=0.01):
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
    def __init__(self, fs: float, n_ch: int, mains=50.0, win_ms=200, step_ms=50, vote=5):
        self.fs, self.n_ch, self.mains = fs, n_ch, mains
        self.win_ms, self.step_ms = win_ms, step_ms
        self.model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        self.thresh = 0.01
        self.rest_mav = None
        self.mvc_mav = None
        self._votes: deque = deque(maxlen=vote)

    # -- training -------------------------------------------------------------
    def fit(self, x_filt, labels):
        rest = x_filt[labels == 0]
        self.thresh = 3 * np.std(rest) if len(rest) else 0.01
        X, y, _ = extract(x_filt, self.fs, labels, None, self.win_ms, self.step_ms, self.thresh)
        self.model.fit(X, y)
        mav = X[:, : self.n_ch].mean(1)
        self.rest_mav = np.percentile(mav[y == 0], 95) if np.any(y == 0) else 0.0
        self.mvc_mav = np.percentile(mav[y != 0], 95) if np.any(y != 0) else 1.0
        return self

    def predict_windows(self, x_filt):
        X, _, _ = extract(x_filt, self.fs, None, None, self.win_ms, self.step_ms, self.thresh)
        return self.model.predict(X)

    # -- online ---------------------------------------------------------------
    def stream(self, raw_chunks):
        """Consume raw (n, n_ch) chunks; yield a Command every step_ms."""
        filt = CausalFilter(self.fs, self.n_ch, self.mains)
        w = int(self.fs * self.win_ms / 1000)
        s = int(self.fs * self.step_ms / 1000)
        buf = np.zeros((0, self.n_ch))
        pending = 0
        for chunk in raw_chunks:
            buf = np.vstack([buf, filt(chunk)])[-w:]
            pending += len(chunk)
            while pending >= s and len(buf) >= w:
                pending -= s
                yield self._decide(buf)

    def _decide(self, win) -> Command:
        f = td_features(win, self.thresh)[None]
        proba = self.model.predict_proba(f)[0]
        cls = int(self.model.classes_[proba.argmax()])
        self._votes.append(cls)
        voted, n = Counter(self._votes).most_common(1)[0]
        mav = f[0, : self.n_ch].mean()
        speed = np.clip((mav - self.rest_mav) / (self.mvc_mav - self.rest_mav + 1e-12), 0, 1)
        if voted == 0 or speed == 0:
            return Command("rest", 0.0, float(proba.max()))
        return Command(GESTURES[voted] if voted < len(GESTURES) else str(voted),
                       float(speed), n / len(self._votes))


# --------------------------------------------------------------------------- #
# Two-site direct control (commercial-hand style)
# --------------------------------------------------------------------------- #
class DirectControl:
    """Flexor -> close, extensor -> open, co-contraction -> next grip pattern."""

    GRIPS = ["power", "tripod", "pinch", "point", "key"]

    def __init__(self, fs, on_thresh=0.15, cocon_ms=150, refractory_ms=600):
        self.fs = fs
        self.on = on_thresh
        self.cocon_n = int(cocon_ms / 1000 * fs)
        self.refr_n = int(refractory_ms / 1000 * fs)
        self.grip = 0
        self._cocon = 0
        self._refr = 0
        self.lp = signal.butter(2, 3, fs=fs, output="sos")  # envelope low-pass
        self.zi = np.zeros((1, 2, 2))
        self.mvc = np.ones(2)

    def calibrate(self, mvc_envelope: np.ndarray):
        """mvc_envelope: (n, 2) envelope recorded during max voluntary contractions."""
        self.mvc = np.percentile(mvc_envelope, 95, axis=0)

    def envelope(self, x_filt):
        env, self.zi = signal.sosfilt(self.lp, np.abs(x_filt), axis=0, zi=self.zi)
        return np.clip(env / self.mvc, 0, 1)

    def step(self, e_flex, e_ext):
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
def synthesize(fs=2000, n_ch=8, reps=6, hold_s=3.0, rest_s=2.0, mains=50.0, seed=0):
    """Band-limited noise amplitude-modulated per gesture-specific channel pattern,
    plus mains hum, low-frequency motion artefact and per-repetition variability."""
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
def evaluate(x, labels, reps, fs, mains=50.0):
    """Leave-repetitions-out: train on early reps, test on the last ones.
    Never split overlapping windows randomly — neighbours leak into the test set
    and inflate accuracy by 10-20 points."""
    xf = preprocess(x, fs, mains)
    test_reps = np.unique(reps)[-2:]
    tr, te = ~np.isin(reps, test_reps), np.isin(reps, test_reps)
    dec = PatternRecognitionDecoder(fs, x.shape[1], mains).fit(xf[tr], labels[tr])
    _, y_te, _ = extract(xf[te], fs, labels[te], None, dec.win_ms, dec.step_ms)
    pred = dec.predict_windows(xf[te])
    names = [GESTURES[i] if i < len(GESTURES) else str(i) for i in np.unique(y_te)]
    print(f"Train reps {np.setdiff1d(np.unique(reps), test_reps).tolist()}, test reps {test_reps.tolist()}")
    print(classification_report(y_te, pred, target_names=names, digits=3))
    print("Confusion matrix (rows=true):\n", confusion_matrix(y_te, pred))

    # Mean frequency on the training set shows whether filtering worked
    # (healthy sEMG: ~60-120 Hz) — a value near the mains frequency means hum leaked through.
    active = xf[tr][labels[tr] != 0][: int(fs)]
    if len(active):
        print("Mean frequency per channel (Hz):", np.round(mean_frequency(active, fs), 1))
    return dec


def demo_stream(dec, x, labels, fs, chunk_ms=10):
    """Replay a recording through the causal online path, as on the device."""
    n = int(fs * chunk_ms / 1000)
    chunks = (x[i : i + n] for i in range(0, len(x) - n + 1, n))
    step = int(fs * dec.step_ms / 1000)
    w = int(fs * dec.win_ms / 1000)
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


def demo_direct(fs=1000, seed=1):
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


def load_csv(path):
    import csv
    with open(path) as fh:
        header = next(csv.reader(fh))
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    cols = {h.strip().lower(): i for i, h in enumerate(header)}
    lab = data[:, cols["label"]].astype(int) if "label" in cols else None
    rep = data[:, cols["rep"]].astype(int) if "rep" in cols else None
    ch = [i for h, i in cols.items() if h not in ("label", "rep", "time", "timestamp")]
    return data[:, ch], lab, rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--csv")
    ap.add_argument("--fs", type=float, default=2000)
    ap.add_argument("--mains", type=float, default=50.0, help="50 (India/EU) or 60 (US)")
    a = ap.parse_args()

    if a.csv:
        x, lab, rep = load_csv(a.csv)
        if lab is None:
            xf = preprocess(x, a.fs, a.mains)
            print("No labels: per-channel RMS", np.round(np.sqrt((xf**2).mean(0)), 4))
            print("Mean frequency (Hz):", np.round(mean_frequency(xf, a.fs), 1))
            return
        if rep is None:  # fall back to contiguous fifths
            rep = np.minimum(np.arange(len(x)) * 5 // len(x), 4)
        dec = evaluate(x, lab, rep, a.fs, a.mains)
        demo_stream(dec, x[rep == rep.max()], lab[rep == rep.max()], a.fs)
    else:
        x, lab, rep = synthesize(fs=a.fs, mains=a.mains)
        print(f"Synthetic: {x.shape[1]} channels, {len(x) / a.fs:.0f}s, {len(GESTURES)} classes\n")
        dec = evaluate(x, lab, rep, a.fs, a.mains)
        demo_stream(dec, x[rep == rep.max()], lab[rep == rep.max()], a.fs)
        demo_direct()


if __name__ == "__main__":
    main()
